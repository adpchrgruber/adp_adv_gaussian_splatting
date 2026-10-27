// Splat Trainer - written with generative AI assistance (Claude Opus 5.5, Anthropic); see README, Credits and AI disclosure.
// A tiny Gaussian splatting trainer in 2D, on the CPU.
// Same pipeline as 3D Gaussian Splatting (Kerbl et al. 2023), minus the third
// dimension: splats are projected (here: already flat), sorted by depth, blended
// front to back, compared with the photo, and nudged by gradient descent (Adam).

export const P = 9; // numbers per splat: x, y, log sx, log sy, rotation, r, g, b, opacity (colour + opacity as logits)
const TS = 8;       // tile size in pixels
export const DILATE = 0.3; // px² added to every splat: the low-pass filter of the paper, so no splat falls between pixels
const MIN_A = 1 / 255, MAX_A = 0.99, T_EPS = 1e-4;
const LR = [0.25, 0.25, 0.03, 0.03, 0.03, 0.06, 0.06, 0.06, 0.06]; // Adam step per parameter
const B1 = 0.9, B2 = 0.999;

const sig = (x) => 1 / (1 + Math.exp(-x));
const logit = (p) => { p = Math.min(0.995, Math.max(0.005, p)); return Math.log(p / (1 - p)); };

export function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function gauss(rand) {
  return Math.sqrt(-2 * Math.log(rand() + 1e-12)) * Math.cos(2 * Math.PI * rand());
}

export class Trainer {
  constructor(target, w, h, max = 4000) {
    this.target = target; this.w = w; this.h = h; this.max = max;
    this.bg = 0;
    this.n = 0; this.iter = 0; this.t = 0;
    this.prm = new Float32Array(max * P);
    this.m = new Float32Array(max * P);
    this.v = new Float32Array(max * P);
    this.g = new Float64Array(max * P);
    this.z = new Float32Array(max);      // depth: small = in front
    this.order = new Int32Array(max);    // splat indices, front to back
    this.gradSum = new Float32Array(max); this.gradCnt = new Uint16Array(max);
    // per-step preprocessed values
    this.pA = new Float32Array(max); this.pB = new Float32Array(max); this.pC = new Float32Array(max);
    this.pc = new Float32Array(max * 3); this.po = new Float32Array(max);
    this.tx = Math.ceil(w / TS); this.ty = Math.ceil(h / TS);
    this.tileStart = new Int32Array(this.tx * this.ty + 1);
    this.tileList = new Int32Array(0);
    this.out = new Float32Array(w * h * 3);
    this.Tf = new Float32Array(w * h);
    this.last = new Int32Array(w * h);
    this.dpix = new Float32Array(w * h * 3);
    this.mse = 1; this.lastDensify = null;
  }

  // ---------- set-up ----------
  init(count, mode = 'random', seed = 1) {
    const { w, h, target } = this, rand = rng(seed);
    count = Math.min(count, this.max);
    let cdf = null;
    if (mode === 'points') {
      // like structure-from-motion: start points where the photo has detail (edges, corners)
      cdf = new Float64Array(w * h); let acc = 0;
      for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
        const i = y * w + x, xr = Math.min(x + 1, w - 1), yd = Math.min(y + 1, h - 1);
        let e = 0;
        for (let c = 0; c < 3; c++) {
          e += Math.abs(target[(y * w + xr) * 3 + c] - target[i * 3 + c]) + Math.abs(target[(yd * w + x) * 3 + c] - target[i * 3 + c]);
        }
        acc += e * e + 1e-4; cdf[i] = acc;
      }
    }
    const s0 = Math.max(0.7, 0.5 * Math.sqrt(w * h / count));
    this.n = 0;
    for (let k = 0; k < count; k++) {
      let x, y;
      if (cdf) {
        const r = rand() * cdf[cdf.length - 1];
        let lo = 0, hi = cdf.length - 1;
        while (lo < hi) { const mid = (lo + hi) >> 1; if (cdf[mid] < r) lo = mid + 1; else hi = mid; }
        x = (lo % w) + rand(); y = Math.floor(lo / w) + rand();
      } else { x = rand() * w; y = rand() * h; }
      const pi = (Math.min(h - 1, y | 0) * w + Math.min(w - 1, x | 0)) * 3;
      this.add([x, y, Math.log(s0), Math.log(s0), rand() * Math.PI,
        logit(target[pi]), logit(target[pi + 1]), logit(target[pi + 2]), 0], rand());
    }
    this.iter = 0; this.t = 0; this.lastDensify = null;
    this.sort();
  }

  add(p, z, mFrom = -1) {
    const i = this.n++;
    for (let k = 0; k < P; k++) {
      this.prm[i * P + k] = p[k];
      this.m[i * P + k] = mFrom >= 0 ? this.m[mFrom * P + k] : 0;
      this.v[i * P + k] = mFrom >= 0 ? this.v[mFrom * P + k] : 0;
    }
    this.z[i] = z; this.gradSum[i] = 0; this.gradCnt[i] = 0;
    return i;
  }

  sort() {
    const idx = Array.from({ length: this.n }, (_, i) => i);
    idx.sort((a, b) => this.z[a] - this.z[b]);
    this.order.set(idx);
  }

  // ---------- render ----------
  preprocess() {
    const { n, prm, w, h, tx, ty } = this;
    const counts = new Int32Array(tx * ty + 1);
    const box = new Int32Array(n * 4);
    for (let i = 0; i < n; i++) {
      const o = i * P, sx = Math.exp(prm[o + 2]), sy = Math.exp(prm[o + 3]);
      const c = Math.cos(prm[o + 4]), s = Math.sin(prm[o + 4]);
      // covariance R S S Rt, widened by DILATE, then inverted (the "conic")
      const sxx = c * c * sx * sx + s * s * sy * sy + DILATE, syy = s * s * sx * sx + c * c * sy * sy + DILATE;
      const sxy = c * s * (sx * sx - sy * sy), det = sxx * syy - sxy * sxy;
      this.pA[i] = syy / det; this.pB[i] = -sxy / det; this.pC[i] = sxx / det;
      for (let k = 0; k < 3; k++) this.pc[i * 3 + k] = sig(prm[o + 5 + k]);
      this.po[i] = sig(prm[o + 8]);
      // 3 sigma box, axis aligned
      const ex = 3 * Math.sqrt(sxx), ey = 3 * Math.sqrt(syy);
      const x0 = Math.max(0, Math.floor((prm[o] - ex) / TS)), x1 = Math.min(tx - 1, Math.floor((prm[o] + ex) / TS));
      const y0 = Math.max(0, Math.floor((prm[o + 1] - ey) / TS)), y1 = Math.min(ty - 1, Math.floor((prm[o + 1] + ey) / TS));
      box[i * 4] = x0; box[i * 4 + 1] = x1; box[i * 4 + 2] = y0; box[i * 4 + 3] = y1;
      for (let b = y0; b <= y1; b++) for (let a = x0; a <= x1; a++) counts[b * tx + a]++;
    }
    const start = this.tileStart; start[0] = 0;
    for (let t = 0; t < tx * ty; t++) start[t + 1] = start[t] + counts[t];
    if (this.tileList.length < start[tx * ty]) this.tileList = new Int32Array(start[tx * ty] * 2);
    const fill = start.slice(0, tx * ty);
    // walk in depth order, so every tile list comes out sorted front to back
    for (let k = 0; k < n; k++) {
      const i = this.order[k];
      const x0 = box[i * 4], x1 = box[i * 4 + 1], y0 = box[i * 4 + 2], y1 = box[i * 4 + 3];
      for (let b = y0; b <= y1; b++) for (let a = x0; a <= x1; a++) this.tileList[fill[b * tx + a]++] = i;
    }
    // each tile's splats packed side by side (x, y, A, B, C, opacity, r, g, b): the pixel loops read them in order
    const E = start[tx * ty];
    if (!this.td || this.td.length < E * 9) { this.td = new Float32Array(E * 18); this.tg = new Float64Array(E * 18); }
    const td = this.td;
    for (let k = 0; k < E; k++) {
      const i = this.tileList[k], q = k * 9;
      td[q] = prm[i * P]; td[q + 1] = prm[i * P + 1];
      td[q + 2] = this.pA[i]; td[q + 3] = this.pB[i]; td[q + 4] = this.pC[i]; td[q + 5] = this.po[i];
      td[q + 6] = this.pc[i * 3]; td[q + 7] = this.pc[i * 3 + 1]; td[q + 8] = this.pc[i * 3 + 2];
    }
    this.entries = E;
    void w; void h;
  }

  forward() {
    const { w, h, tx, ty, td, tileStart, out, Tf, last, bg } = this;
    for (let t = 0; t < tx * ty; t++) {
      const s0 = tileStart[t], s1 = tileStart[t + 1];
      const X0 = (t % tx) * TS, Y0 = Math.floor(t / tx) * TS;
      for (let py = Y0; py < Math.min(h, Y0 + TS); py++) for (let px = X0; px < Math.min(w, X0 + TS); px++) {
        const fx = px + 0.5, fy = py + 0.5;
        let T = 1, r = 0, g = 0, b = 0, lc = 0;
        for (let k = s0; k < s1; k++) {
          const q = k * 9, dx = td[q] - fx, dy = td[q + 1] - fy;
          const power = -0.5 * (td[q + 2] * dx * dx + td[q + 4] * dy * dy) - td[q + 3] * dx * dy;
          if (power < -5.6) continue; // alpha < 1/255 for any opacity
          const alpha = Math.min(MAX_A, td[q + 5] * Math.exp(power));
          if (alpha < MIN_A) continue;
          const Tn = T * (1 - alpha);
          if (Tn < T_EPS) break;
          const wgt = alpha * T;
          r += td[q + 6] * wgt; g += td[q + 7] * wgt; b += td[q + 8] * wgt;
          T = Tn; lc = k - s0 + 1;
        }
        const p = py * w + px;
        out[p * 3] = r + T * bg; out[p * 3 + 1] = g + T * bg; out[p * 3 + 2] = b + T * bg;
        Tf[p] = T; last[p] = lc;
      }
    }
  }

  render() { this.preprocess(); this.forward(); return this.out; }

  // ---------- train ----------
  loss() {
    const { out, target, dpix } = this, N = out.length;
    let se = 0;
    for (let k = 0; k < N; k++) { const d = out[k] - target[k]; se += d * d; dpix[k] = 2 * d; }
    this.mse = se / N;
    return this.mse;
  }

  backward() {
    const { w, h, tx, ty, prm, td, tg, tileStart, tileList, Tf, last, dpix, bg, g } = this;
    tg.fill(0, 0, this.entries * 9);
    for (let t = 0; t < tx * ty; t++) {
      const s0 = tileStart[t];
      const X0 = (t % tx) * TS, Y0 = Math.floor(t / tx) * TS;
      for (let py = Y0; py < Math.min(h, Y0 + TS); py++) for (let px = X0; px < Math.min(w, X0 + TS); px++) {
        const p = py * w + px, fx = px + 0.5, fy = py + 0.5;
        const dR = dpix[p * 3], dG = dpix[p * 3 + 1], dB = dpix[p * 3 + 2];
        const TF = Tf[p], bgDot = bg * (dR + dG + dB);
        let T = TF, aR = 0, aG = 0, aB = 0, lA = 0, lR = 0, lG = 0, lB = 0;
        for (let k = s0 + last[p] - 1; k >= s0; k--) {
          const q = k * 9, dx = td[q] - fx, dy = td[q + 1] - fy;
          const A = td[q + 2], B = td[q + 3], C = td[q + 4];
          const power = -0.5 * (A * dx * dx + C * dy * dy) - B * dx * dy;
          if (power < -5.6) continue;
          const G = Math.exp(power), alpha = Math.min(MAX_A, td[q + 5] * G);
          if (alpha < MIN_A) continue;
          T = T / (1 - alpha);
          const wgt = alpha * T, cR = td[q + 6], cG = td[q + 7], cB = td[q + 8];
          // colour of everything behind this splat, as seen from here
          aR = lA * lR + (1 - lA) * aR; aG = lA * lG + (1 - lA) * aG; aB = lA * lB + (1 - lA) * aB;
          lR = cR; lG = cG; lB = cB; lA = alpha;
          tg[q + 5] += wgt * dR; tg[q + 6] += wgt * dG; tg[q + 7] += wgt * dB;
          let dA = ((cR - aR) * dR + (cG - aG) * dG + (cB - aB) * dB) * T;
          dA += (-TF / (1 - alpha)) * bgDot;
          const dPow = dA * alpha; // alpha = o * exp(power)
          tg[q + 8] += dA * G;
          tg[q] += -dPow * (A * dx + B * dy);
          tg[q + 1] += -dPow * (B * dx + C * dy);
          // conic entries, stashed in the scale/rotation slots and resolved below
          tg[q + 2] += dPow * -0.5 * dx * dx;
          tg[q + 3] += dPow * -dx * dy;
          tg[q + 4] += dPow * -0.5 * dy * dy;
        }
      }
    }
    // sum the tile entries back into their splats
    g.fill(0, 0, this.n * P);
    for (let k = 0; k < this.entries; k++) {
      const o = tileList[k] * P, q = k * 9;
      for (let j = 0; j < 9; j++) g[o + j] += tg[q + j];
    }
    const { pc, po } = this;
    // chain rule: conic (A, B, C) -> log scales + rotation; logits for colour and opacity
    for (let i = 0; i < this.n; i++) {
      const o = i * P, gA = g[o + 2], gB = g[o + 3], gC = g[o + 4];
      const sx = Math.exp(prm[o + 2]), sy = Math.exp(prm[o + 3]);
      const c = Math.cos(prm[o + 4]), s = Math.sin(prm[o + 4]);
      // through the inverse: dL/dSigma = -Q (dL/dQ) Q, with Q the conic
      const A = this.pA[i], B = this.pB[i], C = this.pC[i], hB = gB / 2;
      const m00 = gA * A + hB * B, m01 = gA * B + hB * C, m10 = hB * A + gC * B, m11 = hB * B + gC * C; // (dL/dQ) Q
      const gxx = -(A * m00 + B * m10), gyy = -(B * m01 + C * m11), gxy = -2 * (A * m01 + B * m11);
      const a = sx * sx, b = sy * sy;
      const dA2 = gxx * c * c + gyy * s * s + gxy * c * s; // d/d(sx²)
      const dB2 = gxx * s * s + gyy * c * c - gxy * c * s; // d/d(sy²)
      g[o + 4] = gxx * 2 * c * s * (b - a) + gyy * 2 * c * s * (a - b) + gxy * (c * c - s * s) * (a - b);
      g[o + 2] = dA2 * 2 * a; g[o + 3] = dB2 * 2 * b;
      for (let k = 0; k < 3; k++) { const cc = pc[i * 3 + k]; g[o + 5 + k] *= cc * (1 - cc); }
      g[o + 8] *= po[i] * (1 - po[i]);
      const gn = Math.hypot(g[o], g[o + 1]);
      if (gn > 0) { this.gradSum[i] += gn; this.gradCnt[i]++; }
    }
  }

  adam(lrScale = 1) {
    const { n, prm, m, v, g } = this;
    this.t++;
    const bc1 = 1 - Math.pow(B1, this.t), bc2 = 1 - Math.pow(B2, this.t);
    // positions settle down over time, like the exponential schedule of the paper
    const posDecay = Math.max(0.1, Math.pow(0.1, this.iter / 3000));
    for (let i = 0; i < n; i++) for (let k = 0; k < P; k++) {
      const j = i * P + k, gj = g[j];
      m[j] = B1 * m[j] + (1 - B1) * gj;
      v[j] = B2 * v[j] + (1 - B2) * gj * gj;
      const lr = LR[k] * lrScale * (k < 2 ? posDecay : 1);
      prm[j] -= lr * (m[j] / bc1) / (Math.sqrt(v[j] / bc2) + 1e-8);
    }
    for (let i = 0; i < n; i++) {
      const o = i * P;
      prm[o + 2] = Math.min(Math.max(prm[o + 2], -1.2), Math.log(this.w / 2));
      prm[o + 3] = Math.min(Math.max(prm[o + 3], -1.2), Math.log(this.w / 2));
    }
  }

  step(lrScale = 1) {
    this.preprocess();
    this.forward();
    this.loss();
    this.backward();
    this.adam(lrScale);
    this.iter++;
    return this.mse;
  }

  psnr() { return -10 * Math.log10(Math.max(this.mse, 1e-10)); }

  // ---------- adaptive density control ----------
  // split big splats and clone small ones where the position gradient stays high; prune faint ones
  densify(maxSplats, thresh = 0.004, seed = this.iter) {
    const rand = rng(seed * 7919 + 1);
    const { prm } = this, n0 = this.n, big = 0.012 * Math.max(this.w, this.h);
    const keep = new Uint8Array(n0).fill(1);
    let split = 0, clone = 0, pruned = 0;
    const cand = [];
    for (let i = 0; i < n0; i++) {
      const avg = this.gradCnt[i] ? this.gradSum[i] / this.gradCnt[i] : 0;
      if (avg > thresh) cand.push([avg, i]);
    }
    cand.sort((a, b) => b[0] - a[0]);
    // grow by at most a third per round, so the splats can settle in between
    let room = Math.min(Math.max(0, maxSplats - n0), Math.ceil(n0 / 3));
    for (const [, i] of cand) {
      if (room <= 0) break;
      const o = i * P, sx = Math.exp(prm[o + 2]), sy = Math.exp(prm[o + 3]);
      const p = Array.from(prm.subarray(o, o + P));
      if (Math.max(sx, sy) > big) {
        // split: two smaller splats sampled inside the old one
        const c = Math.cos(p[4]), s = Math.sin(p[4]);
        for (let k = 0; k < 2; k++) {
          const a = gauss(rand) * sx, b = gauss(rand) * sy;
          const q = p.slice();
          q[0] += c * a - s * b; q[1] += s * a + c * b;
          q[2] -= Math.log(1.6); q[3] -= Math.log(1.6);
          this.add(q, this.z[i] + (rand() - 0.5) * 1e-3, i);
        }
        keep[i] = 0; split++; room -= 1;
      } else {
        // clone: a copy that the next steps pull apart
        this.add(p, this.z[i] + (rand() - 0.5) * 1e-3, i);
        clone++; room -= 1;
      }
    }
    for (let i = 0; i < n0; i++) if (keep[i] && sig(prm[i * P + 8]) < 0.01) { keep[i] = 0; pruned++; }
    // compact
    let j = 0;
    for (let i = 0; i < this.n; i++) {
      if (i < n0 && !keep[i]) continue;
      if (i !== j) {
        this.prm.copyWithin(j * P, i * P, i * P + P);
        this.m.copyWithin(j * P, i * P, i * P + P);
        this.v.copyWithin(j * P, i * P, i * P + P);
        this.z[j] = this.z[i];
      }
      j++;
    }
    this.n = j;
    this.gradSum.fill(0); this.gradCnt.fill(0);
    this.sort();
    this.lastDensify = { split, clone, pruned, iter: this.iter };
    return this.lastDensify;
  }

  // the splat that contributes most to a point (front to back blending weight)
  pick(x, y) {
    const { prm, n } = this;
    let T = 1, best = -1, bw = 0;
    for (let k = 0; k < n; k++) {
      const i = this.order[k], o = i * P;
      const [sxx, sxy, syy] = this.cov(i), det = sxx * syy - sxy * sxy;
      const dx = x - prm[o], dy = y - prm[o + 1];
      const q = (syy * dx * dx - 2 * sxy * dx * dy + sxx * dy * dy) / det;
      const alpha = Math.min(MAX_A, sig(prm[o + 8]) * Math.exp(-0.5 * q));
      if (alpha < MIN_A) continue;
      const wgt = alpha * T;
      if (wgt > bw) { bw = wgt; best = i; }
      T *= 1 - alpha;
      if (T < T_EPS) break;
    }
    return best;
  }

  // widened covariance [xx, xy, yy] of splat i
  cov(i) {
    const o = i * P, p = this.prm, sx = Math.exp(p[o + 2]), sy = Math.exp(p[o + 3]);
    const c = Math.cos(p[o + 4]), s = Math.sin(p[o + 4]);
    return [c * c * sx * sx + s * s * sy * sy + DILATE, c * s * (sx * sx - sy * sy), s * s * sx * sx + c * c * sy * sy + DILATE];
  }

  splat(i) {
    const o = i * P, p = this.prm;
    return {
      x: p[o], y: p[o + 1], sx: Math.exp(p[o + 2]), sy: Math.exp(p[o + 3]), rot: p[o + 4],
      r: sig(p[o + 5]), g: sig(p[o + 6]), b: sig(p[o + 7]), o: sig(p[o + 8]),
      depth: this.order.indexOf(i), z: this.z[i],
    };
  }
}
