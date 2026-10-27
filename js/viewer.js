// Splat Trainer - written with generative AI assistance (Claude Opus 5.5, Anthropic); see README, Credits and AI disclosure.
// WebGL2 splat renderer: one quad per splat, Gaussian falloff in the fragment shader,
// drawn back to front with premultiplied "over" blending: the same maths as the trainer,
// only on the GPU and at screen resolution.

const VS = `#version 300 es
layout(location=0) in vec2 corner;
layout(location=1) in vec3 iPos;
layout(location=2) in vec3 iShape;
layout(location=3) in vec4 iCol;
uniform mat4 uMVP;
uniform float uSize, uExtent, uDotPx;
uniform vec2 uView;
uniform int uMode;
out vec2 vUV; out vec4 vCol;
void main() {
  vec2 c = corner * uExtent;
  float cs = cos(iShape.z), sn = sin(iShape.z);
  vec2 s = iShape.xy * uSize;
  vec2 local = vec2(cs * c.x * s.x - sn * c.y * s.y, sn * c.x * s.x + cs * c.y * s.y);
  vec4 p = uMVP * vec4(iPos.xy + local, iPos.z, 1.0);
  if (uMode == 2) { p = uMVP * vec4(iPos, 1.0); p.xy += corner * uDotPx / uView * 2.0 * p.w; }
  vUV = c; vCol = iCol; gl_Position = p;
}`;

const FS = `#version 300 es
precision highp float;
precision highp int;
in vec2 vUV; in vec4 vCol;
uniform int uMode;
uniform vec3 uTint;
uniform float uRing, uLine;
out vec4 o;
void main() {
  float r2 = dot(vUV, vUV);
  if (uMode == 0) {            // splat
    float a = min(0.99, vCol.a * exp(-0.5 * r2));
    if (a < 1.0 / 255.0) discard;
    o = vec4(vCol.rgb * a, a);
  } else if (uMode == 1) {     // outline at uRing sigma
    float r = sqrt(r2), d = abs(r - uRing) / fwidth(r);
    float a = clamp(uLine - d, 0.0, 1.0) * (0.35 + 0.65 * vCol.a);
    if (a <= 0.0) discard;
    vec3 c = uTint.r < 0.0 ? vCol.rgb : uTint;
    o = vec4(c * a, a);
  } else {                     // centre dot
    float r = sqrt(r2), aa = fwidth(r);
    float a = 1.0 - smoothstep(1.0 - aa, 1.0, r);
    if (a <= 0.0) discard;
    vec3 c = r < 0.6 ? uTint : vec3(0.0);
    o = vec4(c * a, a);
  }
}`;

const STRIDE = 10; // x y z, sx sy rot, r g b o

export class Viewer {
  constructor(canvas) {
    this.canvas = canvas;
    const gl = this.gl = canvas.getContext('webgl2', { premultipliedAlpha: true, antialias: false, preserveDrawingBuffer: true });
    if (!gl) throw new Error('WebGL2 is not available');
    const sh = (type, src) => {
      const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
      return s;
    };
    const pr = this.prog = gl.createProgram();
    gl.attachShader(pr, sh(gl.VERTEX_SHADER, VS)); gl.attachShader(pr, sh(gl.FRAGMENT_SHADER, FS));
    gl.linkProgram(pr);
    if (!gl.getProgramParameter(pr, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(pr));
    this.u = {};
    for (const n of ['uMVP', 'uSize', 'uExtent', 'uDotPx', 'uView', 'uMode', 'uTint', 'uRing', 'uLine']) this.u[n] = gl.getUniformLocation(pr, n);
    this.vao = gl.createVertexArray(); gl.bindVertexArray(this.vao);
    const quad = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, quad);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    this.inst = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, this.inst);
    const B = STRIDE * 4;
    gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 3, gl.FLOAT, false, B, 0); gl.vertexAttribDivisor(1, 1);
    gl.enableVertexAttribArray(2); gl.vertexAttribPointer(2, 3, gl.FLOAT, false, B, 12); gl.vertexAttribDivisor(2, 1);
    gl.enableVertexAttribArray(3); gl.vertexAttribPointer(3, 4, gl.FLOAT, false, B, 24); gl.vertexAttribDivisor(3, 1);
    this.data = new Float32Array(0);
    this.yaw = -0.45; this.pitch = 0.22; this.dist = 2.4;
  }

  resize() {
    const c = this.canvas, dpr = Math.min(2, devicePixelRatio || 1);
    const w = Math.max(1, Math.round(c.clientWidth * dpr)), h = Math.max(1, Math.round(c.clientHeight * dpr));
    if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
    this.dpr = dpr;
  }

  // where the image square sits on the canvas, in CSS pixels
  frame() {
    const cw = this.canvas.clientWidth, ch = this.canvas.clientHeight, s = Math.min(cw, ch) - 24;
    return { x: (cw - s) / 2, y: (ch - s) / 2, s };
  }

  matrix(mode, W) {
    const cw = this.canvas.width, ch = this.canvas.height;
    if (mode !== 'layers') {
      const f = this.frame(), d = this.dpr, s = f.s * d / W;
      // image px -> canvas px -> clip
      const sx = 2 * s / cw, sy = -2 * s / ch;
      const tx = 2 * f.x * d / cw - 1, ty = 1 - 2 * f.y * d / ch;
      return new Float32Array([sx, 0, 0, 0, 0, sy, 0, 0, 0, 0, 0, 0, tx, ty, 0, 1]);
    }
    // perspective orbit around the image centre; image units scaled to 1
    const k = 1 / W, cy = Math.cos(this.yaw), sy = Math.sin(this.yaw), cp = Math.cos(this.pitch), sp = Math.sin(this.pitch);
    // model: centre, flip y up, scale
    const M = [k, 0, 0, 0, 0, -k, 0, 0, 0, 0, k, 0, -0.5, 0.5, 0, 1];
    // view: rotate yaw (around y), pitch (around x), push back
    const R = [cy, sy * sp, -sy * cp, 0, 0, cp, sp, 0, sy, -cy * sp, cy * cp, 0, 0, 0, -this.dist, 1];
    // field of view fitted to the shorter side of the pane
    const f = 1 / Math.tan(0.4), a = cw / ch, n = 0.05, fa = 50;
    const fx = a >= 1 ? f / a : f, fy = a >= 1 ? f : f * a;
    const Pm = [fx, 0, 0, 0, 0, fy, 0, 0, 0, 0, (fa + n) / (n - fa), -1, 0, 0, 2 * fa * n / (n - fa), 0];
    return mul(Pm, mul(R, M));
  }

  // splats: array of {x,y,z, sx,sy,rot, r,g,b,o} in back-to-front order is built here
  draw(tr, opt) {
    const gl = this.gl, W = tr.w, n = tr.n;
    this.resize();
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.disable(gl.SCISSOR_TEST);
    gl.clearColor(opt.paper[0], opt.paper[1], opt.paper[2], 1); gl.clear(gl.COLOR_BUFFER_BIT);
    if (opt.mode !== 'layers') {
      // flat views: a black picture square (the training background), splats clipped to it
      const f = this.frame(), d = this.dpr;
      gl.enable(gl.SCISSOR_TEST);
      gl.scissor(Math.round(f.x * d), Math.round(this.canvas.height - (f.y + f.s) * d), Math.round(f.s * d), Math.round(f.s * d));
      gl.clearColor(0, 0, 0, 1); gl.clear(gl.COLOR_BUFFER_BIT);
    }
    const mvp = this.matrix(opt.mode, W);
    if (this.data.length < n * STRIDE) this.data = new Float32Array(n * STRIDE * 2);
    const d = this.data, prm = tr.prm, P = 9;
    const sig = (x) => 1 / (1 + Math.exp(-x));
    // depth for the exploded view: rank in the sort order, spread through the volume
    const rank = new Float32Array(tr.max);
    for (let k = 0; k < n; k++) rank[tr.order[k]] = k / Math.max(1, n - 1);
    let ids = [];
    for (let k = n - 1; k >= 0; k--) ids.push(tr.order[k]); // back to front
    if (opt.solo >= 0) ids = ids.filter((i) => i === opt.solo);
    const zOf = (i) => (0.5 - rank[i]) * opt.spread * W; // front splats nearer the camera
    if (opt.mode === 'layers') {
      const v = (i) => { const x = prm[i * P], y = prm[i * P + 1], z = zOf(i); return mvp[3] * x + mvp[7] * y + mvp[11] * z + mvp[15]; };
      const w = new Float32Array(tr.max); for (const i of ids) w[i] = v(i);
      ids.sort((a, b) => w[b] - w[a]);
    }
    let q = 0;
    for (const i of ids) {
      const o = i * P;
      d[q] = prm[o]; d[q + 1] = prm[o + 1]; d[q + 2] = opt.mode === 'layers' ? zOf(i) : 0;
      // axes of the widened covariance, the shape the trainer actually sees
      const [xx, xy, yy] = tr.cov(i), mid = (xx + yy) / 2, r = Math.hypot((xx - yy) / 2, xy);
      d[q + 3] = Math.sqrt(mid + r); d[q + 4] = Math.sqrt(Math.max(1e-6, mid - r)); d[q + 5] = 0.5 * Math.atan2(2 * xy, xx - yy);
      d[q + 6] = sig(prm[o + 5]); d[q + 7] = sig(prm[o + 6]); d[q + 8] = sig(prm[o + 7]); d[q + 9] = sig(prm[o + 8]);
      q += STRIDE;
    }
    const count = ids.length;
    gl.useProgram(this.prog); gl.bindVertexArray(this.vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.inst); gl.bufferData(gl.ARRAY_BUFFER, d.subarray(0, q), gl.DYNAMIC_DRAW);
    gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    gl.uniformMatrix4fv(this.u.uMVP, false, mvp);
    gl.uniform2f(this.u.uView, this.canvas.width, this.canvas.height);
    gl.uniform1f(this.u.uSize, opt.size);
    gl.uniform1f(this.u.uLine, 1.2 * this.dpr);
    if (opt.mode === 'ellipses') {
      gl.uniform1i(this.u.uMode, 1); gl.uniform1f(this.u.uExtent, 2.3); gl.uniform1f(this.u.uRing, 2.0);
      gl.uniform3f(this.u.uTint, -1, 0, 0);
    } else {
      gl.uniform1i(this.u.uMode, 0); gl.uniform1f(this.u.uExtent, 3.0);
    }
    gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, count);
    if (opt.centres) {
      gl.uniform1i(this.u.uMode, 2); gl.uniform1f(this.u.uExtent, 1.0); gl.uniform1f(this.u.uDotPx, 2.2 * this.dpr);
      gl.uniform3f(this.u.uTint, 1, 1, 1);
      gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, count);
    }
    if (opt.selected >= 0 && opt.selected < n) {
      // highlight ring of the inspected splat, redrawn alone on top
      const j = ids.indexOf(opt.selected);
      if (j >= 0) {
        gl.uniform1i(this.u.uMode, 1); gl.uniform1f(this.u.uExtent, 2.3); gl.uniform1f(this.u.uRing, 2.0);
        gl.uniform1f(this.u.uLine, 2.4 * this.dpr);
        gl.uniform3f(this.u.uTint, opt.accent[0], opt.accent[1], opt.accent[2]);
        gl.bufferData(gl.ARRAY_BUFFER, d.subarray(j * STRIDE, j * STRIDE + STRIDE), gl.DYNAMIC_DRAW);
        gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, 1);
      }
    }
  }

  // CSS px on the canvas -> image px (flat views only)
  toImage(cx, cy, W) {
    const f = this.frame();
    return [(cx - f.x) / f.s * W, (cy - f.y) / f.s * W];
  }
}

function mul(a, b) {
  const r = new Float32Array(16);
  for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) {
    let s = 0; for (let k = 0; k < 4; k++) s += a[k * 4 + j] * b[i * 4 + k];
    r[i * 4 + j] = s;
  }
  return r;
}
