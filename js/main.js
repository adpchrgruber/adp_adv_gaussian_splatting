// Splat Trainer - written with generative AI assistance (Claude Opus 5.5, Anthropic); see README, Credits and AI disclosure.
import { Trainer, P } from './splats.js';
import { IMAGES, makeTarget } from './images.js';
import { Viewer } from './viewer.js';

const $ = (id) => document.getElementById(id);
const fmt = (n) => Math.round(n).toLocaleString('en').replace(/,/g, ' ');

const DENSIFY_EVERY = 100, DENSIFY_UNTIL = 2000, STOP_AT = 5000;

const ui = {
  run: $('btnRun'), step: $('btnStep'), reset: $('btnReset'),
  image: $('image'), file: $('file'), initMode: $('initMode'), initCount: $('initCount'), res: $('res'), lr: $('lr'),
  densify: $('chkDensify'), maxSplats: $('maxSplats'), densifyInfo: $('densifyInfo'),
  size: $('size'), centres: $('chkCentres'), spread: $('spread'),
  solo: $('chkSolo'),
};

let tr = null, target = null, ownImage = null;
let running = false, photoView = 'photo', splatView = 'render';
let selected = -1, pickAt = null;
let rate = 0, rateT = performance.now(), rateIter = 0;

const photoCanvas = $('photoCanvas'), pg = photoCanvas.getContext('2d');
const viewer = new Viewer($('splatCanvas'));

// ---------- set-up ----------
for (const im of IMAGES) ui.image.add(new Option(im.name, im.id));
ui.image.add(new Option('Your own image...', 'own'));

function bindOut(input, out, f) {
  const o = $(out);
  const upd = () => { o.textContent = f(+input.value); };
  input.addEventListener('input', upd); upd();
}
bindOut(ui.initCount, 'initCountOut', (v) => v);
bindOut(ui.lr, 'lrOut', (v) => v.toFixed(2) + 'x');
bindOut(ui.maxSplats, 'maxSplatsOut', (v) => fmt(v));
bindOut(ui.size, 'sizeOut', (v) => Math.round(v * 100) + '%');
bindOut(ui.spread, 'spreadOut', (v) => v.toFixed(2));

function reset() {
  const res = +ui.res.value;
  const src = ui.image.value === 'own' && ownImage ? ownImage : (ui.image.value === 'own' ? 'still' : ui.image.value);
  target = makeTarget(src, res, res);
  tr = new Trainer(target.data, res, res, 4000);
  tr.init(+ui.initCount.value, ui.initMode.value, 1);
  tr.render(); tr.loss();
  setRunning(false);
  selected = -1; pickAt = null; showSplat();
  ui.densifyInfo.innerHTML = 'Where the error stays high, big splats are <b>split</b> in two and small ones <b>cloned</b>. Nearly invisible splats are <b>pruned</b>. Runs until step ' + DENSIFY_UNTIL + '.';
  rateIter = 0; rate = 0;
}

function setRunning(on) {
  running = on && tr.iter < STOP_AT;
  ui.run.textContent = running ? 'Pause' : (tr && tr.iter > 0 ? 'Resume' : 'Start');
}

function trainStep() {
  tr.step(+ui.lr.value);
  if (ui.densify.checked && tr.iter % DENSIFY_EVERY === 0 && tr.iter <= DENSIFY_UNTIL) {
    const d = tr.densify(+ui.maxSplats.value);
    ui.densifyInfo.innerHTML = `Step <b>${d.iter}</b>: <b>${d.split}</b> split, <b>${d.clone}</b> cloned, <b>${d.pruned}</b> pruned. Now <b>${fmt(tr.n)}</b> splats.` +
      (tr.iter === DENSIFY_UNTIL ? ' Densifying is done; the splats now only fine-tune.' : '');
    if (pickAt) selected = tr.pick(pickAt[0], pickAt[1]); // indices change, so find the splat on that spot again
  }
  if (tr.iter >= STOP_AT) setRunning(false);
}

// ---------- drawing ----------
function cssColor(name) {
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const m = v.match(/^#([0-9a-f]{6})$/i);
  const n = m ? parseInt(m[1], 16) : 0;
  return [(n >> 16 & 255) / 255, (n >> 8 & 255) / 255, (n & 255) / 255];
}

function photoFrame() {
  const cw = photoCanvas.clientWidth, ch = photoCanvas.clientHeight, s = Math.min(cw, ch) - 24;
  return { x: (cw - s) / 2, y: (ch - s) / 2, s };
}

let errCanvas = document.createElement('canvas');
function drawPhoto() {
  const dpr = Math.min(2, devicePixelRatio || 1);
  const w = Math.round(photoCanvas.clientWidth * dpr), h = Math.round(photoCanvas.clientHeight * dpr);
  if (photoCanvas.width !== w || photoCanvas.height !== h) { photoCanvas.width = w; photoCanvas.height = h; }
  pg.setTransform(dpr, 0, 0, dpr, 0, 0);
  pg.clearRect(0, 0, w, h);
  pg.imageSmoothingEnabled = false;
  const f = photoFrame();
  if (photoView === 'photo') pg.drawImage(target.canvas, f.x, f.y, f.s, f.s);
  else {
    const W = tr.w, H = tr.h;
    if (errCanvas.width !== W) { errCanvas.width = W; errCanvas.height = H; }
    const eg = errCanvas.getContext('2d'), img = eg.createImageData(W, H);
    for (let i = 0; i < W * H; i++) {
      let e = 0;
      for (let c = 0; c < 3; c++) e += Math.abs(tr.out[i * 3 + c] - target.data[i * 3 + c]);
      const v = Math.min(255, e / 3 * 4 * 255);
      img.data[i * 4] = v; img.data[i * 4 + 1] = v; img.data[i * 4 + 2] = v; img.data[i * 4 + 3] = 255;
    }
    eg.putImageData(img, 0, 0);
    pg.drawImage(errCanvas, f.x, f.y, f.s, f.s);
  }
  pg.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue('--line'); pg.lineWidth = 1;
  pg.strokeRect(f.x - 0.5, f.y - 0.5, f.s + 1, f.s + 1);
  const mark = $('photoMark');
  if (selected >= 0) {
    mark.style.display = 'block';
    mark.style.left = f.x + tr.prm[selected * P] / tr.w * f.s + 'px';
    mark.style.top = f.y + tr.prm[selected * P + 1] / tr.h * f.s + 'px';
  } else mark.style.display = 'none';
}

function drawSplats() {
  viewer.draw(tr, {
    mode: splatView, size: +ui.size.value, centres: ui.centres.checked, spread: +ui.spread.value,
    selected, solo: ui.solo.checked ? selected : -1,
    accent: cssColor('--blue-fill'), paper: cssColor('--paper'),
  });
}

function readouts() {
  const W = tr.w;
  $('readoutL').innerHTML =
    `<span>Photo <b>${W} x ${W}</b> px = <b>${fmt(W * W * 3)}</b> numbers</span>` +
    `<span>Match <b>${tr.psnr().toFixed(1)} dB</b> PSNR</span>`;
  $('readoutR').innerHTML =
    `<span>Step <b>${fmt(tr.iter)}</b>${tr.iter >= STOP_AT ? ' (done)' : ''}</span>` +
    `<span>Splats <b>${fmt(tr.n)}</b> x 9 = <b>${fmt(tr.n * 9)}</b> numbers</span>` +
    (running ? `<span><b>${rate.toFixed(0)}</b> steps/s</span>` : '');
}

function showSplat() {
  const has = selected >= 0 && selected < tr.n;
  $('splatEmpty').classList.toggle('hidden', has);
  $('splatInfo').classList.toggle('hidden', !has);
  if (!has) { ui.solo.checked = false; return; }
  const s = tr.splat(selected);
  const hex = '#' + [s.r, s.g, s.b].map((v) => Math.round(v * 255).toString(16).padStart(2, '0')).join('');
  $('swatch').style.background = hex;
  $('splatTitle').textContent = 'Splat ' + (selected + 1);
  const deg = ((s.rot * 180 / Math.PI) % 180 + 180) % 180;
  $('splatTable').innerHTML = [
    ['Position x, y', `${s.x.toFixed(1)}, ${s.y.toFixed(1)} px`],
    ['Size (1 sigma)', `${s.sx.toFixed(2)} x ${s.sy.toFixed(2)} px`],
    ['Rotation', `${deg.toFixed(0)}&deg;`],
    ['Colour', hex],
    ['Opacity', `${Math.round(s.o * 100)}%`],
    ['Depth order', `${s.depth + 1} of ${fmt(tr.n)}`],
  ].map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join('');
}

let lastInfo = 0;
function frame(now) {
  if (running) {
    const t0 = performance.now();
    do trainStep(); while (running && performance.now() - t0 < 14);
    if (now - rateT > 500) { rate = (tr.iter - rateIter) / ((now - rateT) / 1000); rateT = now; rateIter = tr.iter; }
  } else { rateT = now; rateIter = tr.iter; }
  drawPhoto(); drawSplats(); readouts();
  if (now - lastInfo > 200) { showSplat(); lastInfo = now; }
  requestAnimationFrame(frame);
}

// ---------- interaction ----------
ui.run.onclick = () => setRunning(!running);
ui.step.onclick = () => { setRunning(false); trainStep(); tr.render(); tr.loss(); };
ui.reset.onclick = reset;
for (const el of [ui.initMode, ui.initCount, ui.res]) el.addEventListener('change', reset);
ui.image.addEventListener('change', () => {
  if (ui.image.value === 'own') ui.file.click();
  else reset();
});
ui.file.addEventListener('change', () => {
  const f = ui.file.files[0];
  if (!f) return;
  const img = new Image();
  img.onload = () => { ownImage = img; ui.image.value = 'own'; reset(); };
  img.src = URL.createObjectURL(f);
});
// drop a picture anywhere
addEventListener('dragover', (e) => e.preventDefault());
addEventListener('drop', (e) => {
  e.preventDefault();
  const f = [...e.dataTransfer.files].find((x) => x.type.startsWith('image/'));
  if (!f) return;
  const img = new Image();
  img.onload = () => { ownImage = img; ui.image.value = 'own'; reset(); };
  img.src = URL.createObjectURL(f);
});

$('btnDeselect').onclick = () => { selected = -1; pickAt = null; showSplat(); };
ui.solo.addEventListener('change', () => {});

function pickFrom(x, y) {
  if (x < 0 || y < 0 || x >= tr.w || y >= tr.h) { selected = -1; pickAt = null; }
  else { pickAt = [x, y]; selected = tr.pick(x, y); }
  showSplat();
}

photoCanvas.addEventListener('click', (e) => {
  const f = photoFrame(), r = photoCanvas.getBoundingClientRect();
  pickFrom((e.clientX - r.left - f.x) / f.s * tr.w, (e.clientY - r.top - f.y) / f.s * tr.h);
});

const sc = $('splatCanvas');
let drag = null;
sc.addEventListener('pointerdown', (e) => { drag = { x: e.clientX, y: e.clientY, moved: false }; sc.setPointerCapture(e.pointerId); });
sc.addEventListener('pointermove', (e) => {
  if (!drag) return;
  const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
  if (Math.abs(dx) + Math.abs(dy) > 3) drag.moved = true;
  if (splatView === 'layers' && drag.moved) {
    viewer.yaw += dx * 0.008;
    viewer.pitch = Math.max(-1.3, Math.min(1.3, viewer.pitch + dy * 0.008));
    drag.x = e.clientX; drag.y = e.clientY;
  }
});
sc.addEventListener('pointerup', (e) => {
  if (drag && !drag.moved && splatView !== 'layers') {
    const r = sc.getBoundingClientRect();
    const [x, y] = viewer.toImage(e.clientX - r.left, e.clientY - r.top, tr.w);
    pickFrom(x, y);
  }
  drag = null;
});
sc.addEventListener('wheel', (e) => {
  if (splatView !== 'layers') return;
  e.preventDefault();
  viewer.dist = Math.max(0.8, Math.min(6, viewer.dist * Math.exp(e.deltaY * 0.001)));
}, { passive: false });

function tabs(pane, set) {
  for (const b of document.querySelectorAll(`#${pane} .tab`)) {
    b.onclick = () => {
      for (const o of document.querySelectorAll(`#${pane} .tab`)) o.classList.toggle('active', o === b);
      set(b.dataset.view);
    };
  }
}
tabs('photoPane', (v) => { photoView = v; });
tabs('splatPane', (v) => { splatView = v; $('splatStage').classList.toggle('orbit', v === 'layers'); });

addEventListener('keydown', (e) => {
  if (e.code !== 'Space' || /INPUT|SELECT|TEXTAREA/.test(e.target.tagName)) return;
  e.preventDefault();
  setRunning(!running);
});

// console hook for testing
window.splats = { get trainer() { return tr; }, reset, run: (n) => { for (let k = 0; k < n; k++) trainStep(); tr.render(); tr.loss(); } };

reset();
requestAnimationFrame(frame);
