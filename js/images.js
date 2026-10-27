// Splat Trainer - written with generative AI assistance (Claude Opus 5.5, Anthropic); see README, Credits and AI disclosure.
// Procedural "photos" to train on (drawn on a canvas, so no image files or licences),
// plus loading your own picture.

export const IMAGES = [
  { id: 'still', name: 'Still life' },
  { id: 'landscape', name: 'Landscape' },
  { id: 'letters', name: 'Letters' },
  { id: 'stripes', name: 'Stripes (hard)' },
];

const S = 512; // draw size; scaled down to the training size afterwards

function stillLife(g) {
  let gr = g.createLinearGradient(0, 0, 0, S * 0.62);
  gr.addColorStop(0, '#d9cbb4'); gr.addColorStop(1, '#a8957a');
  g.fillStyle = gr; g.fillRect(0, 0, S, S * 0.62);
  gr = g.createLinearGradient(0, S * 0.62, 0, S);
  gr.addColorStop(0, '#7a4a2a'); gr.addColorStop(1, '#4a2a16');
  g.fillStyle = gr; g.fillRect(0, S * 0.62, S, S * 0.38);
  g.strokeStyle = 'rgba(0,0,0,.18)'; g.lineWidth = 3;
  for (let k = 0; k < 6; k++) { g.beginPath(); g.moveTo(0, S * (0.68 + k * 0.055)); g.bezierCurveTo(S * .3, S * (0.66 + k * 0.055), S * .7, S * (0.71 + k * 0.055), S, S * (0.68 + k * 0.055)); g.stroke(); }
  // window light
  g.fillStyle = 'rgba(255,250,235,.35)'; g.fillRect(S * 0.62, S * 0.08, S * 0.26, S * 0.34);
  g.strokeStyle = 'rgba(90,70,50,.5)'; g.lineWidth = 8; g.strokeRect(S * 0.62, S * 0.08, S * 0.26, S * 0.34);
  g.beginPath(); g.moveTo(S * 0.75, S * 0.08); g.lineTo(S * 0.75, S * 0.42); g.moveTo(S * 0.62, S * 0.25); g.lineTo(S * 0.88, S * 0.25); g.stroke();
  const ball = (x, y, r, c0, c1) => {
    g.fillStyle = 'rgba(20,10,0,.45)';
    g.beginPath(); g.ellipse(x + r * 0.25, y + r * 0.92, r * 1.05, r * 0.26, 0, 0, Math.PI * 2); g.fill();
    const b = g.createRadialGradient(x - r * .35, y - r * .4, r * .05, x, y, r);
    b.addColorStop(0, '#fff8e8'); b.addColorStop(0.12, c0); b.addColorStop(1, c1);
    g.fillStyle = b; g.beginPath(); g.arc(x, y, r, 0, Math.PI * 2); g.fill();
  };
  ball(S * 0.27, S * 0.6, S * 0.13, '#e2342a', '#5a0c08');
  ball(S * 0.52, S * 0.66, S * 0.09, '#f2c832', '#7a5208');
  ball(S * 0.74, S * 0.6, S * 0.12, '#3a8a3c', '#0c2a10');
  // a vase
  g.fillStyle = '#2a4f9a';
  g.beginPath(); g.moveTo(S * .42, S * .62); g.bezierCurveTo(S * .34, S * .48, S * .36, S * .36, S * .44, S * .28);
  g.lineTo(S * .44, S * .2); g.lineTo(S * .5, S * .2); g.lineTo(S * .5, S * .28);
  g.bezierCurveTo(S * .58, S * .36, S * .6, S * .48, S * .52, S * .62); g.closePath(); g.fill();
  const hl = g.createLinearGradient(S * .36, 0, S * .58, 0);
  hl.addColorStop(0, 'rgba(255,255,255,0)'); hl.addColorStop(0.3, 'rgba(255,255,255,.35)'); hl.addColorStop(0.6, 'rgba(0,0,0,0)'); hl.addColorStop(1, 'rgba(0,0,0,.35)');
  g.fillStyle = hl; g.fill();
  ball(S * 0.47, S * 0.66, S * 0.08, '#f08a2a', '#6a2a04');
}

function landscape(g) {
  let gr = g.createLinearGradient(0, 0, 0, S * 0.6);
  gr.addColorStop(0, '#2b5fb8'); gr.addColorStop(1, '#f2d6a8');
  g.fillStyle = gr; g.fillRect(0, 0, S, S * 0.6);
  const sun = g.createRadialGradient(S * .7, S * .3, 0, S * .7, S * .3, S * .2);
  sun.addColorStop(0, '#fffbe8'); sun.addColorStop(0.25, '#ffe9a0'); sun.addColorStop(1, 'rgba(255,220,150,0)');
  g.fillStyle = sun; g.fillRect(0, 0, S, S);
  const ridge = (y0, amp, f, col, seed) => {
    g.fillStyle = col; g.beginPath(); g.moveTo(0, S);
    for (let x = 0; x <= S; x += 4) {
      const t = x / S;
      g.lineTo(x, y0 - amp * (Math.sin(t * f + seed) * .5 + Math.sin(t * f * 2.3 + seed * 2) * .3 + Math.sin(t * f * 5.1 + seed) * .12));
    }
    g.lineTo(S, S); g.closePath(); g.fill();
  };
  ridge(S * .5, S * .16, 6, '#8a96b8', 1);
  ridge(S * .58, S * .1, 9, '#5f7a6a', 4);
  // lake with reflection
  gr = g.createLinearGradient(0, S * .62, 0, S * .8);
  gr.addColorStop(0, '#c8b8a0'); gr.addColorStop(1, '#3a5a88');
  g.fillStyle = gr; g.fillRect(0, S * .62, S, S * .18);
  g.fillStyle = 'rgba(255,240,200,.5)'; g.fillRect(S * .64, S * .63, S * .12, S * .01); g.fillRect(S * .66, S * .66, S * .08, S * .008);
  ridge(S * .86, S * .05, 4, '#3e6a2a', 2);
  g.fillStyle = '#1c3a1a';
  for (const [x, h] of [[.08, .2], [.14, .26], [.2, .18], [.83, .24], [.9, .3]]) {
    g.beginPath(); g.moveTo(S * x, S * (.86 - h)); g.lineTo(S * (x - .045), S * .87); g.lineTo(S * (x + .045), S * .87); g.closePath(); g.fill();
  }
}

function letters(g) {
  g.fillStyle = '#002FA7'; g.fillRect(0, 0, S, S);
  g.fillStyle = '#ffffff';
  g.font = `700 ${S * 0.34}px "Helvetica Neue", Helvetica, Arial, sans-serif`;
  g.textAlign = 'center'; g.textBaseline = 'middle';
  g.fillText('ADP', S / 2, S * 0.42);
  g.fillStyle = '#ffcf3a';
  g.font = `700 ${S * 0.12}px "Helvetica Neue", Helvetica, Arial, sans-serif`;
  g.fillText('splat', S / 2, S * 0.72);
  g.fillStyle = '#ff5a3a'; g.beginPath(); g.arc(S * 0.84, S * 0.16, S * 0.07, 0, Math.PI * 2); g.fill();
}

function stripes(g) {
  g.fillStyle = '#f4f1ea'; g.fillRect(0, 0, S, S);
  g.save(); g.translate(S / 2, S / 2); g.rotate(0.5);
  g.fillStyle = '#1a1a1a';
  for (let k = -12; k < 12; k++) g.fillRect(k * S * 0.06, -S, S * 0.03, S * 2);
  g.restore();
  const n = 8, c = S / 2 / n;
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
    g.fillStyle = (x + y) % 2 ? '#e2342a' : '#f4f1ea';
    g.fillRect(S / 2 + x * c, S / 2 + y * c, c, c);
  }
}

const DRAW = { still: stillLife, landscape, letters, stripes };

// draw an image id (or an HTMLImageElement) at w x h; returns { data: Float32Array rgb 0..1, canvas }
export function makeTarget(src, w, h) {
  const big = document.createElement('canvas'); big.width = S; big.height = Math.round(S * h / w);
  const g = big.getContext('2d');
  if (typeof src === 'string') DRAW[src](g);
  else {
    // cover-fit the picture
    const r = Math.max(big.width / src.width, big.height / src.height);
    const dw = src.width * r, dh = src.height * r;
    g.drawImage(src, (big.width - dw) / 2, (big.height - dh) / 2, dw, dh);
  }
  const c = document.createElement('canvas'); c.width = w; c.height = h;
  const cg = c.getContext('2d', { willReadFrequently: true });
  cg.imageSmoothingQuality = 'high';
  cg.drawImage(big, 0, 0, w, h);
  const px = cg.getImageData(0, 0, w, h).data, data = new Float32Array(w * h * 3);
  for (let i = 0; i < w * h; i++) { data[i * 3] = px[i * 4] / 255; data[i * 3 + 1] = px[i * 4 + 1] / 255; data[i * 3 + 2] = px[i * 4 + 2] / 255; }
  return { data, canvas: c };
}
