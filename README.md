# How Gaussian splatting works — a Splat Trainer

**[Open the live trainer →](https://adpchrgruber.github.io/adp_adv_gaussian_splatting/)**

Soft, coloured, semi-transparent ellipses learn a picture, step by step. On the left is
the photo, on the right the splats: you watch them being blended, compared with the
photo, nudged by gradient descent, split, cloned and pruned until they match. Everything
runs in the browser, no build step and no libraries.

> Made with generative AI assistance — see [Credits and AI disclosure](#credits-and-ai-disclosure).

It is the companion to the [Scanner Simulator](https://github.com/adpchrgruber/adp_adv_pointcloud_simulator):
the scanner shows how a point cloud is measured, this page shows how Gaussian splatting
turns points and photos into a picture that renders in real time.

## Use

1. **01 Train** — *Start* (Space) trains, *Step* does a single step, *Reset* starts over.
   *Photo* picks a drawn test image (still life, landscape, letters, stripes) or your own
   (also by dropping a file on the page). *Start from* places the first splats on the
   photo's edges and corners (**sparse points**, as structure from motion would) or at
   random. *Splats* is how many to start with, *Resolution* the size of the photo,
   *Step size* scales the learning rates.
2. **02 Densify** — every 100 steps until step 2000, splats that are still pulled hard
   are **split** (big ones, into two smaller ones) or **cloned** (small ones), nearly
   invisible ones are **pruned**. *Max splats* is the budget.
3. **03 Display** — *Splat size* shrinks every splat to show that the picture is made of
   blobs; *Show centres* marks their positions; *3D depth* sets how far the 3D view
   pulls the splats apart.
4. **04 One splat** — click the photo or the splats: the splat that paints that spot is
   ringed and its nine numbers are listed (position, size, rotation, colour, opacity,
   depth order). In 3D a click picks the nearest splat centre. *Show only this splat* hides all others.
5. **05 How it works** — the full 3D pipeline in six steps, and which of them this page
   shows.

**The photo** — *Photo* is the target at its real pixel size; *Error* shows where the
splats still differ from it (white = wrong). **The splats** — *Render* is the blended
result, drawn smoothly at screen resolution; *Ellipses* outlines every splat at 2 sigma
in its colour. **3D** works with both: it pulls the splats (or their ellipses) apart in
their blending order, front splats nearest to you — drag to orbit, wheel to zoom.

The readouts compare the photo (W × H × 3 numbers) with the splats (n × 9 numbers) and
give the match as PSNR in dB (higher is better; above ~30 dB differences are hard to see).

## How it works

- **A splat** here is 9 numbers: x, y, two log scales, a rotation, colour (3, as logits)
  and opacity (as a logit). Its covariance is R S Sᵀ Rᵀ, widened by 0.3 px² — the
  low-pass filter of the paper, so no splat falls between two pixels.
- **Rendering** (`js/splats.js`): the image is cut into 8 × 8 px tiles; every splat is
  listed in the tiles its 3-sigma box touches, in depth order. Each pixel blends its
  tile's splats front to back, `C = Σ cᵢ αᵢ Tᵢ`, with `αᵢ = opacity · exp(−½ dᵀ Σ⁻¹ d)`
  and `Tᵢ = Π (1 − αⱼ)`, and stops when the pixel is full — as in the paper's CUDA
  rasterizer, here in plain JavaScript.
- **Training**: L2 loss against the photo; the gradients run back through the blending
  (back to front), the conic Σ⁻¹ and the covariance to all nine numbers, and Adam updates
  them. The position learning rate decays exponentially. Gradients were checked against
  finite differences.
- **Densification** follows the paper's adaptive density control: average positional
  gradient per splat over 100 steps; above the threshold, split if large, clone if small;
  prune below 1 % opacity. Growth is limited to a third per round.
- **Display** (`js/viewer.js`): WebGL2, one instanced quad per splat, Gaussian falloff in
  the fragment shader and premultiplied "over" blending back to front — the same maths
  as the trainer, at screen resolution. The splats are resolution independent; the photo
  is not.
- **What is left out**: the third dimension. In 3D every splat is an ellipsoid with 59
  numbers (colour as spherical harmonics, so it can change with the view), it is
  projected into each training photo with the camera's Jacobian, and every step uses a
  different photo — which is what puts each splat at the right depth. Here the depth
  order is fixed at random and only one photo is fitted.

## Credits and AI disclosure

The Splat Trainer was co-created by **adpchrgruber** and generative AI.

**Use of generative AI.** The site was developed in conversation with *Claude Opus 5.5*
(Anthropic), working as a coding agent in *Claude Code*, in October 2026. The AI:

- wrote the source code: the 2D Gaussian splatting trainer with its gradients, Adam
  optimiser and densification (`js/splats.js`), the WebGL2 viewer (`js/viewer.js`), the
  code that draws the test pictures (`js/images.js`), the interface (`index.html`,
  `css/style.css`, `js/main.js`);
- checked the gradients against finite differences and tested the page in a browser;
- drafted the texts on the page and this README.

**Human contribution.** adpchrgruber conceived the site, set its brief and direction
(a simpler companion to the Scanner Simulator, dark mode first), made the design
decisions, reviewed the results and publishes it. The author takes responsibility for the
published site.

**No AI-generated imagery.** The test pictures are drawn by code on a canvas, and every
splat image is computed live in the visitor's browser by an optimisation algorithm
(gradient descent), not by a generative AI model.

**Citing the AI tools.** APA 7:

> Anthropic. (2026). *Claude* (Opus 5.5 version) [Large language model].
> https://www.anthropic.com/claude
>
> Anthropic. (2026). *Claude Code* [Computer software]. https://claude.com/claude-code

Chicago (note):

> 1. Code and text generated by Claude Opus 5.5, Anthropic, via Claude Code, October 2026,
> https://claude.com/claude-code; prompted and reviewed by adpchrgruber.

**Citing the Splat Trainer.** See [`CITATION.cff`](CITATION.cff), or *Cite this repository*
on the GitHub page.

**Method and acknowledgements.**

- B. Kerbl, G. Kopanas, T. Leimkühler, G. Drettakis:
  [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/),
  ACM Transactions on Graphics (SIGGRAPH) 2023. Blending, gradients, the 0.3 px filter
  and densification follow the paper, reduced to 2D.
- UI style follows [Scan Matter](https://github.com/adpchrgruber/adp_adv_point_lab) and the
  [Scanner Simulator](https://github.com/adpchrgruber/adp_adv_pointcloud_simulator).

## Run locally

Static site, no build step. The ES modules need http, not `file://`:

```bash
python3 -m http.server 8000
```

Open http://localhost:8000.



## Repository layout

```
index.html       page, panels and theme switch (dark first)
css/style.css    UI style
js/main.js       UI, training loop, inspector
js/splats.js     2D Gaussian splatting: tiles, blending, gradients, Adam, densification
js/viewer.js     WebGL2 splat renderer: render and ellipses, flat or in 3D
js/images.js     drawn test photos, loading your own
CITATION.cff     how to cite, incl. the AI tools used
```
