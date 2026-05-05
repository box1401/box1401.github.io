---
layout: post
title: "Implementing the Lee Filter as a Kornia Transform"
date: 2026-05-03 18:35:00 +0800
categories: [SAR, PyTorch]
tags: [sar, speckle, kornia, torchgeo, pytorch, lee-filter, despeckling]
math: true
image:
  path: /assets/img/posts/lee_filter_demo.png
  alt: "Lee filter demo: noisy SAR-like input vs. filtered output, showing speckle reduction with edge preservation."
---

This post is the technical walkthrough behind [torchgeo PR #3655](https://github.com/torchgeo/torchgeo/pull/3655), a small contribution that adds the classic Lee (1980) speckle filter to the PyTorch geospatial stack as a `Kornia` augmentation. I'll cover three things: (1) why SAR images need a speckle filter at all, (2) the math the filter is doing and how the prototype fell out of it, and (3) the design decisions involved in wrapping it as `IntensityAugmentationBase2D` so it composes with the rest of the augmentation pipeline.

## 1. Why SAR images need a speckle filter

If you've only worked with optical imagery, the texture of a SAR (synthetic aperture radar) image is jarring the first time you see it. A uniform field, like a flat sea surface, a parking lot, or a recently harvested crop, looks like static. That static is **speckle**: a multiplicative-noise artefact that follows from how SAR works (coherent imaging summing many sub-resolution scatterers in random phase), not a sensor defect. It's not going to be calibrated away upstream.

For my Master's thesis I'm comparing how different speckle-reduction strategies affect deep-learning **ship detection** in SAR imagery. Detector heads trained on raw single-look intensity see two superficial cues: ships *and* high-amplitude speckle peaks in calm water. The downstream model then either pays a precision tax (false positives in open ocean) or learns implicit denoising in its early layers, eating capacity it could spend on harder discrimination. Pre-filtering the input is one way to give the detector a cleaner signal, but only if the filter doesn't smear the ship edges in the process. That edge-vs-smoothing tradeoff is exactly what the Lee filter is designed to handle.

## 2. The math, and what falls out of it

The Lee filter assumes a **multiplicative-noise** model:

$$x = s \cdot v$$

where \\(x\\) is what the sensor measures, \\(s\\) is the underlying scene intensity we want, and \\(v\\) is unit-mean speckle noise with variance \\(\sigma_v^2 = 1/L\\) for an \\(L\\)-look intensity image.

The filter applies a per-pixel **linear minimum-mean-squared-error (LMMSE)** estimator over a local window:

$$\hat{s} = \mu_x + k \cdot (x - \mu_x), \quad k = \frac{\sigma_s^2}{\sigma_s^2 + \sigma_v^2 \cdot \mu_x^2}$$

with \\(\mu_x\\) and \\(\sigma_x^2\\) the local mean and variance, and \\(\sigma_s^2 = \max(\sigma_x^2 - \sigma_v^2 \mu_x^2, 0)\\) the estimated *signal* variance under the multiplicative model.

Read \\(k\\) intuitively. In a **homogeneous** patch, \\(\sigma_x^2 \approx \sigma_v^2 \mu_x^2\\), so \\(\sigma_s^2 \to 0\\), \\(k \to 0\\), and the output approaches the local mean (heavy smoothing). Near an **edge**, the local variance is dominated by real signal change rather than speckle, \\(k \to 1\\), and the output approaches the input (pass-through, edge preserved). The filter *adapts its smoothing strength to the local signal-to-noise ratio*, which is the property worth caring about.

The whole thing is tensor-friendly. Local mean and local mean-of-squares are box filters; everything else is elementwise. That's the entire prototype:

```python
import torch.nn.functional as F

def _box_filter(x, window_size):
    pad = window_size // 2
    x_padded = F.pad(x, (pad, pad, pad, pad), mode='reflect')
    kernel = torch.ones(1, 1, window_size, window_size,
                        device=x.device, dtype=x.dtype) / float(window_size**2)
    C = x.shape[1]
    kernel = kernel.expand(C, 1, window_size, window_size)
    return F.conv2d(x_padded, kernel, groups=C)

def lee_filter(image, window_size=7, num_looks=1.0, eps=1e-8):
    sigma_v_sq = 1.0 / float(num_looks)
    mean_local = _box_filter(image, window_size)
    mean_sq_local = _box_filter(image * image, window_size)
    var_local = (mean_sq_local - mean_local ** 2).clamp(min=0.0)
    var_signal = (var_local - sigma_v_sq * mean_local ** 2).clamp(min=0.0)
    weight = var_signal / (var_signal + sigma_v_sq * mean_local ** 2 + eps)
    return mean_local + weight * (image - mean_local)
```

Three details worth flagging that aren't in the textbook statement:

- **Reflect padding, not zero padding.** Zero padding fakes a step edge into the image at every border, which the LMMSE estimator faithfully *preserves*, leaving a thick black halo. Reflect padding hides the boundary from the filter.
- **Per-channel grouped conv.** SAR products often come polarisation-stacked (VV, VH). Treating them as independent channels via `groups=C` is the right default; cross-channel mixing isn't physically motivated for speckle.
- **`clamp(min=0)` on `var_signal`.** Under finite samples in a noisy patch, \\(\sigma_x^2\\) can dip below \\(\sigma_v^2 \mu_x^2\\) and produce a negative variance estimate. The clamp keeps the estimator well-defined; without it you get NaNs in the gradient on the first batch.

Below is the demo figure I generated for the proposal: synthetic SAR-like input on the left, Lee-filtered output on the right. The point isn't *"look how clean"*; it's that the bright-on-dark step edges in the middle survive while the homogeneous patches calm down.

![Lee filter on synthetic SAR-like input: noisy input vs. filtered output](/assets/img/posts/lee_filter_demo.png){: w="800" }
_Synthetic 1-look intensity image with multiplicative speckle, before and after Lee filtering (window size 7)._

## 3. Wrapping it as a Kornia `IntensityAugmentationBase2D`

The functional version above is fine for a script. To actually live inside the torchgeo / Kornia ecosystem, where a transform composes with random crops, geometric augmentations, normalisation, and the usual `p=0.5`-style probabilistic application, it needs to subclass `kornia.augmentation.IntensityAugmentationBase2D`. Kornia splits augmentations into intensity-only (don't change geometry) and geometric (do); the Lee filter is unambiguously the former, which means it inherits the right batched/probabilistic plumbing for free.

```python
from kornia.augmentation import IntensityAugmentationBase2D

class LeeFilter(IntensityAugmentationBase2D):
    def __init__(self, window_size=7, num_looks=1.0,
                 p=1.0, same_on_batch=False, keepdim=False):
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)
        if window_size < 1 or window_size % 2 == 0:
            raise ValueError(f'window_size must be a positive odd integer, got {window_size}')
        if num_looks <= 0:
            raise ValueError(f'num_looks must be > 0, got {num_looks}')
        self.flags = {'window_size': window_size, 'num_looks': num_looks}

    def apply_transform(self, input, params, flags, transform=None):
        return lee_filter(
            input,
            window_size=int(flags['window_size']),
            num_looks=float(flags['num_looks']),
        )
```

A few choices worth justifying:

- **`flags`, not `__init__` attributes, hold the parameters.** Kornia's base class re-passes `flags` into `apply_transform`, which is the contract you want to honour so the transform plays nicely with `same_on_batch`, `keepdim`, and serialisation. Storing `self.window_size` instead works in the simple case but breaks when you stack the filter inside a `nn.Sequential` of augmentations and Kornia walks the tree.
- **Validation in `__init__`, not in `apply_transform`.** Speckle filters get applied many times per epoch; doing the odd-integer check on every call wastes compute. Fail loud at construction.
- **Default `p=1.0`.** Unlike a flip or a colour jitter, a despeckling filter isn't a *random* augmentation. It's a preprocessing step you almost always want applied. `p` is exposed so users *can* randomise it (e.g. mixing filtered and raw samples to make the detector robust to either), but the default reflects how the filter is actually used in practice.

The PR ([#3655](https://github.com/torchgeo/torchgeo/pull/3655), +316 / −0, currently in review with five maintainers) ships this `LeeFilter` class plus the functional `lee_filter` API, exposed through `torchgeo.transforms.LeeFilter`, with 28 tests covering the new module at 100% line coverage: parameter-validation paths, batched application, single- and multi-channel inputs, gradient flow, device placement, and the `same_on_batch` / `keepdim` Kornia contracts. There's a fix commit on top of the original push that wraps the test-module's `import scipy` in `pytest.importorskip(...)` to keep torchgeo's `optional` CI job (Python 3.14, no `[datasets]` extras) happy. Small but project-specific, worth respecting if you're sending your first torchgeo PR.

## What's next

The natural follow-up (and the one a maintainer has already flagged as the logical PR #2) is **RefinedLeeFilter**, Lee's own 1981 follow-up that replaces the box-window local statistics with edge-aware directional sub-windows. Better edge sharpness at the cost of a more complicated kernel. That's the next post in this series.

If you spot something off in the implementation above, or want to discuss SAR despeckling for object-detection pipelines, the PR thread is the best place. You can also [open an issue against my fork](https://github.com/box1401) and I'll get back.
