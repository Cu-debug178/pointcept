# V13 DA-Radius for KPConvX Standalone

V13 is retained as a reproducible historical ablation. The stronger
identity-plus-ring redesign recommended by the later Pro review is implemented
separately in `V18_DUAL_SUPPORT.md`; do not rename V18 results as V13.

This extension keeps the official KPConvX S3DIS baseline unchanged. DA-Radius
is enabled only through a dedicated launcher or `--enable_da_radius 1`.

## Implemented semantics

For selected encoder stages, V13 estimates density from the first 16 ordered
nearest-neighbor candidates. Density is normalized independently inside each
packed cloud with the 10th and 90th percentiles. Sparse points receive a larger
effective radius; candidates outside each query radius are replaced by the
official shadow index.

The Standalone implementation filters the existing ordered H-NN candidate
graph. It does not depend on Pointcept's custom `adaptive_ball_query_idx` CUDA
extension. This is equivalent when the adaptive query returns the nearest H
points inside the radius, but it is not bitwise equivalent to a first-hit hash
query. Results must therefore be reported as `V13-on-Standalone`, not as an
exact backend reproduction of the Pointcept experiment.

## Profiles

`train_S3DIS_v13.sh` uses the original Pointcept V13 settings:

- stages 3 and 4;
- stage 3 range `[0.95, 1.20]`;
- stage 4 range `[0.90, 1.30]`;
- strength `0.5`.

After applying the strength interpolation, the effective scale intervals are
approximately `[0.975, 1.10]` for stage 3 and `[0.95, 1.15]` for stage 4.

`train_S3DIS_v13_conservative.sh` is the recommended first Standalone run:

- stage 4 only;
- range `[0.95, 1.10]`;
- strength `0.35`.

Its effective scale interval is approximately `[0.9825, 1.035]`.

The conservative profile is intentionally smaller because the official 0.04 m
S3DIS recipe has approximately 1.83 times the physical layer support of the
0.02 m Pointcept recipe where V13 was developed.

The source recipe at commit `54e644a` uses `(1, 14, 28)`. Earlier project
audits of Apple's downloadable S3DIS checkpoint recorded `(1, 14, 42)` in the
checkpoint configuration. That release artifact is not present in this local
directory, so reproduce its saved configuration from the archive before using
57 kernel points as an experimental baseline. Kernel count and physical radius
must be ablated separately.

## Evidence before the Standalone port

Under the existing Pointcept deterministic identity protocol, V13 improved the
best checkpoint from `0.65557` to `0.65854` mIoU. This is only a `+0.30` point
single-run signal. Historical random validation was lower than the baseline.

The V13 diagnostics also show that the graph changed only slightly: stage 3
usually retained about `19.5 / 20` neighbors and stage 4 retained about
`18.9-19.7 / 20`. The shadow ratio was commonly `2-5%`. Fixed-test class IoU
improved most for ceiling and clutter, while door, window, column, and bookcase
regressed. This evidence does not support a claim that V13 already provides a
substantially larger sparse-region receptive field.

## Required comparisons

1. Official baseline, unchanged.
2. V13 conservative, official linear influence.
3. V13 Pointcept profile, official linear influence.
4. Constant-influence baseline.
5. V13 Pointcept profile with constant influence.

Use the same seeds, checkpoints, vote count, batch calibration, and full-room
test protocol. Record per-stage scale distributions, valid-neighbor ratios,
peak GPU memory, throughput, and per-class IoU. A single best checkpoint is not
enough evidence; use at least three seeds before claiming an improvement.

Stop this branch if two matched seeds fail to improve fixed-protocol mIoU, if
door/window/column regress systematically, or if more than 95% of queries keep
the full candidate graph. In the last case the radius controller is nearly an
identity operation and the expansion hypothesis is not being exercised.
