# V18 dual-support diagnostic

V18 implements the strong mechanism test proposed in the GPT-5.6 Pro review
archived at:

`docs/research/reviews/GPT-5.6Pro_V13到V18双支持改造评审原文_20260728.md`

It does not replace V13. The official KPConvX path always receives the original
nearest-H graph, while a separate residual branch sees only new candidates in:

```text
base_radius < neighbor_distance <= adaptive_radius
candidate_slot >= base_H
```

This makes the supports disjoint and prevents the ring branch from processing
the baseline neighbors twice.

## Integration safety fixes

The initial V18 snapshot had two model-level integration errors that were not
covered by the graph utility tests:

- slicing `H` neighbors from a larger candidate tensor produced a non-contiguous
  view, which failed in the official `index_select(... indices.view(-1))` path;
- ring modules used the stage input width even though `grid_pool=True` makes the
  final encoder block project to the next stage width before the ring is called.

The identity neighbor slices are now made contiguous locally, and ring channels
are derived from the actual post-encoder width. For the S3DIS profile, stage 3
and stage 4 use 192 and 256 channels. A model-level CPU regression test verifies
the complete synthetic forward and exact `gamma=0` equality with official
KPNeXt. Real S3DIS CUDA identity equivalence is still required before training.

## Diagnostic profile

Standalone stages are one-based. Stages `[3, 4]` correspond to the Pro
review's zero-based resolution indices `[2, 3]`.

| Setting | Stage 3 | Stage 4 |
| --- | ---: | ---: |
| Base H | 20 | 20 |
| Candidate K | 40 | 48 |
| Maximum ring points | 8 | 12 |
| Scale range | 0.75-1.45 | 0.70-1.55 |
| Gamma | 0.25 fixed | 0.25 fixed |

The ring KPConvX scales both kernel positions and influence sigma by the same
per-query radius scale. Therefore expanded neighbors are compared with an
expanded kernel geometry instead of being forced onto the unchanged outer
shell.

Run:

```bash
bash audit_S3DIS_v18.sh
bash train_S3DIS_v18_diagnostic.sh
```

Run the audit first. It writes `KPConvX/results/v18_graph_audit.json`; do not
start the diagnostic training when `mechanism_too_weak` is true for both target
stages.

The lower-strength `v18_formal` profile is present only for the next phase. Do
not run it until the strong diagnostic shows that the ring is non-trivial and
useful. The formal profile intentionally refuses to start until fixed spacing
bounds for both stages are provided.

## Density normalization

V18 uses the dimensionless mean-neighbor spacing:

```text
mean_nonself_KNN_distance / current_stage_grid_size
```

The diagnostic initially uses packed-cloud 10/90 percentiles because global
training statistics have not yet been measured. Debug output reports spacing
q10/q90. After auditing representative training rooms, write fixed stage-wise
bounds to `dual_support_fixed_spacing_bounds` and enable
`dual_support_use_fixed_spacing`. Do not invent these values.

Use string keys because Standalone's `EasyDict` does not accept integer-keyed
nested mappings:

```python
cfg.model.dual_support_fixed_spacing_bounds = {
    "3": [stage3_q10, stage3_q90],
    "4": [stage4_q10, stage4_q90],
}
```

## Required pre-training audit

The debug output records:

- mean base-radius and adaptive-radius counts;
- fraction with at least H points inside the base radius;
- mean newly added ring neighbors;
- graph change rate;
- `d_H / base_radius`;
- scale and dimensionless-spacing quantiles.

Stop tuning scale ranges if mean added neighbors are below 1 or graph change is
below 10%. If more than 85% of points already contain H candidates inside the
base radius, radius expansion is structurally weak at that stage.

## Evidence gate

Advance to the formal three-seed experiment only when the strong diagnostic:

- changes the graph by at least 20-30%;
- adds at least one neighbor per query on average;
- improves identity fixed-test mIoU by at least 0.8 point, or gives a clear and
  consistent boundary/small-class improvement;
- increases peak GPU memory by no more than 20%;
- slows throughput by no more than 25%.

The official baseline, V13, and V18 must use identical input, training budget,
checkpoint selection, voting, and full-room restoration protocols.
