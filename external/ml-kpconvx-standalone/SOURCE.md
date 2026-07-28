# KPConvX Standalone source

This directory started as a local copy of the official Apple KPConvX
Standalone code and now also contains an isolated local V13 DA-Radius
extension.

- Repository: https://github.com/apple/ml-kpconvx
- Source commit: `54e644a9f3bddd4c344a58193897a44582b0fea4`
- Copied from: `tmp/ml-kpconvx-official/Standalone`
- Copy date: 2026-07-27
- Upstream license: Apple Sample Code License, copied verbatim from the
  repository root to `LICENSE` for redistribution with this source snapshot.

The Standalone pipeline is intentionally kept independent from Pointcept. The
S3DIS reproduction entry point is `KPConvX/experiments/S3DIS/train_S3DIS.py`;
the shell wrapper is `train_S3DIS.sh`. Dataset paths and Python dependencies
are configured by the Standalone code itself and should not be inferred from
Pointcept configuration files.

## Local extension

The official baseline remains the default. Local V13 files and integration
points are:

- `KPConvX/utils/da_radius.py`
- `KPConvX/models/KPNextV13.py`
- optional DA-Radius settings in `KPConvX/experiments/S3DIS/train_S3DIS.py`
- `train_S3DIS_v13.sh`
- `train_S3DIS_v13_conservative.sh`
- `V13_STANDALONE.md`
- `KPConvX/utils/dual_support.py`
- `KPConvX/models/KPNextV18.py`
- `train_S3DIS_v18_diagnostic.sh`
- `train_S3DIS_v18_formal.sh`
- `audit_S3DIS_v18.sh`
- `V18_DUAL_SUPPORT.md`

Run `train_S3DIS.sh` for the unchanged official behavior. The V13 launchers
must be treated as local experiments, not as Apple-provided recipes.
