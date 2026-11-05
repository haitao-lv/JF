# Fluorescence quantum yield and lifetime: molecular generalization

Core implementation accompanying **Predicting fluorescence quantum yield and lifetime: molecular generalization and the limits of solvent-response prediction**.

Authors: Jianjun Shi, Shuo Wang, Juan Wang and Haitao Lv.
Correspondence: 920345@hainnu.edu.cn.

This repository contains the same scientific scripts, protocols and pinned
dependencies as Online Resource 2 (ESM_2.zip) of the manuscript submitted to
Journal of Fluorescence. It excludes manuscript-generation and plotting code.
No trained model weights are supplied.

## Data and reproduction

Place the contents of **Online Resource 3 (ESM_3.zip)** in the repository root,
so that `scripts/`, `data/` and `results/` share a parent directory. The archive
contains original and processed data, features, fixed partitions, saved
predictions and numerical analyses. The original measurements alone do not
replace this analysis archive. Until the supplementary files are published,
request ESM_3.zip from the corresponding author.

Use Python 3.10 and install the pinned dependencies:

```sh
python -m pip install -r requirements_core.txt
python -X utf8 scripts/validate_pilot.py
python -X utf8 scripts/validate_extension.py
python -X utf8 scripts/analyze_extension.py
python -X utf8 scripts/analyze_revision_v4.py
```

See `README_Core_Code.md` for refitting, run-lock and floating-point details.
Its scientific commands are retained from the journal archive. Read prediction
CSV files with `float_precision="round_trip"`.

## Interpretation

The comparison concerns 13 fixed configurations and pair, chromophore and
scaffold holdouts. Expanded comparisons reuse inspected holdouts and are
exploratory. Absolute-property performance does not establish reliable joint
solvent-response prediction. Report response types, molecular coverage and
contributing molecule and source counts when interpreting solvent comparisons.

## Data attribution

Joung et al., *Scientific Data* **7**, 295 (2020),
https://doi.org/10.1038/s41597-020-00634-8.
Versioned original dataset: https://doi.org/10.6084/m9.figshare.12045567.v2,
CC BY 4.0. The data archive retains the original attribution and license
metadata. This study adds filtering, structural descriptors, partitions and
computational analyses. The dataset license does not imply a software license.
