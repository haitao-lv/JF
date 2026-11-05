Predicting fluorescence quantum yield and lifetime: molecular generalization and the limits of solvent-response prediction
Journal of Fluorescence
Jianjun Shi, Shuo Wang, Juan Wang, Haitao Lv
Corresponding author: Haitao Lv, 920345@hainnu.edu.cn
College of Chemistry and Chemical Engineering, Hainan Normal University, Haikou 571158, China
Key Laboratory of Water Pollution Treatment and Resource Reuse of Hainan Province, Haikou 571158, China

# Core code: fluorescence quantum yield and lifetime

This attachment contains data-processing, feature-generation, training and numerical-validation code only. It excludes plotting and Word-export scripts. Python 3.10 was used.

Extract ESM_2.zip and ESM_3.zip into the SAME working folder so that scripts/, data/ and results/ share a parent. Keep the original archives unchanged. The data are already present; no download is required to reproduce the saved analysis.

Install the pinned core dependencies and validate the supplied results:

    python -m pip install -r requirements_core.txt
    python -X utf8 scripts/validate_pilot.py
    python -X utf8 scripts/validate_extension.py
    python -X utf8 scripts/analyze_extension.py
    python -X utf8 scripts/analyze_revision_v4.py

Use pandas.read_csv(..., float_precision='round_trip') for stored predictions. Default CSV parsing can alter inclusion at an interval boundary through floating-point rounding.

To fit the extension from scratch, work in a separate extracted copy, rename that copy's results/extension_v2 directory, then run scripts/prepare_extension.py and scripts/run_extension.py. The original paired data, features and partitions are already provided. On interrupted runs, resume run_extension.py directly; do not regenerate input files first. Run locks compare both source and input checksums. If protocols or inputs change, use a new output directory rather than reuse locked results.

To refit the initial models using the provided features and protocol, use another working copy, rename results/pilot_v1, and run scripts/run_pilot.py. These partitions are regenerated deterministically from the protocol. An independently regenerated pilot can change byte hashes (and floating-point last digits on different platforms), so regenerate extension inputs and its run lock consistently if using that new pilot. Do not mix old extension artifacts with a new pilot manifest.

The v4 analysis script performs postprocessing only: response subsets, effective counts, paired conditional bootstrap intervals, source sensitivity, interval decisions and coverage-selected chemical series. It does not refit the supplied models. Its protocol is saved in results/revision_v4/analysis_protocol.json. These analyses reuse previously inspected holdouts. Bootstrap intervals condition on the fitted models and evaluated observations, rather than representing refitting variability or a new validation cohort. The source metadata audit is supplied separately; running the numerical analyzer does not retrieve or verify experimental source tables.

No trained model weights are supplied. All configurations, original observations, partitions and predictions are provided. The extension is exploratory and reuses previously inspected paired holdouts. Additional training uses real available labels only, with all calibration/test groups excluded from every target training set.

Source data: Joung et al., Scientific Data 7, 295 (2020), doi:10.1038/s41597-020-00634-8; Figshare v2 doi:10.6084/m9.figshare.12045567.v2, CC BY 4.0. Preserve the source attribution, license metadata and an indication that filtering, structure processing and computational analyses were added. The paper's SI documents scope and interpretation limits.


Journal packaging note: only attachment names and documentation changed. Scientific scripts, data, protocols and result files are byte-identical to the reviewed v5 package. The two archives have distinct README and manifest names and can be extracted into one folder without these files overwriting each other.
