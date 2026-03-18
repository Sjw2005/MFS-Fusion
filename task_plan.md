# MFS-Fusion Stage-1 Plan

## Objective
- Build a complete stage-1 execution plan for this project.
- Focus first on implementing and validating the P0 and P1 experiment guides under `md/`.
- Analyze the curated 70-module list and identify additional low-compute modules that fit the current codebase and can improve specific weaknesses without major training/inference overhead.

## Active Task - NaN Stabilization
- Goal: make a minimal, low-risk patch to reduce `loss=nan` during training.
- Constraints: keep code changes small, preserve current training pipeline, and prefer config-level enablement where possible.
- Planned actions:
  1. Ensure final network output is bounded to `[0, 1]`.
  2. Remove phase-branch negative-value truncation in `FSDA`.
  3. Enable gradient clipping in the active training config.
  4. Keep targeted debug prints so first failing module can be identified quickly if instability remains.

## Scope
- Baseline target network: `Net/MyNet.py` with PVTv2 backbone (`MyNet`).
- Primary docs:
  - `md/MFS_Fusion_P0_实验操作指南_FSDA替换AttenFFT.md`
  - `md/MFS_Fusion_P1_实验操作指南_CMA替换MEF_IDC替换ESI.md`
  - `md/MFS_Fusion_Improvement_Analysis.md`
  - `md/Larry同学整理的模块V3.md`
  - `md/project_core_code_summary.md`

## Current Code Status
- `Net/MyNet.py` already contains `FSDA` and `ECALayer`.
- `ASI` already instantiates `FSDA` instead of `AttenFFT`.
- `MEF`, `ESI`, `SpatialAttention`, `sa_layer`, `Decode`, and `MyNet` are still largely in the original design.
- This means P0 is partially integrated in code, but the experiment control switches, validation hooks, and ablation path are not yet complete.

## Work Breakdown

### Phase A - Reproducible Baseline
1. Verify baseline training/inference entrypoints and config usage.
2. Record baseline metrics on at least MSRS / TNO / RoadScene.
3. Add temporary debug hooks for:
   - `AttenFFT` numeric behavior (for original branch if retained)
   - `MEF` elementwise fusion statistics
   - `ESI` branch response statistics

### Phase B - P0 Execution (FSDA for AttenFFT)
1. Refactor `ASI` to support `use_fsda=True/False` instead of hard replacement only.
2. Keep both `AttenFFT` and `FSDA` available for fair ablation.
3. Add integrity checks:
   - shape consistency
   - gradient flow
   - params/FLOPs comparison
4. Run P0 ablations:
   - A0: AttenFFT
   - A1: FSDA + ECA
   - A2: FSDA + SE (optional if time allows)
   - B-series and C-series as secondary ablations

### Phase C - P1 Execution (CMA + IDC)
1. Implement bidirectional channel-level CMA fusion as a new `ImprovedMEF`.
2. Add `use_cma` switch so original `MEF` remains available.
3. Implement IDC-based `ImprovedESI` with shape-compatible decoder replacement.
4. Add controlled combinations:
   - baseline/P0-best
   - +CMA only
   - +IDC only
   - +CMA+IDC

### Phase D - Extra Module Screening
1. Prioritize modules that map directly to known weaknesses:
   - weak spatial attention
   - hard-coded grouped attention
   - decoder high-frequency loss
   - low-cost multi-scale context enhancement
2. Select only low-compute, code-friendly candidates for this repo.
3. Rank by: fit to problem, code intrusion, compute overhead, likely measurable gain.

## Recommended Additional Modules Beyond P0/P1
1. `HFP` (#66): best fit for decoder high-frequency compensation.
2. `EMA` (#13): strongest drop-in replacement for current `SpatialAttention`.
3. `SCSA` (#54): strongest replacement candidate for `sa_layer`.
4. `CoordAttention` (#4): lighter backup option when EMA/SCSA are too intrusive.
5. `scSE` (#15): cheap enhancement for output head / shallow decoder fusion.

## Suggested Stage Order
1. Finish P0 control + validation.
2. Run P0 main ablation and lock baseline for stage 1.
3. Implement P1 in two separable branches: CMA first, IDC second.
4. After P1, test one extra low-cost module at a time, starting with EMA or HFP.

## Success Criteria
- P0/P1 have executable experiment instructions tied to actual code locations.
- Each modification can be toggled for ablation.
- Additional module shortlist is small, justified, and low-risk.
- No recommendation materially increases compute without a clear target benefit.
