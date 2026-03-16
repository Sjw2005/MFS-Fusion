# Findings

## Project Architecture Snapshot
- Main target is `Net/MyNet.py`.
- Encoder: dual-stream PVTv2 backbone.
- Fusion blocks: `MEF` at 3 scales.
- Decoder: `Decode` with repeated `ESI` + `sa_layer`.
- Output head: `ESI(48)` + `sa_layer(48)` + final conv.

## What Is Already Done
- `FSDA` and `ECALayer` are already present in `Net/MyNet.py`.
- `ASI` already uses `FSDA` in all four branches.
- So the repo is not at pure baseline anymore; stage-1 planning must treat current code as a partially modified worktree.

## Gaps Between Docs and Code
- P0 doc asks for a switchable comparison between `AttenFFT` and `FSDA`; current code hardwires `FSDA` in `ASI`.
- P1 doc asks for `ImprovedMEF` and `ImprovedESI`; these are not integrated yet.
- `SpatialAttention` still uses only channel max pooling, matching the weakness noted in `md/MFS_Fusion_Improvement_Analysis.md`.
- `sa_layer` still uses fixed `groups=4`, also matching the documented weakness.
- `ESI` still uses 1x1/3x3/5x5/7x7 standard conv branches before Mamba, which remains a likely compute hotspot.

## P0 Practical Notes
- Keep `AttenFFT` for fair ablation; do not delete it yet.
- Best implementation path is to add a `use_fsda` flag to `ASI`, and then expose that flag upward through `MEF` and `MyNet`.
- Validate not only shape and gradients, but also FFT normalization consistency because current `FSDA` uses `norm='ortho'` while original `AttenFFT` uses default FFT normalization.
- P0 should be considered complete only after the code can instantiate both old/new branches from the same codebase.

## P1 Practical Notes
- There is already an unused cross-modal attention implementation in `Net/interact.py` (`Attention_conv`), which confirms that deeper cross-modal interaction is architecturally compatible with this repo.
- For P1, it is cleaner to add `ImprovedMEF` rather than patch old `MEF` in place.
- For decoder efficiency, `IDC` is the most direct replacement for `ESI`'s large-kernel conv branches.

## Additional Module Screening Beyond P0/P1

### High-priority candidates

#### 1. HFP (#66)
- Best target: insert after each decoder upsample block and before/after `ESI` replacement.
- Solves: decoder-side lack of explicit frequency refinement, weak high-frequency recovery.
- Cost: moderate but still controlled; higher than pure attention blocks, lower risk than adding another heavy transformer-style unit.
- Why it fits: directly addresses a known structural asymmetry in this project.

#### 2. EMA (#13)
- Best target: replace `SpatialAttention` inside `MEF`, or use in shallow decoder attention positions.
- Solves: current spatial attention is too simplified and ignores avg-pool/global context.
- Cost: low.
- Why it fits: directly maps to the documented weakness and is more expressive than the current max-pool-only block.

#### 3. SCSA (#54)
- Best target: replace `sa_layer` in decoder and output head.
- Solves: fixed-group `sa_layer` and weak channel-spatial coordination.
- Cost: low-to-moderate.
- Why it fits: specifically improves channel-spatial coupling without changing the overall topology.

### Secondary candidates

#### 4. CoordAttention (#4)
- Best target: replace lightweight attention points where EMA is still too heavy or too invasive.
- Solves: poor positional encoding in lightweight attention spots.
- Cost: very low.
- Why it fits: stable, cheap, easy to insert, especially useful in decoder/output head.

#### 5. scSE (#15)
- Best target: final head or skip-fusion refinement.
- Solves: weak simultaneous channel+spatial recalibration in output refinement.
- Cost: low.
- Why it fits: easy ablation candidate if SCSA is too complex for first pass.

## Modules Not Recommended First
- Heavy self-attention/transformer-style modules from the 70-list are not first-choice here because this repo already has PVTv2 + Mamba, and the immediate bottlenecks are fusion quality, decoder efficiency, and lightweight attention quality rather than missing global modeling capacity.
- Random-mask or segmentation-oriented modules without clear fusion-task alignment are lower priority.

## Recommended Ranking
1. P0: FSDA switchable ablation cleanup
2. P1-A: CMA for `MEF`
3. P1-B: IDC for `ESI`
4. P2 candidate: EMA
5. P2 candidate: HFP
6. P3 candidate: SCSA
7. Low-risk fallback: CoordAttention / scSE
