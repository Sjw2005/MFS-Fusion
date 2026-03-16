# Progress

## Completed
- Located core planning and experiment documents under `md/`.
- Mapped current project architecture to actual code in `Net/MyNet.py`.
- Confirmed current worktree already contains partial P0 code (`FSDA` integrated in `ASI`).
- Screened the curated module list and produced an initial shortlist of compatible low-compute modules.

## In Progress
- Converting document-level guidance into a concrete stage-1 implementation and experiment order.

## Next Actions
1. Refactor `ASI`/`MEF`/`MyNet` to expose `use_fsda` and preserve the original branch.
2. Add P0 verification scripts or test snippets for shape, gradient, and FLOPs.
3. Implement `ImprovedMEF` with CMA and make it switchable.
4. Implement `ImprovedESI` with IDC and make decoder replacement switchable.
5. After P0/P1, evaluate one extra module at a time, starting from `EMA`.
