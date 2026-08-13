# BullionVault Multi-Horizon Research — Stage 3

Stage 3 executed the Stage-2 preregistered development-only comparison on commit `a9b014a1b92495259447e02506513d6677498fa5`.

The locked historical-test blocks were not fitted, predicted, scored, or used for selection. The formal future holdout and Shadow62 were not read or modified.

## Development result

No preregistered candidate passed the development gate on any available horizon.

- 4h: Random Walk retained. Ridge was the least-worse candidate at -6.96% MAE improvement versus baseline; all candidates failed the gate.
- 12h: Random Walk retained. Huber was the least-worse candidate at -13.46%; all candidates failed.
- 2d: Random Walk retained. Ridge was the least-worse candidate at -20.12%; all candidates failed.
- 30d: Random Walk retained. Ridge was the least-worse candidate at -9.31%; all candidates failed.
- 1d: remains `DATA_PENDING`.

Because no candidate passed development selection, historical confirmation is not authorized for any horizon. The reserved historical-test rows remain sealed for a future, newly versioned candidate.

## Evidence lock

- Stage-2 preregistration fingerprint: `fcf19e14ef55932093cd5406034700469b1e04723ac3d11b6c543345cb33b1d6`
- GitHub Actions run: `31697106962`
- Artifact ID: `9179752084`
- Artifact digest: `sha256:20b2cbb61ed5e37fda95d562b9162fdab8e68784f236eabbd1cf4a71232795d4`
- Full development report SHA256: `279b824c0775710b9b60a03a39564519e5ed728a54caca4febcefd71a24586f9`

The compact immutable selection lock is stored at `research_data/bullionvault_horizons/stage3_development_selection_lock.json`.

## Scientific interpretation

This is a valid negative result. The current small BullionVault chart samples do not justify promoting Ridge, Huber, or ElasticNet for these horizons. Stage 3 therefore does not create forecast model artifacts or public forecast pages.

The next research version must add genuinely new evidence or a preregistered new candidate family. It must not tune against the still-sealed historical-test blocks.

## Guardrails

`edge_status=NOT_PROVEN`, `research_only=true`, no BUY/SELL, no execution, no automatic promotion, no mutation of the frozen 52-feature production graph, and no mutation of Shadow62.
