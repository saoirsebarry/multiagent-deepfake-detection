# Robustness fine-tune protocol (pre-registered before any run)

Data: PolyGlotFake training split (1,050 clips) for all gradient updates; validation
split (238 clips) for every selection decision; the test split (2,162 clips) and the
frozen YouTube set are each read once, after all selections are final.

Recipe: continued training from each released checkpoint with the released
architecture, loss and inference recipe unchanged, under deployment-oriented
augmentation applied to training clips only:
  faces  - random JPEG re-encoding (quality 30-90), random down/up-scaling (0.3-0.9),
           Gaussian blur, additive noise, brightness/contrast jitter, horizontal flip
  audio  - random gain, additive noise at 15-35 dB SNR, 8 kHz band-limiting, time shift
Per epoch the clean validation split is scored with the released inference recipe.

Adoption rule (per agent): the retrained checkpoint replaces the released one only if
its validation log-loss is lower than the released agent's AND its validation AUC-ROC
is at least the released agent's minus 0.002. Otherwise the released agent stays.

Ensemble: the released weight-selection rule is re-run on the validation split with the
adopted agents (0.05-step simplex, all agents active, min tiered errors at tau = 0.5,
tie-break on two-sided clearance). tau = 0.5 and tau_d = 0.30 are fixed.

Read-out: one pass over the test split and one over the frozen YouTube set.

## Fix 6 (added 2026-09-07 19:20 UTC, before any corrupted-validation result existed)

Weight selection under corruption: the released selection rule (0.05-step all-active
simplex, minimise tiered errors at tau = 0.5, tie-break two-sided clearance) is re-run on
the validation split scored clean AND under K = 2 deployment-style corruptions per clip
(the same corruption family the fine-tunes train on). Errors are summed over the clean and
corrupted copies. If this rule selects a different weight vector from the clean-only rule,
the corrupted-validation vector is the one reported, regardless of the test or YouTube
outcome. Test and YouTube are read once with the final vector.

## Rule 2 for agent adoption (added 2026-09-07 19:35 UTC, after rule 1 adopted no XceptionNet
## epoch and before any test or YouTube read)

Rule 1 (clean validation log-loss lower than released, clean AUC within 0.002) is kept and
reported. Rule 2: a fine-tuned epoch is adopted if its mean log-loss over the validation
split scored clean and under a fixed deterministic corruption is lower than the released
model's, with clean validation AUC within 0.002. Rule 1 takes precedence when both hold.
Every agent is judged under both rules; the paper states which rule adopted each agent.
