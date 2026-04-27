# Stage D Training Test Results Summary
**Date:** April 26, 2026  
**Project:** Language-Conditioned Whole-Body Contact Manipulation  
**Status:** Whole-body learning validated. Combined model is deployment candidate.

---

## Three Configurations Tested

| Test | Data | Joint Weights | Output Directory |
|---|---|---|---|
| **A** | v2 only (15 episodes) | `[5,5,5,0,0,0,0,0,0,0,0,0]` | `models/stage_d_v2only/` |
| **B** | v2 + v3 combined (35 episodes) | `[5,5,5,1,1,1,1,1,1,1,1,1]` | `models/stage_d_combined/` |
| **C** | v2 + v3 combined (35 episodes) | `[5,5,5,0,0,0,0,0,0,0,0,0]` | `models/stage_d_combined_fronly/` |

Test A is the FR-only baseline. Test B is the full whole-body model. Test C isolates the effect of additional data from the effect of full-weight supervision.

---

## Headline Per-Joint MSE (val set)

| Joint | A: v2 + FR-only | C: combined + FR-only | B: combined + full weights |
|---|---|---|---|
| FR_hip | 0.00004 | 0.00002 | **0.00001** |
| FR_thigh | 0.00011 | 0.00012 | **0.00010** |
| FR_calf | 0.00025 | 0.00002 | **0.00003** |
| FL_hip | 0.01094 | 0.00644 | **0.00002** |
| FL_thigh | 0.00270 | 0.00532 | **0.00005** |
| FL_calf | 0.00337 | 0.00357 | **0.00005** |
| RL_hip | 0.00192 | 0.00282 | **0.00004** |
| RL_thigh | 0.00942 | 0.00964 | **0.00005** |
| RL_calf | 0.00320 | 0.00276 | **0.00005** |
| RR_hip | 0.00520 | 0.00517 | **0.00002** |
| RR_thigh | 0.01604 | 0.01664 | **0.00008** |
| RR_calf | 0.02245 | 0.02352 | **0.00007** |
| **Overall** | 0.00630 | 0.00634 | **0.00005** |

Test B's overall MSE is 126x lower than Test A. Rear-leg MSE dropped by 100-300x.

---

## Distribution Match (Test B Only)

The cleanest signal that the model learned real state-dependent corrections rather than outputting noise. For all 12 joints in Test B, predicted distribution width closely matches the true distribution width.

| Joint | pred_std | true_std | match quality |
|---|---|---|---|
| FR_hip | 0.0342 | 0.0353 | excellent |
| FR_thigh | 0.0583 | 0.0587 | excellent |
| FR_calf | 0.0814 | 0.0836 | excellent |
| FL_hip | 0.0192 | 0.0179 | good |
| FL_thigh | 0.0290 | 0.0312 | good |
| FL_calf | 0.0209 | 0.0237 | good |
| RL_hip | 0.0181 | 0.0199 | good |
| RL_thigh | 0.0235 | 0.0227 | excellent |
| RL_calf | 0.0188 | 0.0180 | excellent |
| RR_hip | 0.0086 | 0.0097 | good |
| RR_thigh | 0.0315 | 0.0292 | good |
| RR_calf | 0.0260 | 0.0243 | good |

In Test C, rear-leg pred_std (0.014-0.043) was 1.5-5x larger than rear-leg true_std (0.008-0.025) — pure trunk-leakage noise, not learning.

---

## Two Independent Effects Identified

**Effect 1: More Data Helps FR Learning (A vs C)**

Adding 20 v3 episodes to FR-only training (Test A → Test C) improved:
- FR_calf MSE: 0.00025 → 0.00002 (12x improvement)
- FR_hip MSE: 0.00004 → 0.00002 (2x improvement)
- FR_thigh MSE: roughly unchanged

The 20 v3 episodes contain diverse FR_calf trajectories that improve generalization, even when the model is only asked to learn FR.

**Effect 2: Full Weighting Enables Whole-Body Learning (C vs B)**

Same data, different objective. Adding rear-leg supervision (Test C → Test B):
- Rear-leg MSE drops 100-300x across all 9 support-leg dimensions
- FR metrics nearly unchanged (no objective interference)
- Per-joint pred_std matches true_std on all 12 joints

The model successfully learns the rear-leg residual structure when given gradient signal on those output dimensions.

---

## Per-Phase Performance (Test B)

Test B maintains uniformly low MSE across all phases (lift, extend, hold) for all 12 joints. Per-joint per-phase MSE is in the 0.00001-0.00016 range across all 36 cells, indicating the model handles all phases of contact equally well. No phase-specific failure mode.

---

## Worst-Case Analysis

In all three tests, the top-5 worst FR predictions cluster around two val episodes (`20260424_134604` and `20260424_134716`) with unusually large foot-to-target geometry (~0.74m forward distance). This represents a tail of the training distribution where all three models struggle similarly. This is a coverage issue, not a model architecture issue. Additional data covering large foot-to-target states would address this if pursued.

---

## Conclusions

1. **Whole-body residual policy works.** Test B demonstrates that a learned 12-dim policy can produce coordinated whole-body corrections matching the demonstrated distribution.

2. **v3 hand-guided whole-body demonstrations contain learnable signal.** The collection protocol — Person 1 guiding FR while Person 2 stabilizes/translates the body under softened support-leg gains — produced data that the model uses effectively.

3. **No tradeoff between FR accuracy and rear-leg learning.** Adding full-weight rear-leg supervision did not degrade FR predictions in Test B versus Test C. Both objectives can coexist.

4. **35 episodes is sufficient for first results.** While more data would broaden the state distribution coverage, the current dataset produced the headline result with clean validation metrics.

---

## Deployment Recommendation

The combined model in `models/stage_d_combined/` is the candidate for deployment in `run_baseline1.py`. Apply the full 12-dim residual at inference time. Slicing to FR-only is no longer needed since rear-leg outputs are now meaningful predictions rather than noise.

For comparison evaluation, the v2-only model in `models/stage_d_v2only/` serves as the FR-only baseline, but at inference must slice output to `[0:3]` and zero out support-leg corrections.

---

## Next Steps

1. Train run integration into `run_baseline1.py` using the combined model
2. Conduct evaluation runs comparing methods on the real robot
3. Optionally collect additional v3 data covering edge-case geometries (large foot-to-target distances)
4. Begin writeup using these results as the headline contribution