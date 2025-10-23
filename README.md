# EndoMonST3R — LoRA Fine-Tuning + Dynamic Global Optimization for Monoscopic Endoscopy

This repository adapts **MonST3R** (built on DUSt3R) to **EndoSLAM** and **C3VD** by inserting **LoRA adapters** **only** in the **decoder** and **pointmap/confidence heads**. Pairwise training uses the native DUSt3R/MonST3R point-map + confidence losses, and evaluation reports **SI-log-RMSE** on (A) raw pairwise predictions and (B) **dynamic global optimization** outputs that stitch pairwise pointmaps into a global, per-frame reconstruction with trajectory smoothness and flow-projection terms.

---

## What’s in this repo

```
.
├─ dust3r/                      # MonST3R/DUSt3R core (model, heads, losses, cloud_opt, utils, training)
│  ├─ losses.py                 # Regr3D*, ConfLoss, etc. (pairwise training losses)
│  └─ cloud_opt/                # Dynamic global optimization (alignment + smoothness + flow projection)
├─ lora_monst3r.py              # LoRA injection (decoder + heads), native losses adapter, datasets, train/val
├─ train.py                     # Train entrypoint (LoRA fine-tuning, logs, checkpoints, loss curves)
├─ eval.py                      # SI-log-RMSE for pairwise and global-optimized depth in dynamic sequences
├─ endoslam_dataset.py          # EndoSLAM loader (if you use your own, keep this path)
├─ train_test_split_endoslam.py # EndoSLAM splits (train/val/test)
├─ train_test_split_c3vd.py     # C3VD splits (train/val/test)
└─ assets/
   ├─ fig1_teaser.png
   └─ results/                  # Quantitative tables & qualitative figures
```

---

## How MonST3R helps in endoscopy

Endoscopic video is **dynamic**, **non-Lambertian** (specular tissues), and often **low-texture**. MonST3R directly regresses **per-pixel 3D pointmaps** and **confidence** via a transformer decoder with cross-attention between frames, avoiding brittle multi-stage pipelines. Fine-tuning with **LoRA** on the **decoder + heads** lets us adapt geometry prediction to endoscopy while keeping the encoder’s general vision priors intact.

---

## LoRA scope (decoder + heads only)

We wrap **only** `nn.Linear` layers whose qualified names contain any of:

```
decoder, head, pointmap, conf, confidence, reg, regressor, pred
```

and **exclude** any path containing:

```
encoder
```

This freezes the encoder while adapting geometric reasoning and uncertainty calibration where it matters.

---

## Losses used for pairwise training

**Robust 3D regression** on pointmaps with **confidence weighting** (heteroscedastic). Let ( \mathbf{P}_i ) be predicted 3D points, ( \mathbf{P}_i^{*} ) ground truth, and ( c_i \in (0,1] ) confidence.

Robust regression:
$$
\mathcal{L}_{\text{reg}}
========================

\frac{1}{N}\sum_{i=1}^{N}
\rho!\left(\left\lVert \mathbf{P}_i - \mathbf{P}_i^{*} \right\rVert_2\right),
\qquad
\rho(x)=\sqrt{x^2+\varepsilon^2},.
$$

Confidence-weighted term (heteroscedastic):
$$
\mathcal{L}_{\text{cw}}
=======================

\frac{1}{N}\sum_{i=1}^{N}
\exp!\left(-s_i\right)
\left\lVert \mathbf{P}_i - \mathbf{P}_i^{*} \right\rVert_2^2
+\lambda, s_i,
\qquad
s_i = -\log c_i,.
$$

Total:
$$
\mathcal{L}
===========

\alpha,\mathcal{L}*{\text{reg}}
+
\beta,\mathcal{L}*{\text{cw}}
+
\gamma,\mathcal{L}_{\text{aux}} , .
$$

Shift/scale-invariant variants are used as provided by `dust3r/losses.py` (e.g., `Regr3D_ScaleShiftInv(L21)`), and wrapped by `ConfLoss` at pixel level.

---

## Dynamic global optimization (video mode)

Pairwise pointmaps (X_{t;t'}) and (X_{t';t}) are **accumulated into a global frame** to recover per-frame **global pointmaps** (X_t) and **camera poses** (P_t = [R_t|T_t]). The global objective combines:

Alignment across edges:
$$
L_{\text{align}}(X, \sigma, P^W)
================================

\sum_{W_i \in W}
\sum_{e \in W_i}
\sum_{t \in e}
\left\lVert
C_{t;e}\cdot
\bigl(
X_t - \sigma_e, P_{t;e}^{}, X_{t;e}
\bigr)
\right\rVert_1 .
$$

Trajectory smoothness:
$$
L_{\text{smooth}}(X)
====================

\sum_{t}
\left\lVert R_t^\top R_{t+1} - I \right\rVert_F
+
\left\lVert T_{t+1}-T_t \right\rVert_2 .
$$

Flow projection (camera-induced vs. estimated flow over confident static regions):
$$
L_{\text{flow}}(X)
==================

\sum_{W_i \in W}
\sum_{t\rightarrow t' \in W_i}
\left\lVert
S^{\text{global}}*{t\rightarrow t'} \cdot
\left(
F^{\text{global,cam}}*{t\rightarrow t'} -
F^{\text{est}}_{t\rightarrow t'}
\right)
\right\rVert_1 .
$$

The optimizer in `dust3r/cloud_opt/` solves
$$
\hat{X}
=======

\arg\min_{X,,P^W,,\sigma}
;
L_{\text{align}} + w_{\text{smooth}},L_{\text{smooth}} + w_{\text{flow}},L_{\text{flow}} .
$$

---

## Metric: SI-log-RMSE (monocular scale robustness)

Given predicted depth (\hat D_i) and ground truth (D_i) at valid pixels:

$$
\mathrm{SI}\text{-}\log\mathrm{RMSE}
====================================

\sqrt{
\frac{1}{n}\sum_{i=1}^{n}
\bigl(\log \hat{D}_i - \log D_i\bigr)^2
---------------------------------------

\frac{1}{n^2}
\left(\sum_{i=1}^{n}\bigl(\log \hat{D}_i - \log D_i\bigr)\right)^2
}, .
$$

We report SI-log-RMSE for (A) **pairwise** depths (direct Z from pointmaps in the anchor frame) and (B) **global-optimized** depths (re-projected from global pointmaps with optimized poses).

---

## Training (LoRA fine-tuning)

Use **native DUSt3R/MonST3R losses** while adapting only decoder + heads.

```bash
python train.py \
  --dataset endoslam \
  --data-root /path/to/endoslam_processed \
  --split-train train --split-val val \
  --arch base --ckpt /path/to/pretrained_monst3r_or_dust3r.pt \
  --image-size 384 \
  --batch-size 4 --epochs 15 --lr 1e-4 \
  --lora-rank 8 --lora-alpha 16 --lora-dropout 0.05 \
  --train-norms \
  --alpha-conf 1.0 --norm-mode avg_dis \
  --output-dir outputs/endoslam_monst3r_lora
```

For C3VD, set `--dataset c3vd` and the corresponding `--data-root` and split folder.

**Outputs**

* `outputs/.../best.pt`, `outputs/.../last.pt`
* `outputs/.../loss_curve.png`

---

## Evaluation (pairwise vs. dynamic global optimization)

Compute SI-log-RMSE for both **pairwise** and **optimized** reconstructions:

```bash
python eval.py \
  --dataset endoslam \
  --data-root /path/to/endoslam_processed \
  --split val \
  --image-size 384 \
  --arch base \
  --ckpt outputs/endoslam_monst3r_lora/best.pt \
  --device cuda \
  --output-csv outputs/eval_si_log_rmse_endoslam.csv
```

This will:

1. Run consecutive pairs to get **pairwise** pointmaps and compute **pairwise SI-log-RMSE**.
2. Accumulate all pairwise outputs within each trajectory and run **cloud optimizer** (`dust3r/cloud_opt`) to produce **global** pointmaps & poses; re-project to depth and compute **optimized SI-log-RMSE**.
3. Save per-trajectory and overall means in the CSV.

---

## Files added/updated

* **`lora_monst3r.py`**
  LoRA modules (decoder + heads), dataset wrappers (EndoSLAM/C3VD via split files), native loss adapter (`Regr3D_ScaleShiftInv(L21).with_reduction('none')` wrapped by `ConfLoss`), train/val loops, checkpointing.

* **`train.py`**
  CLI for LoRA fine-tuning (hyper-params for LoRA and loss), logging, curve plotting.

* **`eval.py`**
  SI-log-RMSE evaluation for **pairwise** and **global-optimized** depths; integrates with `dust3r/cloud_opt` to run alignment + smoothness + flow-projection optimization.

---

## Benchmarks (from the attached report)

**EndoSLAM (UnityCam)** — RMSE (m), lower is better:

| Model          | Colon      | Intestine  | Stomach    |
| -------------- | ---------- | ---------- | ---------- |
| EndoSfMLearner | 0.0064     | 0.02398    | 0.0126     |
| **MonST3R**    | **0.0062** | **0.0199** | **0.0108** |
| MonoLoT        | 0.0091     | 0.0223     | 0.0431     |

**C3VD** — RMSE (m):

| Setting               | RMSE (m)   |
| --------------------- | ---------- |
| EndoSfMLearner        | 12.0321    |
| MonST3R (before FT)   | 12.4161    |
| MonoLoT               | **0.1330** |
| **MonST3R (LoRA FT)** | **0.0760** |

(“FT” = LoRA fine-tuned decoder + heads. See the report for full tables and qualitative comparisons.)

