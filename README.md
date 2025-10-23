EndoMonST3R — LoRA Fine-Tuning + Dynamic Global Optimization for Monoscopic Endoscopy

This repository adapts MonST3R (built on DUSt3R) to EndoSLAM and C3VD by inserting LoRA adapters only in the decoder and pointmap/confidence heads. Pairwise training uses the native DUSt3R/MonST3R point-map + confidence losses, and evaluation reports SI-log-RMSE on (A) raw pairwise predictions and (B) dynamic global optimization outputs that stitch pairwise pointmaps into a global, per-frame reconstruction with trajectory smoothness and flow-projection terms.

What’s in this repo
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

How MonST3R helps in endoscopy

Endoscopic video is dynamic, non-Lambertian (specular tissues), and often low-texture. MonST3R directly regresses per-pixel 3D pointmaps and confidence via a transformer decoder with cross-attention between frames, avoiding brittle multi-stage pipelines. Fine-tuning with LoRA on the decoder + heads lets us adapt geometry prediction to endoscopy while keeping the encoder’s general vision priors intact.

LoRA scope (decoder + heads only)

We wrap only nn.Linear layers whose qualified names contain any of:

decoder, head, pointmap, conf, confidence, reg, regressor, pred


and exclude any path containing:

encoder


This freezes the encoder while adapting geometric reasoning and uncertainty calibration where it matters.

Losses used for pairwise training

Robust 3D regression on pointmaps with confidence weighting (heteroscedastic). Let 
𝑃
𝑖
P
i
	​

 be predicted 3D points, 
𝑃
𝑖
∗
P
i
∗
	​

 ground truth, and 
𝑐
𝑖
∈
(
0
,
1
]
c
i
	​

∈(0,1] confidence.

Robust regression:

𝐿
reg
=
1
𝑁
∑
𝑖
=
1
𝑁
𝜌
 ⁣
(
∥
𝑃
𝑖
−
𝑃
𝑖
∗
∥
2
)
,
𝜌
(
𝑥
)
=
𝑥
2
+
𝜀
2
 
.
L
reg
	​

=
N
1
	​

i=1
∑
N
	​

ρ(∥P
i
	​

−P
i
∗
	​

∥
2
	​

),ρ(x)=
x
2
+ε
2
	​

.

Confidence-weighted term (heteroscedastic):

𝐿
cw
=
1
𝑁
∑
𝑖
=
1
𝑁
exp
⁡
 ⁣
(
−
𝑠
𝑖
)
∥
𝑃
𝑖
−
𝑃
𝑖
∗
∥
2
2
+
𝜆
 
𝑠
𝑖
,
𝑠
𝑖
=
−
log
⁡
𝑐
𝑖
 
.
L
cw
	​

=
N
1
	​

i=1
∑
N
	​

exp(−s
i
	​

)∥P
i
	​

−P
i
∗
	​

∥
2
2
	​

+λs
i
	​

,s
i
	​

=−logc
i
	​

.

Total:

𝐿
=
𝛼
 
𝐿
reg
+
𝛽
 
𝐿
cw
+
𝛾
 
𝐿
aux
 
.
L=αL
reg
	​

+βL
cw
	​

+γL
aux
	​

.

Shift/scale-invariant variants are used as provided by dust3r/losses.py (e.g., Regr3D_ScaleShiftInv(L21)), and wrapped by ConfLoss at pixel level.

Dynamic global optimization (video mode)

Pairwise pointmaps 
𝑋
𝑡
;
𝑡
′
X
t;t
′
	​

 and 
𝑋
𝑡
′
;
𝑡
X
t
′
;t
	​

 are accumulated into a global frame to recover per-frame global pointmaps 
𝑋
𝑡
X
t
	​

 and camera poses 
𝑃
𝑡
=
[
𝑅
𝑡
∣
𝑇
𝑡
]
P
t
	​

=[R
t
	​

∣T
t
	​

]. The global objective combines:

Alignment across edges:

𝐿
align
(
𝑋
,
𝜎
,
𝑃
𝑊
)
=
∑
𝑊
𝑖
∈
𝑊
∑
𝑒
∈
𝑊
𝑖
∑
𝑡
∈
𝑒
∥
𝐶
𝑡
;
𝑒
⋅
(
𝑋
𝑡
−
𝜎
𝑒
 
𝑃
𝑡
;
𝑒
 
𝑋
𝑡
;
𝑒
)
∥
1
.
L
align
	​

(X,σ,P
W
)=
W
i
	​

∈W
∑
	​

e∈W
i
	​

∑
	​

t∈e
∑
	​

	​

C
t;e
	​

⋅(X
t
	​

−σ
e
	​

P
t;e
	​

X
t;e
	​

)
	​

1
	​

.

Trajectory smoothness:

𝐿
smooth
(
𝑋
)
=
∑
𝑡
∥
𝑅
𝑡
⊤
𝑅
𝑡
+
1
−
𝐼
∥
𝐹
+
∥
𝑇
𝑡
+
1
−
𝑇
𝑡
∥
2
.
L
smooth
	​

(X)=
t
∑
	​

	​

R
t
⊤
	​

R
t+1
	​

−I
	​

F
	​

+∥T
t+1
	​

−T
t
	​

∥
2
	​

.

Flow projection (camera-induced vs. estimated flow over confident static regions):

𝐿
flow
(
𝑋
)
=
∑
𝑊
𝑖
∈
𝑊
∑
𝑡
→
𝑡
′
∈
𝑊
𝑖
∥
𝑆
𝑡
→
𝑡
′
global
⋅
(
𝐹
𝑡
→
𝑡
′
global,cam
−
𝐹
𝑡
→
𝑡
′
est
)
∥
1
.
L
flow
	​

(X)=
W
i
	​

∈W
∑
	​

t→t
′
∈W
i
	​

∑
	​

	​

S
t→t
′
global
	​

⋅(F
t→t
′
global,cam
	​

−F
t→t
′
est
	​

)
	​

1
	​

.

The optimizer in dust3r/cloud_opt/ solves

𝑋
^
=
arg
⁡
min
⁡
𝑋
,
 
𝑃
𝑊
,
 
𝜎
  
𝐿
align
+
𝑤
smooth
 
𝐿
smooth
+
𝑤
flow
 
𝐿
flow
.
X
^
=arg
X,P
W
,σ
min
	​

L
align
	​

+w
smooth
	​

L
smooth
	​

+w
flow
	​

L
flow
	​

.
Metric: SI-log-RMSE (monocular scale robustness)

Given predicted depth 
𝐷
^
𝑖
D
^
i
	​

 and ground truth 
𝐷
𝑖
D
i
	​

 at valid pixels:

S
I
-
log
⁡
R
M
S
E
=
1
𝑛
∑
𝑖
=
1
𝑛
(
log
⁡
𝐷
^
𝑖
−
log
⁡
𝐷
𝑖
)
2
−
1
𝑛
2
(
∑
𝑖
=
1
𝑛
(
log
⁡
𝐷
^
𝑖
−
log
⁡
𝐷
𝑖
)
)
2
 
.
SI-logRMSE=
n
1
	​

i=1
∑
n
	​

(log
D
^
i
	​

−logD
i
	​

)
2
−
n
2
1
	​

(
i=1
∑
n
	​

(log
D
^
i
	​

−logD
i
	​

))
2
	​

.

We report SI-log-RMSE for (A) pairwise depths (direct Z from pointmaps in the anchor frame) and (B) global-optimized depths (re-projected from global pointmaps with optimized poses).

Training (LoRA fine-tuning)

Use native DUSt3R/MonST3R losses while adapting only decoder + heads.

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


For C3VD, set --dataset c3vd and the corresponding --data-root and split folder.

Outputs

outputs/.../best.pt, outputs/.../last.pt

outputs/.../loss_curve.png

Evaluation (pairwise vs. dynamic global optimization)

Compute SI-log-RMSE for both pairwise and optimized reconstructions:

python eval.py \
  --dataset endoslam \
  --data-root /path/to/endoslam_processed \
  --split val \
  --image-size 384 \
  --arch base \
  --ckpt outputs/endoslam_monst3r_lora/best.pt \
  --device cuda \
  --output-csv outputs/eval_si_log_rmse_endoslam.csv


This will:

Run consecutive pairs to get pairwise pointmaps and compute pairwise SI-log-RMSE.

Accumulate all pairwise outputs within each trajectory and run cloud optimizer (dust3r/cloud_opt) to produce global pointmaps & poses; re-project to depth and compute optimized SI-log-RMSE.

Save per-trajectory and overall means in the CSV.

Files added/updated

lora_monst3r.py
LoRA modules (decoder + heads), dataset wrappers (EndoSLAM/C3VD via split files), native loss adapter (Regr3D_ScaleShiftInv(L21).with_reduction('none') wrapped by ConfLoss), train/val loops, checkpointing.

train.py
CLI for LoRA fine-tuning (hyper-params for LoRA and loss), logging, curve plotting.

eval.py
SI-log-RMSE evaluation for pairwise and global-optimized depths; integrates with dust3r/cloud_opt to run alignment + smoothness + flow-projection optimization.

Benchmarks (from the attached report)

EndoSLAM (UnityCam) — RMSE (m), lower is better:

Model	Colon	Intestine	Stomach
EndoSfMLearner	0.0064	0.02398	0.0126
MonST3R	0.0062	0.0199	0.0108
MonoLoT	0.0091	0.0223	0.0431

C3VD — RMSE (m):

Setting	RMSE (m)
EndoSfMLearner	12.0321
MonST3R (before FT)	12.4161
MonoLoT	0.1330
MonST3R (LoRA FT)	0.0760

(“FT” = LoRA fine-tuned decoder + heads. See the report for full tables and qualitative comparisons.)
