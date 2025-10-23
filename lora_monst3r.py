# lora_monst3r.py
# LoRA finetuning for MonST3R's DUSt3R-based model — ONLY decoder + pointmap/confidence heads.

import os
import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# -----------------------------
# Safe import helper
# -----------------------------
def try_import(module: str, name: Optional[str] = None):
    try:
        mod = __import__(module, fromlist=[name] if name else [])
        return getattr(mod, name) if name else mod
    except Exception:
        return None

# ==========================================
# Build MonST3R/DUSt3R model from local repo
# ==========================================
def build_monst3r(arch: str = "base", checkpoint: Optional[str] = None, device: str = "cuda"):
    """
    Attempts common builders in the MonST3R repo. Adjust if your local API differs.
    """
    model = None

    # MonST3R ships a DUSt3R fork under dust3r/, with builder patterns:
    #  - dust3r.build.build_model(arch)
    #  - dust3r.model.get_model(arch)
    # If MonST3R exposes a top-level factory, you can add it below.
    build_mod = try_import("dust3r.build")
    if build_mod and hasattr(build_mod, "build_model"):
        model = build_mod.build_model(arch)

    if model is None:
        model_mod = try_import("dust3r.model")
        if model_mod and hasattr(model_mod, "get_model"):
            model = model_mod.get_model(arch)

    if model is None:
        raise RuntimeError("Could not locate a dust3r/MonST3R builder. Edit build_monst3r() to your local API.")

    model.to(device)

    # (Optional) load pretrained MonST3R/DUSt3R weights
    if checkpoint and os.path.isfile(checkpoint):
        ckpt = torch.load(checkpoint, map_location=device)
        sd = ckpt.get("state_dict", ckpt)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[Load] Missing: {len(missing)}  Unexpected: {len(unexpected)}")

    return model

# =================
# LoRA for Linear
# =================
class LoRALinear(nn.Module):
    """
    W(x) + scale * B(A(x)), with base W frozen.
    """
    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        self.base = base
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r if r > 0 else 1.0
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if r > 0:
            self.A = nn.Linear(base.in_features, r, bias=False)
            self.B = nn.Linear(r, base.out_features, bias=False)
            nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.B.weight)
        else:
            self.register_parameter("A", None)
            self.register_parameter("B", None)

        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x):
        y = self.base(x)
        if self.r > 0:
            y = y + self.drop(self.B(self.A(x))) * self.scaling
        return y

def _qual_named_children(m: nn.Module, prefix: str = ""):
    for n, ch in m.named_children():
        fq = f"{prefix}.{n}" if prefix else n
        yield fq, ch
        yield from _qual_named_children(ch, fq)

def inject_lora_decoder_and_heads(
    model: nn.Module,
    r: int,
    alpha: int,
    dropout: float,
    verbose: bool = True
):
    """
    Wrap ONLY decoder and head Linear layers with LoRA:
      - Include paths containing: 'decoder' OR head keywords
      - Exclude anything containing: 'encoder'
    Head keywords cover pointmap & confidence heads.
    """
    include_keywords = ["decoder", "head", "pointmap", "conf", "confidence", "reg", "regressor", "pred"]
    exclude_keywords = ["encoder"]

    replaced = 0
    for fq, ch in list(_qual_named_children(model)):
        # Skip encoders entirely
        if any(kw in fq.lower() for kw in exclude_keywords):
            continue

        if isinstance(ch, nn.Linear):
            lower = fq.lower()
            if any(kw in lower for kw in include_keywords):
                # Replace this Linear with LoRA-wrapped Linear
                parent = model
                parts = fq.split(".")
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                setattr(parent, parts[-1], LoRALinear(ch, r=r, alpha=alpha, dropout=dropout))
                replaced += 1
                if verbose:
                    print(f"[LoRA] Wrapped Linear at: {fq}")

    if verbose:
        print(f"[LoRA] Total wrapped Linear layers (decoder+heads only): {replaced}")
    return model

def mark_trainable_lora_and_norms(model: nn.Module, train_norms: bool = False):
    trainable, total = 0, 0
    for n, p in model.named_parameters():
        needs_grad = ("A.weight" in n) or ("B.weight" in n)
        if train_norms and any(k in n.lower() for k in ["norm.weight", "norm.bias", "layernorm", "ln.weight", "ln.bias", "bn.weight", "bn.bias"]):
            needs_grad = True
        p.requires_grad = needs_grad
        total += p.numel()
        if needs_grad:
            trainable += p.numel()
    print(f"[LoRA] Trainable params: {trainable:,} / {total:,}")
    return model

# =================
# Datasets / Pairs
# =================

def _img_to_tensor(img: Image.Image) -> torch.Tensor:
    return transforms.ToTensor()(img)

def _normalize(t: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std  = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    return (t - mean) / std

class GenericFramesDataset(Dataset):
    """
    Fallback dataset if your project-specific datasets aren't importable.
    Expects split files like:
      - root/train_test_c3vd/{train,val}.txt    OR
      - root/train_test_endoslam/{train,val}.txt
    Each line: "<relative_trajectory_path> <frame_idx>"
    """
    def __init__(self, root: str, split: str, split_folder: str, image_size: int = 384, load_depth: bool = True):
        self.root = os.path.abspath(root)
        self.split = split
        self.load_depth = load_depth

        split_file = os.path.join(self.root, split_folder, f"{split}.txt")
        if not os.path.isfile(split_file):
            raise FileNotFoundError(f"Missing split file: {split_file}")

        with open(split_file, "r") as f:
            self.samples = [line.strip().split() for line in f if line.strip()]

        # Preload intrinsics per trajectory if available
        self.traj_intr = {}
        for rel_traj, _ in self.samples:
            if rel_traj not in self.traj_intr:
                intr_path = os.path.join(self.root, rel_traj, "intrinsics", "intrinsics.npy")
                self.traj_intr[rel_traj] = np.load(intr_path) if os.path.isfile(intr_path) else None

        self.resize = transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rel_traj, frame_str = self.samples[idx]
        frame_idx = int(frame_str)
        traj_dir = os.path.join(self.root, rel_traj)

        img_path = os.path.join(traj_dir, "images", f"{frame_idx:06d}.png")
        pose_path = os.path.join(traj_dir, "poses", f"{frame_idx:06d}.npy")
        depth_path = os.path.join(traj_dir, "depths", f"{frame_idx:06d}.npy")

        img = Image.open(img_path).convert("RGB")
        img = self.resize(img)
        img_t = _normalize(_img_to_tensor(img))

        pose = np.load(pose_path) if os.path.isfile(pose_path) else None
        depth = np.load(depth_path) if (self.load_depth and os.path.isfile(depth_path)) else None
        K = self.traj_intr.get(rel_traj, None)

        out = {
            "image": img_t,
            "trajectory": rel_traj,
            "frame_id": frame_idx
        }
        if pose is not None: out["pose"] = torch.from_numpy(pose).float()
        if depth is not None: out["depth"] = torch.from_numpy(depth).float()
        if K is not None: out["intrinsics"] = torch.from_numpy(K).float()
        return out

# Try user's concrete dataset classes if they exist
C3VDDataset = try_import("datasets.c3vd_dataset", "C3VDDataset") or try_import("datasets.c3vd", "C3VDDataset")
EndoSLAMDataset = try_import("datasets.endoslam_dataset", "EndoSLAMDataset") or try_import("datasets.endoslam", "EndoSLAMDataset")

class PairFromSinglesDataset(Dataset):
    """
    Wrap a single-frame dataset to yield pairs from the SAME trajectory.
    Positive selection within ±window frames when possible.
    """
    def __init__(self, base_ds: Dataset, image_size: int = 384, pair_window: int = 10):
        self.base = base_ds
        self.window = pair_window

        self.by_traj: Dict[str, List[int]] = {}
        for i in range(len(self.base)):
            item = self.base[i]
            traj = item["trajectory"]
            self.by_traj.setdefault(traj, []).append(i)

        for traj, idxs in self.by_traj.items():
            idxs.sort(key=lambda j: self.base[j].get("frame_id", j))

    def __len__(self):
        return len(self.base)

    def _pick_positive_index(self, i_anchor: int) -> int:
        a = self.base[i_anchor]
        traj = a["trajectory"]
        idxs = self.by_traj[traj]
        if len(idxs) == 1:
            return i_anchor
        if "frame_id" in a:
            fid = a["frame_id"]
            candidates = [j for j in idxs if j != i_anchor and abs(self.base[j].get("frame_id", fid) - fid) <= self.window]
            if not candidates:
                candidates = [j for j in idxs if j != i_anchor]
        else:
            candidates = [j for j in idxs if j != i_anchor]
        return random.choice(candidates)

    def __getitem__(self, idx):
        a = self.base[idx]
        j = self._pick_positive_index(idx)
        b = self.base[j]
        return {
            "image1": a["image"],
            "image2": b["image"],
            "meta": {
                "traj": a["trajectory"],
                "fid1": a.get("frame_id", -1),
                "fid2": b.get("frame_id", -1),
                "K1": a.get("intrinsics"),
                "K2": b.get("intrinsics"),
                "pose1": a.get("pose"),
                "pose2": b.get("pose"),
                "depth1": a.get("depth"),
                "depth2": b.get("depth"),
            }
        }

def build_dataset(dataset: str, root: str, split: str, image_size: int) -> Dataset:
    dataset = dataset.lower()
    if dataset == "c3vd":
        if C3VDDataset is not None:
            base = C3VDDataset(root, split=split, transform=transforms.Resize((image_size, image_size)))
        else:
            base = GenericFramesDataset(root=root, split=split, split_folder="train_test_c3vd", image_size=image_size)
    elif dataset == "endoslam":
        if EndoSLAMDataset is not None:
            base = EndoSLAMDataset(root, split=split, transform=transforms.Resize((image_size, image_size)))
        else:
            base = GenericFramesDataset(root=root, split=split, split_folder="train_test_endoslam", image_size=image_size)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    return PairFromSinglesDataset(base, image_size=image_size, pair_window=10)

# ===========================
# Loss: fallback contrastive
# ===========================
class SimpleContrastivePairLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.t = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        if z1.dim() > 2:
            z1 = z1.mean(dim=tuple(range(1, z1.dim()-1)))
            z2 = z2.mean(dim=tuple(range(1, z2.dim()-1)))
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        logits = z1 @ z2.t() / self.t
        labels = torch.arange(z1.size(0), device=z1.device)
        return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))

# ===========================================
# Wrapper: use native loss if model provides it
# ===========================================
class Monst3rWithLoss(nn.Module):
    def __init__(self, core: nn.Module, fallback_temperature: float = 0.07):
        super().__init__()
        self.core = core
        self.fallback = SimpleContrastivePairLoss(fallback_temperature)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        # Try MonST3R/DUSt3R native forward
        try:
            out = self.core(batch)
        except TypeError:
            out = self.core(batch["image1"], batch["image2"])

        if isinstance(out, dict) and "loss" in out:
            loss = out["loss"]
            logs = {k: (float(v) if torch.is_tensor(v) else v) for k, v in out.items() if k != "loss"}
            return {"loss": loss, "logs": logs}

        # Otherwise look for embeddings to apply contrastive fallback
        z1 = out.get("z1") if isinstance(out, dict) else None
        z2 = out.get("z2") if isinstance(out, dict) else None
        if z1 is None or z2 is None:
            if hasattr(self.core, "encode"):
                z1 = self.core.encode(batch["image1"])
                z2 = self.core.encode(batch["image2"])
            else:
                raise RuntimeError("Cannot infer embeddings from MonST3R output. Adapt wrapper to your local API.")

        loss = self.fallback(z1, z2)
        return {"loss": loss, "logs": {"contrastive_loss": float(loss.detach().item())}}

# =========================
# Train / Val / Checkpoint
# =========================
@dataclass
class TrainConfig:
    epochs: int = 10
    batch_size: int = 4
    lr: float = 1e-4
    weight_decay: float = 0.0
    num_workers: int = 8
    grad_clip: float = 1.0
    log_every: int = 25
    device: str = "cuda"

def collate_fn(batch):
    return {
        "image1": torch.stack([b["image1"] for b in batch], dim=0),
        "image2": torch.stack([b["image2"] for b in batch], dim=0),
        "meta": [b["meta"] for b in batch]
    }

def make_loader(ds: Dataset, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      pin_memory=True, collate_fn=collate_fn)

def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
                    scaler: Optional[torch.cuda.amp.GradScaler], cfg: TrainConfig) -> Dict[str, float]:
    model.train()
    device = cfg.device
    losses = []
    for it, batch in enumerate(loader):
        batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=device.startswith("cuda")):
            out = model(batch)
            loss = out["loss"]
        if scaler is not None:
            scaler.scale(loss).backward()
            if cfg.grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if cfg.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
        losses.append(loss.detach().item())
        if (it + 1) % cfg.log_every == 0:
            window = losses[-cfg.log_every:]
            print(f"  iter {it+1}/{len(loader)} | loss {sum(window)/len(window):.4f}")
    return {"loss": float(sum(losses) / max(1, len(losses)))}

@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, cfg: TrainConfig) -> Dict[str, float]:
    model.eval()
    device = cfg.device
    losses = []
    for batch in loader:
        batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
        out = model(batch)
        losses.append(out["loss"].detach().item())
    return {"loss": float(sum(losses) / max(1, len(losses)))}

def save_checkpoint(path: str, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, extra: Dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "extra": extra}, path)
    print(f"[CKPT] saved: {path}")

def build_lora_monst3r(
    arch: str,
    checkpoint: Optional[str],
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
    train_norms: bool,
    device: str
) -> nn.Module:
    core = build_monst3r(arch=arch, checkpoint=checkpoint, device=device)
    core = inject_lora_decoder_and_heads(core, r=lora_rank, alpha=lora_alpha, dropout=lora_dropout, verbose=True)
    core = mark_trainable_lora_and_norms(core, train_norms=train_norms)
    return Monst3rWithLoss(core)
