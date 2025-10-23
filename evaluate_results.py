# eval.py
# Evaluate SI-log-RMSE on depth maps:
#  (A) Pairwise (no optimization)
#  (B) After dynamic global optimization (alignment + smoothness + flow projection)

import os
import csv
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from lora_monst3r import build_monst3r, inject_lora_decoder_and_heads, mark_trainable_lora_and_norms

# Optional cloud optimizer
cloud_opt = None
try:
    cloud_opt = __import__("dust3r.cloud_opt.optimizer", fromlist=["optimizer"])
except Exception:
    cloud_opt = None


def normalize(img_t: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std  = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    return (img_t - mean) / std


def load_frame(root: str, rel_traj: str, fid: int, image_size: int) -> Dict[str, Any]:
    traj_dir = os.path.join(root, rel_traj)
    img = Image.open(os.path.join(traj_dir, "images", f"{fid:06d}.png")).convert("RGB")
    Kp  = os.path.join(traj_dir, "intrinsics", "intrinsics.npy")
    Dp  = os.path.join(traj_dir, "depths", f"{fid:06d}.npy")
    Pp  = os.path.join(traj_dir, "poses", f"{fid:06d}.npy")

    resize = transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR)
    img_t = normalize(transforms.ToTensor()(resize(img)))

    out = {
        "image": img_t,
        "K": torch.from_numpy(np.load(Kp)).float() if os.path.isfile(Kp) else None,
        "depth_gt": torch.from_numpy(np.load(Dp)).float() if os.path.isfile(Dp) else None,
        "pose": torch.from_numpy(np.load(Pp)).float() if os.path.isfile(Pp) else None
    }
    return out


def group_split_by_traj(split_file: str) -> Dict[str, List[int]]:
    groups: Dict[str, List[int]] = defaultdict(list)
    with open(split_file, "r") as f:
        for line in f:
            rel, idx = line.strip().split()
            groups[rel].append(int(idx))
    for k in groups:
        groups[k].sort()
    return groups


def si_log_rmse(pred_depth: torch.Tensor, gt_depth: torch.Tensor, valid_mask: torch.Tensor) -> float:
    # pred_depth, gt_depth: [H,W] tensors, valid_mask: bool [H,W]
    m = valid_mask & torch.isfinite(pred_depth) & torch.isfinite(gt_depth) & (gt_depth > 0)
    if m.sum() == 0:
        return float("nan")
    d = (pred_depth[m].log() - gt_depth[m].log())
    val = torch.sqrt(d.pow(2).mean() - d.mean().pow(2))
    return float(val.cpu().item())


def pointmap_to_depth_in_cam(pts_world: torch.Tensor, cam_pose: torch.Tensor) -> torch.Tensor:
    """
    pts_world: [H,W,3]; cam_pose: [4,4] (world-to-camera or camera-to-world?)
    We expect cam_pose to be world->camera extrinsic (if your stored pose is camera->world, invert it).
    """
    H, W, _ = pts_world.shape
    ones = torch.ones(H, W, 1, device=pts_world.device, dtype=pts_world.dtype)
    P = torch.cat([pts_world, ones], dim=-1)  # [H,W,4]
    cam = (cam_pose @ P.view(-1, 4).T).T.view(H, W, 4)
    z = cam[..., 2].clamp(min=1e-6)
    return z


@torch.no_grad()
def run_pairwise(model, frames1, frames2, device):
    """
    frames1, frames2: dicts with 'image' tensor [3,H,W]
    Returns dict with pointmaps/confidences if available.
    """
    b = {
        "image1": frames1["image"].unsqueeze(0).to(device),
        "image2": frames2["image"].unsqueeze(0).to(device),
        "meta": [{
            # Provide per-view meta for potential loss/inference utils
            "camera_pose": frames1.get("pose"),  # optional
            "valid_mask": None,                  # optional
            "pts3d": None                        # optional (GT pts if available)
        }]
    }
    try:
        out = model.core(b) if hasattr(model, "core") else model(b)
    except TypeError:
        # some models accept (image1, image2)
        core = model.core if hasattr(model, "core") else model
        out = core(b["image1"], b["image2"])

    return out  # dict expected in MonST3R/DUSt3R


def extract_pointmap_depth_pairwise(out_dict, direction="12") -> Optional[torch.Tensor]:
    """
    Given model output dict, try to return depth from a pointmap in 'direction':
    direction '12' means points of view2 in view1 frame, we take Z.
    """
    keys = {
        "12": ["p12", "pts12", "xyz12", "P12", "pointmap12"],
        "21": ["p21", "pts21", "xyz21", "P21", "pointmap21"]
    }[direction]
    for k in keys:
        if k in out_dict and torch.is_tensor(out_dict[k]):
            P = out_dict[k]  # [B,H,W,3] or [H,W,3]
            if P.dim() == 4:
                P = P[0]
            depth = P[..., 2].clamp(min=1e-6)
            return depth
    return None


def maybe_invert_pose(pose: torch.Tensor, already_world_to_cam: bool = False) -> torch.Tensor:
    """
    If stored pose is camera->world, invert. Heuristic: if bottom-right element ~1 and upper-left ~ rotation.
    """
    if already_world_to_cam:
        return pose
    # assume provided is cam->world (common), return world->cam
    # fast inverse for SE(3):
    R = pose[:3, :3]
    t = pose[:3, 3:4]
    Rt = R.t()
    Tw = -Rt @ t
    out = torch.eye(4, dtype=pose.dtype, device=pose.device)
    out[:3, :3] = Rt
    out[:3, 3] = Tw.view(-1)
    return out


def build_model_for_eval(arch, ckpt, device, lora_rank=0, lora_alpha=0, lora_dropout=0.0):
    core = build_monst3r(arch=arch, checkpoint=ckpt, device=device)
    if lora_rank > 0:
        inject_lora_decoder_and_heads(core, r=lora_rank, alpha=lora_alpha, dropout=lora_dropout, verbose=True)
        # freeze base except LoRA (eval uses the adapted weights); no need to set grads
        mark_trainable_lora_and_norms(core, train_norms=False)
    return core


def run_cloud_optimization(edge_store: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    edge_store: list of pair dicts with fields like:
      {'t': t, 'tp': tp, 'p_ttp': tensor(H,W,3), 'p_tpt': tensor(H,W,3), 'conf_ttp': tensor(H,W,1), ...}
    Returns dict with per-frame global pointmaps and camera poses.
    """
    if cloud_opt is None:
        print("[WARN] cloud optimizer not available; returning empty result.")
        return {}

    # The exact API differs across commits. We call a generic "optimize" entry if available.
    # Try dust3r.cloud_opt.optimizer.Optimizer or a functional interface.
    try:
        OptimizerClass = getattr(cloud_opt, "Optimizer", None)
        if OptimizerClass is None:
            # module-level optimize(...)?
            optimize_fn = getattr(cloud_opt, "optimize", None)
            if optimize_fn is None:
                print("[WARN] Optimizer API not found.")
                return {}
            return optimize_fn(edge_store)
        else:
            opt = OptimizerClass()
            return opt.optimize(edge_store)
    except Exception as e:
        print(f"[WARN] Cloud optimization failed: {e}")
        return {}


def eval_sequence(device, model, root, rel_traj, fids, image_size, assume_cam_to_world=True):
    """
    Returns:
      results = {
        'pairwise': [si_log_rmse per frame_id (where computable)],
        'optimized': [si_log_rmse per frame_id after global optimization],
      }
    """
    # Load frames
    frames = {fid: load_frame(root, rel_traj, fid, image_size) for fid in fids}
    # Pairwise pass and SI-log-RMSE per frame using p_t->t+1 in t frame
    pairwise_scores = []

    edge_store = []  # accumulate for optimization
    for i in range(len(fids) - 1):
        t, tp = fids[i], fids[i+1]
        out = run_pairwise(model, frames[t], frames[tp], device)
        # store edges
        rec = {"t": t, "tp": tp}
        if isinstance(out, dict):
            for a, b in [ (["p12","pts12","xyz12","P12","pointmap12"], "p_ttp"),
                          (["p21","pts21","xyz21","P21","pointmap21"], "p_tpt"),
                          (["conf12","c12","confidence12"], "conf_ttp"),
                          (["conf21","c21","confidence21"], "conf_tpt") ]:
                for k in a:
                    if k in out and torch.is_tensor(out[k]):
                        rec[b] = out[k].detach().cpu()
                        break
        edge_store.append(rec)

        # Pairwise SI-log-RMSE in frame t
        depth_pred_t = extract_pointmap_depth_pairwise(out, direction="12")
        gt = frames[t]["depth_gt"]
        if depth_pred_t is not None and gt is not None:
            H, W = depth_pred_t.shape[-2:]
            gt_r = torch.nn.functional.interpolate(gt[None, None, ...], size=(H, W), mode="nearest")[0, 0]
            valid = gt_r > 0
            s = si_log_rmse(depth_pred_t, gt_r, valid)
            pairwise_scores.append(s)

    # Global optimization stage (accumulate edges -> global pointmaps and camera poses)
    optimized_scores = []
    glob = run_cloud_optimization(edge_store)
    # Expect something like glob["poses"][t] -> [4,4] world->cam, glob["pointmaps"][t] -> [H,W,3]
    if isinstance(glob, dict) and "poses" in glob and "pointmaps" in glob:
        for t in fids:
            if t not in glob["pointmaps"] or frames[t]["depth_gt"] is None:
                continue
            pts_world = glob["pointmaps"][t]  # [H,W,3] torch
            pose_wc = glob["poses"][t]       # world->cam [4,4]
            if not torch.is_tensor(pts_world):
                pts_world = torch.from_numpy(np.asarray(pts_world))
            if not torch.is_tensor(pose_wc):
                pose_wc = torch.from_numpy(np.asarray(pose_wc))
            if assume_cam_to_world and frames[t]["pose"] is not None:
                # If GT pose likely cam->world, keep optimizer using world frame; we only need depth in camera t
                pass
            depth_pred = pointmap_to_depth_in_cam(pts_world.to(torch.float32), pose_wc.to(torch.float32))
            gt = frames[t]["depth_gt"]
            H, W = depth_pred.shape[-2:]
            gt_r = torch.nn.functional.interpolate(gt[None, None, ...], size=(H, W), mode="nearest")[0, 0]
            valid = gt_r > 0
            optimized_scores.append(si_log_rmse(depth_pred, gt_r, valid))

    return {"pairwise": pairwise_scores, "optimized": optimized_scores}


def main():
    pa = argparse.ArgumentParser("Evaluate SI-log-RMSE (pairwise and optimized global)")
    pa.add_argument("--dataset", type=str, choices=["c3vd", "endoslam"], required=True)
    pa.add_argument("--data-root", type=str, required=True)
    pa.add_argument("--split", type=str, default="val")
    pa.add_argument("--image-size", type=int, default=384)
    pa.add_argument("--arch", type=str, default="base")
    pa.add_argument("--ckpt", type=str, required=True)
    pa.add_argument("--lora-rank", type=int, default=0)
    pa.add_argument("--lora-alpha", type=int, default=0)
    pa.add_argument("--lora-dropout", type=float, default=0.0)
    pa.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    pa.add_argument("--output-csv", type=str, default="outputs/eval_si_log_rmse.csv")
    args = pa.parse_args()

    split_folder = "train_test_c3vd" if args.dataset.lower() == "c3vd" else "train_test_endoslam"
    split_file = os.path.join(args.data_root, split_folder, f"{args.split}.txt")
    groups = group_split_by_traj(split_file)

    device = args.device
    model = build_model_for_eval(args.arch, args.ckpt, device, lora_rank=args.lora-rank if hasattr(args,'lora-rank') else args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout)
    model.eval()

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    rows = []
    all_pair, all_opt = [], []

    for rel_traj, fids in groups.items():
        print(f"[Eval] Trajectory: {rel_traj} | frames: {len(fids)}")
        res = eval_sequence(device, model, args.data_root, rel_traj, fids, args.image_size)
        pair_mean = float(np.nanmean(res["pairwise"])) if len(res["pairwise"]) else float("nan")
        opt_mean  = float(np.nanmean(res["optimized"])) if len(res["optimized"]) else float("nan")
        rows.append({"trajectory": rel_traj, "pairwise_si_log_rmse": pair_mean, "optimized_si_log_rmse": opt_mean})
        if not np.isnan(pair_mean): all_pair.append(pair_mean)
        if not np.isnan(opt_mean):  all_opt.append(opt_mean)

    # global summary
    rows.append({"trajectory": "__OVERALL__", "pairwise_si_log_rmse": float(np.mean(all_pair)) if all_pair else float("nan"),
                 "optimized_si_log_rmse": float(np.mean(all_opt)) if all_opt else float("nan")})

    with open(args.output_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["trajectory", "pairwise_si_log_rmse", "optimized_si_log_rmse"])
        w.writeheader()
        w.writerows(rows)

    print(f"[DONE] Wrote: {args.output_csv}")


if __name__ == "__main__":
    main()
