# train.py
# Fine-tune MonST3R (decoder + heads only) with LoRA on C3VD / EndoSLAM using native pairwise losses.

import os
import argparse
import matplotlib.pyplot as plt
import torch
import torch.optim as optim

from lora_monst3r import (
    build_dataset,
    build_lora_monst3r,
    make_loader,
    train_one_epoch,
    validate,
    TrainConfig,
    save_checkpoint
)


def parse_args():
    ap = argparse.ArgumentParser("LoRA finetuning for MonST3R (decoder+heads) with native pairwise losses")
    # Data
    ap.add_argument("--dataset", type=str, choices=["c3vd", "endoslam"], required=True)
    ap.add_argument("--data-root", type=str, required=True)
    ap.add_argument("--split-train", type=str, default="train")
    ap.add_argument("--split-val", type=str, default="val")
    ap.add_argument("--image-size", type=int, default=384)

    # Model
    ap.add_argument("--arch", type=str, default="base", help="MonST3R/DUSt3R arch name")
    ap.add_argument("--ckpt", type=str, default=None, help="pretrained checkpoint (.pt/.pth)")

    # LoRA
    ap.add_argument("--lora-rank", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--lora-dropout", type=float, default=0.0)
    ap.add_argument("--train-norms", action="store_true")

    # Loss params
    ap.add_argument("--alpha-conf", type=float, default=1.0)
    ap.add_argument("--norm-mode", type=str, default="avg_dis", choices=["avg_dis","med_dis","none"])
    ap.add_argument("--gt-scale", action="store_true")

    # Train
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--log-every", type=int, default=25)

    # Misc
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output-dir", type=str, default="outputs/run")
    return ap.parse_args()


def plot_curves(history, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    plt.figure()
    plt.plot(history["train_loss"], label="train")
    plt.plot(history["val_loss"], label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    path = os.path.join(out_dir, "loss_curve.png")
    plt.savefig(path, bbox_inches="tight")
    print(f"[PLOT] saved {path}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Data
    train_ds = build_dataset(args.dataset, args.data_root, args.split_train, args.image_size)
    val_ds   = build_dataset(args.dataset, args.data_root, args.split_val,   args.image_size)
    train_loader = make_loader(train_ds, args.batch_size, args.num_workers, shuffle=True)
    val_loader   = make_loader(val_ds,   args.batch_size, args.num_workers, shuffle=False)

    # Model (ONLY decoder + heads have LoRA + trainable params)
    model = build_lora_monst3r(
        arch=args.arch,
        checkpoint=args.ckpt,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        train_norms=args.train_norms,
        device=args.device,
        alpha_conf=args.alpha_conf,
        norm_mode=args.norm_mode,
        gt_scale=args.gt_scale
    )

    # Optimizer
    optim_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(optim_params, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.device.startswith("cuda"))

    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        grad_clip=args.grad_clip,
        log_every=args.log_every,
        device=args.device
    )

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        print(f"\n==== Epoch {epoch}/{args.epochs} ====")
        tr = train_one_epoch(model, train_loader, optimizer, scaler, cfg)
        va = validate(model, val_loader, cfg)

        print(f"[Epoch {epoch}] train loss: {tr['loss']:.4f} | val loss: {va['loss']:.4f}")
        history["train_loss"].append(tr["loss"])
        history["val_loss"].append(va["loss"])

        save_checkpoint(os.path.join(args.output_dir, "last.pt"), model, optimizer, epoch, extra={"history": history})
        if va["loss"] < best_val:
            best_val = va["loss"]
            save_checkpoint(os.path.join(args.output_dir, "best.pt"), model, optimizer, epoch, extra={"history": history, "best_val": best_val})

    plot_curves(history, args.output_dir)
    print("[DONE] training complete.")


if __name__ == "__main__":
    main()
