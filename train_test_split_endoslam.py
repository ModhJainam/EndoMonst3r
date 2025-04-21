import argparse
import random
from pathlib import Path
import pandas as pd
import numpy as np

def create_splits(data_root, output_dir, ratios=(0.7, 0.15, 0.15), seed=42):
    """
    Create train/val/test splits
    """
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(seed)

    # Collect all trajectories
    trajectories = []
    for camera_dir in data_root.glob("*"):
        camera_name = camera_dir.name
        for organ_dir in camera_dir.glob("*"):
            # Handle UnityCam differently (no trajectory level)
            if camera_name == "UnityCam":
                frames_dir = organ_dir / "images"
                frames = list(frames_dir.glob("*.*"))
                if frames:
                    trajectories.append({
                        "path": organ_dir.relative_to(data_root),
                        "frame_count": len(frames),
                        "organ": organ_dir.name,
                        "camera": camera_name
                    })
            else:
                # Process normal camera trajectories
                for traj_dir in organ_dir.glob("*"):
                    if traj_dir.is_dir():
                        frames = list((traj_dir / "images").glob("*.*"))
                        if frames:
                            trajectories.append({
                                "path": traj_dir.relative_to(data_root),
                                "frame_count": len(frames),
                                "organ": organ_dir.name,
                                "camera": camera_name
                            })

    # Shuffle and split
    random.shuffle(trajectories)
    n_total = len(trajectories)
    train_end = int(ratios[0] * n_total)
    val_end = train_end + int(ratios[1] * n_total)

    splits = {
        "train": trajectories[:train_end],
        "val": trajectories[train_end:val_end],
        "test": trajectories[val_end:]
    }

    # Save splits
    for split_name, items in splits.items():
        pd.DataFrame(items).to_csv(output_dir / f"{split_name}_trajectories.csv", index=False)
        
        with open(output_dir / f"{split_name}.txt", "w") as f:
            for item in items:
                base_path = item["path"]
                for frame_id in range(item["frame_count"]):
                    f.write(f"{base_path} {frame_id:06d}\n")

    # Print summary
    print(f"Total trajectories: {n_total}")
    print(f"Train: {len(splits['train'])} trajectories")
    print(f"Validation: {len(splits['val'])} trajectories")
    print(f"Test: {len(splits['test'])} trajectories")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True,
                      help="Root directory of processed EndoSLAM data")
    parser.add_argument("--output_dir", type=str, required=True,
                      help="Output directory for split files")
    parser.add_argument("--ratios", type=float, nargs=3, default=[0.7, 0.15, 0.15],
                      help="Train/val/test ratios (must sum to 1)")
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed for reproducibility")
    
    args = parser.parse_args()
    if not np.isclose(sum(args.ratios), 1.0, atol=1e-3):
        raise ValueError("Ratios must sum to 1.0")
    
    create_splits(
        data_root=args.data_root,
        output_dir=args.output_dir,
        ratios=args.ratios,
        seed=args.seed
    )
