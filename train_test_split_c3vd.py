import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path

class C3VDDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, load_depth=True):
        """
        Custom Dataset for processed C3VD data
        Args:
            root_dir (str): Root directory of processed C3VD data
            split (str): One of 'train', 'val', or 'test'
            transform (callable): Optional transform for image augmentation
            load_depth (bool): Whether to load depth maps
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.load_depth = load_depth

        # Load split file
        split_file = self.root_dir / 'train_test_c3vd' / f"{split}.txt"
        with open(split_file, 'r') as f:
            self.samples = [line.strip().split() for line in f]

        # Load trajectory metadata
        self.traj_metadata = {}
        for traj_dir in self.root_dir.iterdir():
            if traj_dir.is_dir() and traj_dir.name != 'train_test_c3vd':
                frames = len(list((traj_dir / 'images').glob('*.png')))
                self.traj_metadata[traj_dir.name] = {
                    'frame_count': frames,
                    'intrinsics': np.load(traj_dir / 'intrinsics' / 'intrinsics.npy')
                }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        traj_name, frame_idx = self.samples[idx]
        frame_idx = int(frame_idx)
        
        # Construct paths
        traj_dir = self.root_dir / traj_name
        image_path = traj_dir / 'images' / f"{frame_idx:06d}.png"
        pose_path = traj_dir / 'poses' / f"{frame_idx:06d}.npy"
        depth_path = traj_dir / 'depths' / f"{frame_idx:06d}.npy"

        # Load data
        image = Image.open(image_path).convert('RGB')
        pose = np.load(pose_path)
        depth = np.load(depth_path) if self.load_depth and depth_path.exists() else None

        # Get intrinsics
        intrinsics = self.traj_metadata[traj_name]['intrinsics']

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        # Convert to tensors
        image = torch.from_numpy(np.array(image).transpose(2, 0, 1)).float() / 255.0
        pose = torch.from_numpy(pose).float()
        intrinsics = torch.from_numpy(intrinsics).float()

        sample = {
            'image': image,
            'pose': pose,
            'intrinsics': intrinsics,
            'trajectory': traj_name,
            'frame_id': frame_idx
        }

        if depth is not None:
            sample['depth'] = torch.from_numpy(depth).float()

        return sample

    def get_trajectory_info(self, idx):
        """Get metadata for a trajectory"""
        traj_name, _ = self.samples[idx]
        return {
            'trajectory': traj_name,
            'frame_count': self.traj_metadata[traj_name]['frame_count'],
            'intrinsics': self.traj_metadata[traj_name]['intrinsics']
        }
