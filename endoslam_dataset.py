import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd

class EndoSLAMDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        """
        Custom Dataset for EndoSLAM data.
        
        Args:
            root_dir (str): Root directory of the prepared EndoSLAM data.
            split (str): One of 'train', 'val', or 'test'.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        # Load the split file
        split_file = os.path.join(root_dir, f"{split}.txt")
        with open(split_file, 'r') as f:
            self.samples = [line.strip().split() for line in f]
        
        # Load trajectory metadata
        metadata_file = os.path.join(root_dir, f"{split}_trajectories.csv")
        self.metadata = pd.read_csv(metadata_file)
        
        # Load camera intrinsics (assuming same for all images in a trajectory)
        self.intrinsics = {}
        for _, row in self.metadata.iterrows():
            camera = row['camera']
            organ = row['organ']
            traj_path = row['path']
            intrinsics_path = os.path.join(root_dir, camera, organ, traj_path, 'intrinsics', 'intrinsics.npy')
            self.intrinsics[traj_path] = np.load(intrinsics_path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        traj_path, frame_id = self.samples[idx]
        frame_id = int(frame_id)
        
        # Construct full paths
        image_path = os.path.join(self.root_dir, traj_path, 'images', f"{frame_id:06d}.png")
        pose_path = os.path.join(self.root_dir, traj_path, 'poses', f"{frame_id:06d}.npy")
        depth_path = os.path.join(self.root_dir, traj_path, 'depths', f"{frame_id:06d}.npy")
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Load pose
        pose = np.load(pose_path)
        
        # Load depth (if available)
        try:
            depth = np.load(depth_path)
        except FileNotFoundError:
            depth = None
        
        # Get intrinsics
        intrinsics = self.intrinsics[traj_path]
        
        # Apply transforms if any
        if self.transform:
            image = self.transform(image)
        
        # Convert to tensor
        image = torch.from_numpy(np.array(image).transpose((2, 0, 1))).float() / 255.0
        pose = torch.from_numpy(pose).float()
        intrinsics = torch.from_numpy(intrinsics).float()
        
        sample = {
            'image': image,
            'pose': pose,
            'intrinsics': intrinsics,
            'traj_path': traj_path,
            'frame_id': frame_id
        }
        
        if depth is not None:
            sample['depth'] = torch.from_numpy(depth).float()
        
        return sample

    def get_trajectory_info(self, idx):
        """
        Get additional information about the trajectory for a given sample.
        
        Args:
            idx (int): Index of the sample.
        
        Returns:
            dict: Dictionary containing trajectory information.
        """
        traj_path, _ = self.samples[idx]
        traj_info = self.metadata[self.metadata['path'] == traj_path].iloc[0]
        return {
            'camera': traj_info['camera'],
            'organ': traj_info['organ'],
            'frame_count': traj_info['frame_count']
        }
