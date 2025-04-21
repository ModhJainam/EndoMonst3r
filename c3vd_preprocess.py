import re
import numpy as np
import cv2
from pathlib import Path
from scipy.spatial.transform import Rotation as R

class C3VDProcessor:
    def __init__(self, input_dir, output_dir, target_res=(512, 288)):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.target_res = target_res  # (width, height)
        
        # Hardcoded omnidirectional -> pinhole camera intrinsics (C3VD specific)
        self.orig_params = {
            'width': 1350,
            'height': 1080,
            'fx': 769.2436,
            'fy': 769.2436,
            'cx': 678.5448,
            'cy': 542.9759
        }
        
        # Calculate scaled intrinsics for target resolution
        self.scale_x = target_res[0] / self.orig_params['width']
        self.scale_y = target_res[1] / self.orig_params['height']
        self.intrinsics = np.array([
            self.orig_params['fx'] * self.scale_x,
            self.orig_params['fy'] * self.scale_y,
            self.orig_params['cx'] * self.scale_x,
            self.orig_params['cy'] * self.scale_y
        ], dtype=np.float32)

    def _create_dirs(self):
        """Create MonST3R-compatible directory structure"""
        (self.output_dir / 'images').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'depths').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'poses').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'intrinsics').mkdir(parents=True, exist_ok=True)

    def _natural_sort_key(self, path):
        """Natural sorting for filenames with numeric prefixes"""
        return [int(c) if c.isdigit() else c.lower() 
                for c in re.split(r'(\d+)', path.stem)]

    def _process_frames(self):
        """Process RGB and depth frames with consistent ordering"""
        # Process color images
        color_files = sorted(
            self.input_dir.glob("*_color.png"),
            key=self._natural_sort_key
        )
        for idx, color_path in enumerate(color_files):
            img = cv2.imread(str(color_path))
            img_resized = cv2.resize(img, self.target_res, cv2.INTER_AREA)
            cv2.imwrite(str(self.output_dir / 'images' / f"{idx:06d}.png"), img_resized)

        # Process depth maps
        depth_files = sorted(
            self.input_dir.glob("*_depth.tiff"),
            key=self._natural_sort_key
        )
        for idx, depth_path in enumerate(depth_files):
            depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            depth = depth.astype(np.float32) / 1000  # mm to meters
            depth_resized = cv2.resize(depth, self.target_res, cv2.INTER_NEAREST)
            np.save(self.output_dir / 'depths' / f"{idx:06d}.npy", depth_resized)

    def _process_poses(self):
        """Convert camera-to-world matrices to MonST3R-compatible world-to-camera poses"""
        pose_path = self.input_dir / "pose.txt"
        poses = []
        
        with open(pose_path) as f:
            for line in f:
                elements = list(map(float, line.strip().split(',')))
                if len(elements) != 16:
                    continue

                # Reshape to 4x4 camera-to-world matrix
                cam_to_world = np.array(elements).reshape(4, 4)
                
                # Convert to world-to-camera matrix
                world_to_cam = np.linalg.inv(cam_to_world)
                
                # Extract translation (in meters) and rotation
                translation = world_to_cam[:3, 3]  # Already in meters
                rotation = R.from_matrix(world_to_cam[:3, :3]).as_quat()  # [x,y,z,w]
                
                poses.append(np.concatenate([translation, rotation]))

        # Save individual pose files
        for idx, pose in enumerate(poses):
            np.save(self.output_dir / 'poses' / f"{idx:06d}.npy", pose)


            # Save individual pose files
            for idx, pose in enumerate(poses):
                np.save(self.output_dir / 'poses' / f"{idx:06d}.npy", pose)

    def process(self):
        """Full processing pipeline"""
        self._create_dirs()
        self._process_frames()
        self._process_poses()
        
        # Save scaled intrinsics
        np.save(self.output_dir / 'intrinsics' / 'intrinsics.npy', self.intrinsics)
        
        # Generate frame pairs
        num_frames = len(list(self.output_dir.glob("images/*.png")))
        pairs = [[i, i+1] for i in range(num_frames-1)]
        np.savez(self.output_dir / 'pairs.npz', pairs=np.array(pairs))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True,
                      help="Input directory of raw C3VD data")
    parser.add_argument("--output_dir", type=str, default="./processed_c3vd",
                      help="Output directory for processed data")

    processor = C3VDProcessor(
        input_dir=Path(parser.parse_args().input_dir),
        output_dir=Path(parser.parse_args().output_dir + '/' + parser.parse_args().input_dir.split('/')[-1])
    )

    processor.process()
