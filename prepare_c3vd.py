import os
import numpy as np
import cv2
from pathlib import Path
import shutil

class C3VDPreprocessor:
    def __init__(self, input_dir, output_dir, target_res=(512, 288)):
        """
        Initialize preprocessor with C3VD directory structure
        :param input_dir: Path to "organ_trajID_view" directory
        :param output_dir: Root directory for processed dataset
        :param target_res: (width, height) for resized images
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.target_res = target_res
        
        # Original omnidirectional camera parameters from C3VD
        self.orig_intrinsics = {
            "width": 1350,
            "height": 1080,
            "cx": 678.544839263292,
            "cy": 542.975887548343,
            "a0": 769.243600037458,
            "a2": -0.000812770624150226,
            "a3": 6.25674244578925e-07,
            "a4": -1.19662182144280e-09,
            "c": 0.999986882249990,
            "d": 0.00288273829525059,
            "e": -0.00296316513429569
        }
        
        # Calculate scaling factors for intrinsics adjustment
        self.scale_x = target_res[0] / self.orig_intrinsics["width"]
        self.scale_y = target_res[1] / self.orig_intrinsics["height"]

    def _create_directory_structure(self):
        """Create MonST3R-compatible directory structure"""
        (self.output_dir / 'images').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'depths').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'poses').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'intrinsics').mkdir(parents=True, exist_ok=True)

    def _process_images(self):
        """Process and resize RGB images with omnidirectional undistortion"""
        rgb_files = sorted(self.input_dir.glob("*.png"))
        for idx, rgb_path in enumerate(rgb_files):
            # Load and undistort image
            img = cv2.imread(str(rgb_path))
            img_undistorted = self._undistort_image(img)
            
            # Resize and save
            img_resized = cv2.resize(img_undistorted, self.target_res, 
                                   interpolation=cv2.INTER_AREA)
            np.save(self.output_dir / 'images' / f"{idx:06d}.npy", img_resized)

    def _undistort_image(self, img):
        """Undistort image using omnidirectional camera model"""
        ray_map = self._generate_ray_map(img.shape[1], img.shape[0])
        return cv2.remap(img, ray_map[..., :2].astype(np.float32), None, 
                        cv2.INTER_LINEAR)

    def _generate_ray_map(self, width, height):
        """Generate omnidirectional ray map for undistortion"""
        ix, iy = np.meshgrid(np.arange(width), np.arange(height))
        uvp_x = ix - self.orig_intrinsics["cx"]
        uvp_y = iy - self.orig_intrinsics["cy"]
        
        # Apply inverse stretch matrix
        stretch_mat = np.array([[self.orig_intrinsics["c"], 
                               self.orig_intrinsics["d"]],
                              [self.orig_intrinsics["e"], 1.0]])
        inv_stretch = np.linalg.inv(stretch_mat)
        uvpp = np.einsum('ij,...j->...i', inv_stretch, np.stack([uvp_x, uvp_y], -1))
        
        # Calculate polynomial distortion
        rho = np.sqrt(uvpp[...,0]**2 + uvpp[...,1]**2)
        z = (self.orig_intrinsics["a0"] + 
             self.orig_intrinsics["a2"]*rho**2 + 
             self.orig_intrinsics["a3"]*rho**3 + 
             self.orig_intrinsics["a4"]*rho**4)
        
        # Generate normalized ray directions
        rays = np.stack([uvpp[...,0], uvpp[...,1], z], -1)
        norms = np.linalg.norm(rays, axis=-1, keepdims=True)
        return rays / np.where(norms == 0, 1e-6, norms)

    def _process_depths(self):
        """Process and align depth maps with RGB frames"""
        depth_files = sorted(self.input_dir.glob("*_depth.tiff"))
        for idx, depth_path in enumerate(depth_files):
            depth = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)
            depth = np.where((depth == 0) | (depth == 65535), np.nan, depth)
            depth_undistorted = self._undistort_depth(depth)
            
            # Resize and convert to meters
            depth_resized = cv2.resize(depth_undistorted, self.target_res,
                                      interpolation=cv2.INTER_NEAREST)
            np.save(self.output_dir / 'depths' / f"{idx:06d}.npy", 
                   depth_resized.astype(np.float32)/1000)

    def _undistort_depth(self, depth):
        """Apply same undistortion to depth maps"""
        ray_map = self._generate_ray_map(depth.shape[1], depth.shape[0])
        return cv2.remap(depth, ray_map[..., :2].astype(np.float32), None,
                        cv2.INTER_NEAREST)

    def _process_poses(self):
        """Convert 4x4 pose matrices to MonST3R format"""
        with open(self.input_dir / "pose.txt") as f:
            poses = [np.fromstring(line, sep=',').reshape(4,4) 
                    for line in f.readlines()]
            
        # Convert to camera-to-world format and adjust scale
        for idx, pose in enumerate(poses):
            # Convert mm to meters for translation
            pose[:3,3] /= 1000  
            np.save(self.output_dir / 'poses' / f"{idx:06d}.npy", pose)

    def _save_intrinsics(self):
        """Save adjusted pinhole intrinsics for target resolution"""
        K = np.array([
            [self.orig_intrinsics["a0"] * self.scale_x, 0, 
             self.orig_intrinsics["cx"] * self.scale_x],
            [0, self.orig_intrinsics["a0"] * self.scale_y, 
             self.orig_intrinsics["cy"] * self.scale_y],
            [0, 0, 1]
        ])
        np.save(self.output_dir / 'intrinsics' / 'intrinsics.npy', K)

    def _generate_pairs(self):
        """Create training pairs with temporal stride"""
        num_frames = len(list(self.output_dir.glob("images/*.npy")))
        pairs = [[i, i+1] for i in range(num_frames-1)]
        np.savez(self.output_dir / 'pairs.npz', pairs=np.array(pairs))

    def process(self):
        """Execute full processing pipeline"""
        self._create_directory_structure()
        self._process_images()
        self._process_depths()
        self._process_poses()
        self._save_intrinsics()
        self._generate_pairs()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True,
                      help="Input directory of raw C3VD data")
    parser.add_argument("--output_dir", type=str, default="./processed_c3vd",
                      help="Output directory for processed data")

    processor = C3VDPreprocessor(
        input_dir=Path(parser.parse_args().input_dir),
        output_dir=Path(parser.parse_args().output_dir + '/' + parser.parse_args().input_dir)
    )

    processor.process()
