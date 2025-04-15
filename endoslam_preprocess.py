import os
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import open3d as o3d

class MonST3RDataPreparer:
    def __init__(self, image_dir, pose_file, intrinsics_file, output_dir, 
                 target_res=(512, 288), depth_dir=None, stl_file=None):
        self.image_dir = Path(image_dir)
        self.pose_file = Path(pose_file)
        self.intrinsics_file = Path(intrinsics_file)
        self.output_dir = Path(output_dir)
        self.target_res = target_res
        self.depth_dir = Path(depth_dir) if depth_dir else None
        self.stl_file = Path(stl_file) if stl_file else None

        # Create directory structure
        (self.output_dir / 'images').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'poses').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'intrinsics').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'depths').mkdir(parents=True, exist_ok=True)

    def load_intrinsics(self):
        """Handle both colon-separated and raw value formats"""
        with open(self.intrinsics_file, 'r') as f:
            content = f.read().strip()
            
        if ':' in content:
            # Original key: value format
            intrinsics = {}
            for line in content.split('\n'):
                key, val = line.split(':')
                intrinsics[key.strip()] = float(val.strip())
            return np.array([
                intrinsics['fx'],
                intrinsics['fy'],
                intrinsics['cx'],
                intrinsics['cy']
            ])
        else:
            # UnityCam's raw value format (fx, fy, cx, cy)
            values = list(map(float, content.split(',')))
            return np.array(values[:4])

    def process_poses(self):
        """Handle pose data with robust format conversion"""
        if self.pose_file.suffix == '.csv':
            df = pd.read_csv(self.pose_file)
            df.columns = df.columns.str.strip().str.replace(r'\W+', '', regex=True)
            column_map = {
                'tX': 'trans_x', 'tY': 'trans_y', 'tZ': 'trans_z',
                'rX': 'quot_x', 'rY': 'quot_y', 'rZ': 'quot_z', 'rW': 'quot_w'
            }
            df = df.rename(columns=column_map)
        elif self.pose_file.suffix == '.txt':
            # For MiroCam's txt format with semicolon separators
            df = pd.read_csv(self.pose_file, sep=';', skipinitialspace=True, low_memory=False)
            
            # Clean column names and map them to expected format
            df.columns = df.columns.str.strip()
            
            # Print available columns for debugging
            print("Original columns:", df.columns.tolist())
            
            # Map columns, explicitly excluding the time column 't'
            column_map = {
                'p_x': 'trans_x', 'p_y': 'trans_y', 'p_z': 'trans_z',
                'Qx': 'quot_x', 'Qy': 'quot_y', 'Qz': 'quot_z', 'Qw': 'quot_w'
            }
            df = df.rename(columns=column_map)
            
            # Print columns after renaming
            print("After renaming:", df.columns.tolist())
            
            # Skip the first row if it contains units (s, m, Qunit)
            if 's' in str(df.iloc[0, 0]) and 'm' in str(df.iloc[0, 1]):
                df = df.iloc[1:].reset_index(drop=True)
            
            # Convert all columns to float
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Check if all required columns exist
            required_columns = ['trans_x', 'trans_y', 'trans_z', 'quot_x', 'quot_y', 'quot_z', 'quot_w']
            if all(col in df.columns for col in required_columns):
                df = df[required_columns]
            else:
                print("Warning: Not all required columns found. Available columns:", df.columns.tolist())
                # If columns aren't found, try to use the original columns directly
                if all(col in df.columns for col in ['p_x', 'p_y', 'p_z', 'Qx', 'Qy', 'Qz', 'Qw']):
                    df = df[['p_x', 'p_y', 'p_z', 'Qx', 'Qy', 'Qz', 'Qw']]
                    # Rename them now
                    df.columns = required_columns
        else:
            # For Excel, use direct column access without rename
            df = pd.read_excel(self.pose_file, usecols=[
                'trans_x', 'trans_y', 'trans_z',
                'quot_x', 'quot_y', 'quot_z', 'quot_w'
            ])
        
        # Convert to numpy array early for consistency
        poses = df.to_numpy(dtype=np.float32, copy=True)
        return poses

    def process_images(self):
        """Process and resize images from PNG/JPG sources"""
        # Get supported image extensions
        image_exts = ('*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG')
        image_paths = []
        
        # Collect images from all extensions
        for ext in image_exts:
            image_paths.extend(self.image_dir.glob(ext))
        
        # Sort by numeric filename
        image_paths = sorted(
            image_paths,
            key=lambda x: int(x.stem) if x.stem.isdigit() else 0
        )
        
        processed_count = 0
        
        for idx, img_path in enumerate(image_paths):
            try:
                # Read image with OpenCV
                img = cv2.imread(str(img_path))
                if img is None:
                    print(f"Warning: Could not read {img_path.name}")
                    continue
                    
                # Resize and save
                img_resized = cv2.resize(img, self.target_res, interpolation=cv2.INTER_AREA)
                output_path = self.output_dir / 'images' / f"{idx:06d}{img_path.suffix}"
                cv2.imwrite(str(output_path), img_resized)
                processed_count += 1
                
            except Exception as e:
                print(f"Error processing {img_path.name}: {str(e)}")
        
        return processed_count


    def save_poses(self):
        """Batch save poses from array"""
        poses = self.process_poses()
        for idx, pose in enumerate(poses):
            np.save(self.output_dir / 'poses' / f"{idx:06d}.npy", pose)


    def generate_pairs(self, num_frames):
        """Generate frame pairs for consecutive frames"""
        pairs = []
        stride = 2  # Matches MonST3R's training configuration
        
        for i in range(num_frames - stride):
            pairs.append([i, i + stride])
        
        np.savez(self.output_dir / 'pairs.npz', pairs=np.array(pairs))

    def process_depth_images(self):
        """Process existing depth maps from directory"""
        if not self.depth_dir or not self.depth_dir.exists():
            return 0

        depth_exts = ('*.png', '*.jpg', '*.jpeg', '*.npy', '*.tiff')
        depth_paths = []
        
        for ext in depth_exts:
            depth_paths.extend(self.depth_dir.glob(ext))
        
        depth_paths = sorted(
            depth_paths,
            key=lambda x: int(x.stem) if x.stem.isdigit() else 0
        )

        processed_count = 0
        
        for idx, depth_path in enumerate(depth_paths):
            try:
                # Read and normalize depth data
                if depth_path.suffix == '.npy':
                    depth = np.load(depth_path)
                else:
                    depth = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)
                    depth = depth.astype(np.float32) / 1000.0  # Convert mm to meters

                # Resize depth map
                depth_resized = cv2.resize(depth, self.target_res, 
                                         interpolation=cv2.INTER_NEAREST)
                
                # Save as float32 numpy array
                np.save(self.output_dir / 'depths' / f"{idx:06d}.npy", depth_resized)
                processed_count += 1
                
            except Exception as e:
                print(f"Error processing depth {depth_path.name}: {str(e)}")
        
        return processed_count

    def process_stl_file(self, intrinsics):
        """STL processing with Open3D compatibility fix"""
       
        mesh = o3d.io.read_triangle_mesh(str(self.stl_file))
        mesh.compute_vertex_normals()

        fx, fy, cx, cy = intrinsics
        width, height = self.target_res

        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=width, height=height, fx=fx, fy=fy, cx=cx, cy=cy)

        poses = self.process_poses()

        for idx, pose in enumerate(poses):
            trans = pose[:3]
            quat = pose[3:]
            quat /= np.linalg.norm(quat)
            
            T = np.eye(4)
            T[:3, :3] = R.from_quat(quat).as_matrix()
            T[:3, 3] = trans

            vis = o3d.visualization.Visualizer()
            vis.create_window(width=width, height=height, visible=False)
            vis.add_geometry(mesh)

            ctr = vis.get_view_control()
            params = o3d.camera.PinholeCameraParameters()
            params.intrinsic = intrinsic
            params.extrinsic = np.linalg.inv(T)
            
            ctr.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)

            vis.poll_events()
            vis.update_renderer()
            depth = vis.capture_depth_float_buffer(True)
            vis.destroy_window()

            np.save(self.output_dir / 'depths' / f"{idx:06d}.npy", np.asarray(depth))

        return True

    def process(self):
        """Main processing pipeline"""
        intrinsics = self.load_intrinsics()
        np.save(self.output_dir / 'intrinsics' / 'intrinsics.npy', intrinsics)
        
        self.save_poses()

        num_frames = self.process_images()
        
        if self.depth_dir:
            num_depth = self.process_depth_images()
            print(f"Processed {num_depth} depth frames")

        if self.stl_file and self.stl_file.exists():
            self.process_stl_file(intrinsics)

        self.generate_pairs(num_frames)
        print(f"Successfully processed {num_frames} frames")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--pose_file', type=str, required=True)
    parser.add_argument('--intrinsics', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--depth_dir', type=str, default=None)
    parser.add_argument('--stl', type=str, default=None)
    parser.add_argument('--res', type=int, nargs=2, default=[512, 288])

    
    args = parser.parse_args()
    
    processor = MonST3RDataPreparer(
        image_dir=args.image_dir,
        pose_file=args.pose_file,
        intrinsics_file=args.intrinsics,
        output_dir=args.output_dir,
        target_res=tuple(args.res),
        depth_dir=args.depth_dir,
        stl_file=args.stl
    )
    
    processor.process()
