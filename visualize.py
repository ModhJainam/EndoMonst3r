import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import filedialog, ttk
from pathlib import Path

class DepthViewer:
        def __init__(self, root):
            self.root = root
            self.root.title("EndoSLAM Visualization Tool")
            self.root.geometry("1200x800")
            
            # State variables
            self.base_dir = None
            self.depth_dir = None
            self.image_dir = None
            self.files = []
            self.current_index = 0
            self.colormap = "viridis"
            self.depth_range = (None, None)
            self.show_images = True
            self.show_depths = True
            self.view_mode = "side-by-side"  # Options: "side-by-side", "overlay", "image-only", "depth-only"
            
            # Create UI elements
            self.create_ui()
            
        def create_ui(self):
            # Main frame
            main_frame = ttk.Frame(self.root)
            main_frame.pack(fill=tk.BOTH, expand=True)
            
            # Top control panel
            control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
            control_frame.pack(fill=tk.X, padx=10, pady=5)
            
            # Directory selection
            dir_frame = ttk.Frame(control_frame)
            dir_frame.pack(fill=tk.X, pady=5)
            
            ttk.Button(dir_frame, text="Select Trajectory Directory", 
                      command=self.select_directory).pack(side=tk.LEFT, padx=5)
            
            self.dir_label = ttk.Label(dir_frame, text="No directory selected")
            self.dir_label.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
            
            # Visualization options
            viz_frame = ttk.Frame(control_frame)
            viz_frame.pack(fill=tk.X, pady=5)
            
            # View mode selection
            ttk.Label(viz_frame, text="View Mode:").pack(side=tk.LEFT, padx=5)
            self.view_mode_var = tk.StringVar(value=self.view_mode)
            view_modes = ["side-by-side", "overlay", "image-only", "depth-only"]
            view_mode_menu = ttk.Combobox(viz_frame, textvariable=self.view_mode_var, 
                                        values=view_modes, width=12)
            view_mode_menu.pack(side=tk.LEFT, padx=5)
            view_mode_menu.bind("<<ComboboxSelected>>", self.update_view_mode)
            
            # Colormap selection
            ttk.Label(viz_frame, text="Colormap:").pack(side=tk.LEFT, padx=5)
            self.colormap_var = tk.StringVar(value=self.colormap)
            colormaps = ["viridis", "plasma", "inferno", "magma", "cividis", "turbo", "gray"]
            colormap_menu = ttk.Combobox(viz_frame, textvariable=self.colormap_var, 
                                        values=colormaps, width=10)
            colormap_menu.pack(side=tk.LEFT, padx=5)
            colormap_menu.bind("<<ComboboxSelected>>", self.update_colormap)
            
            # Auto range checkbox
            self.auto_range_var = tk.BooleanVar(value=True)
            ttk.Checkbutton(viz_frame, text="Auto Range", variable=self.auto_range_var,
                           command=self.toggle_auto_range).pack(side=tk.LEFT, padx=5)
            
            # Min/Max depth inputs
            ttk.Label(viz_frame, text="Min:").pack(side=tk.LEFT, padx=2)
            self.min_depth_var = tk.StringVar(value="0.0")
            self.min_entry = ttk.Entry(viz_frame, textvariable=self.min_depth_var, width=6, state="disabled")
            self.min_entry.pack(side=tk.LEFT, padx=2)
            
            ttk.Label(viz_frame, text="Max:").pack(side=tk.LEFT, padx=2)
            self.max_depth_var = tk.StringVar(value="1.0")
            self.max_entry = ttk.Entry(viz_frame, textvariable=self.max_depth_var, width=6, state="disabled")
            self.max_entry.pack(side=tk.LEFT, padx=2)
            
            self.apply_button = ttk.Button(viz_frame, text="Apply Range", command=self.apply_depth_range,
                                         state="disabled")
            self.apply_button.pack(side=tk.LEFT, padx=5)
            
            # Overlay transparency (for overlay mode)
            ttk.Label(viz_frame, text="Opacity:").pack(side=tk.LEFT, padx=5)
            self.opacity_var = tk.DoubleVar(value=0.5)
            self.opacity_slider = ttk.Scale(viz_frame, from_=0.0, to=1.0, orient=tk.HORIZONTAL,
                                          variable=self.opacity_var, length=100)
            self.opacity_slider.pack(side=tk.LEFT, padx=5)
            self.opacity_slider.bind("<ButtonRelease-1>", self.update_display)
            
            # Frame info
            self.info_label = ttk.Label(viz_frame, text="No frames loaded")
            self.info_label.pack(side=tk.RIGHT, padx=10)
            
            # Create display frame
            display_frame = ttk.Frame(main_frame)
            display_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
            
            # Create figure for visualization
            self.fig = plt.figure(figsize=(12, 6))
            self.canvas = FigureCanvasTkAgg(self.fig, master=display_frame)
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Create slider frame
            slider_frame = ttk.Frame(main_frame)
            slider_frame.pack(fill=tk.X, padx=10, pady=5)
            
            # Previous button
            ttk.Button(slider_frame, text="◀", command=self.prev_frame).pack(side=tk.LEFT, padx=5)
            
            # Slider
            self.slider = ttk.Scale(slider_frame, from_=0, to=0, orient=tk.HORIZONTAL,
                                  command=self.on_slider_change)
            self.slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
            
            # Next button
            ttk.Button(slider_frame, text="▶", command=self.next_frame).pack(side=tk.LEFT, padx=5)
            
            # Status bar
            self.status_var = tk.StringVar(value="Ready")
            status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
            status_bar.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=2)
            
        def select_directory(self):
            """Open directory selection dialog and load files"""
            dir_path = filedialog.askdirectory(title="Select Trajectory Directory")
            if not dir_path:
                return
                
            self.base_dir = Path(dir_path)
            self.dir_label.config(text=str(self.base_dir))
            
            # Look for images and depths directories
            self.image_dir = self.base_dir / "images"
            self.depth_dir = self.base_dir / "depths"
            
            if not self.image_dir.exists() and not self.depth_dir.exists():
                self.status_var.set("Error: Neither images nor depths directory found")
                return
                
            self.status_var.set("Loading files...")
            self.root.update()
            
            # Find all files
            image_files = []
            if self.image_dir.exists():
                image_files = sorted([f for f in self.image_dir.glob("*.*") 
                                    if f.suffix.lower() in ('.png', '.jpg', '.jpeg')])
                
            depth_files = []
            if self.depth_dir.exists():
                depth_files = sorted([f for f in self.depth_dir.glob("*.npy")])
                
            # Match files by index
            self.files = []
            for i in range(max(len(image_files), len(depth_files))):
                item = {}
                if i < len(image_files):
                    item['image'] = image_files[i]
                if i < len(depth_files):
                    item['depth'] = depth_files[i]
                if item:
                    self.files.append(item)
            
            if not self.files:
                self.status_var.set("No valid files found")
                return
                
            # Update slider range
            self.slider.configure(from_=0, to=len(self.files)-1)
            
            # Reset to first frame
            self.current_index = 0
            self.slider.set(0)
            
            # Load and display first frame
            self.load_frame(0)
            self.status_var.set(f"Loaded {len(self.files)} frames")
            
        def load_frame(self, index):
            """Load and display a frame by index"""
            if not self.files or index < 0 or index >= len(self.files):
                return
                
            # Update info
            frame_info = f"Frame {index+1}/{len(self.files)}"
            if 'image' in self.files[index]:
                frame_info += f" | Image: {self.files[index]['image'].name}"
            if 'depth' in self.files[index]:
                frame_info += f" | Depth: {self.files[index]['depth'].name}"
            self.info_label.config(text=frame_info)
            
            # Clear previous plot
            self.fig.clear()
            
            # Load data based on view mode
            view_mode = self.view_mode_var.get()
            
            if view_mode == "image-only" and 'image' in self.files[index]:
                # Show only image
                ax = self.fig.add_subplot(111)
                img = plt.imread(self.files[index]['image'])
                ax.imshow(img)
                ax.set_title(f"RGB Image: {self.files[index]['image'].stem}")
                ax.axis('off')
                
            elif view_mode == "depth-only" and 'depth' in self.files[index]:
                # Show only depth
                ax = self.fig.add_subplot(111)
                depth_map = np.load(self.files[index]['depth'])
                
                # Get depth range
                vmin, vmax = self.depth_range
                if self.auto_range_var.get():
                    # Filter out zeros and get percentiles to avoid outliers
                    valid_depths = depth_map[depth_map > 0]
                    if len(valid_depths) > 0:
                        vmin = np.percentile(valid_depths, 1)
                        vmax = np.percentile(valid_depths, 99)
                
                im = ax.imshow(depth_map, cmap=self.colormap_var.get(), vmin=vmin, vmax=vmax)
                ax.set_title(f"Depth Map: {self.files[index]['depth'].stem}")
                ax.axis('off')
                
                # Add colorbar
                cbar = self.fig.colorbar(im, ax=ax)
                cbar.set_label('Depth (mm)')
                
            elif view_mode == "side-by-side":
                # Show image and depth side by side if available
                if 'image' in self.files[index] and 'depth' in self.files[index]:
                    # Both available
                    ax1 = self.fig.add_subplot(121)
                    ax2 = self.fig.add_subplot(122)
                    
                    # Display image
                    img = plt.imread(self.files[index]['image'])
                    ax1.imshow(img)
                    ax1.set_title(f"RGB Image: {self.files[index]['image'].stem}")
                    ax1.axis('off')
                    
                    # Display depth
                    depth_map = np.load(self.files[index]['depth'])
                    
                    # Get depth range
                    vmin, vmax = self.depth_range
                    if self.auto_range_var.get():
                        valid_depths = depth_map[depth_map > 0]
                        if len(valid_depths) > 0:
                            vmin = np.percentile(valid_depths, 1)
                            vmax = np.percentile(valid_depths, 99)
                    
                    im = ax2.imshow(depth_map, cmap=self.colormap_var.get(), vmin=vmin, vmax=vmax)
                    ax2.set_title(f"Depth Map: {self.files[index]['depth'].stem}")
                    ax2.axis('off')
                    
                    # Add colorbar
                    cbar = self.fig.colorbar(im, ax=ax2)
                    cbar.set_label('Depth (mm)')
                    
                elif 'image' in self.files[index]:
                    # Only image available
                    ax = self.fig.add_subplot(111)
                    img = plt.imread(self.files[index]['image'])
                    ax.imshow(img)
                    ax.set_title(f"RGB Image: {self.files[index]['image'].stem}")
                    ax.axis('off')
                    
                elif 'depth' in self.files[index]:
                    # Only depth available
                    ax = self.fig.add_subplot(111)
                    depth_map = np.load(self.files[index]['depth'])
                    
                    # Get depth range
                    vmin, vmax = self.depth_range
                    if self.auto_range_var.get():
                        valid_depths = depth_map[depth_map > 0]
                        if len(valid_depths) > 0:
                            vmin = np.percentile(valid_depths, 1)
                            vmax = np.percentile(valid_depths, 99)
                    
                    im = ax.imshow(depth_map, cmap=self.colormap_var.get(), vmin=vmin, vmax=vmax)
                    ax.set_title(f"Depth Map: {self.files[index]['depth'].stem}")
                    ax.axis('off')
                    
                    # Add colorbar
                    cbar = self.fig.colorbar(im, ax=ax)
                    cbar.set_label('Depth (mm)')
                    
            elif view_mode == "overlay" and 'image' in self.files[index] and 'depth' in self.files[index]:
                # Overlay depth on RGB image
                ax = self.fig.add_subplot(111)
                
                # Load image and depth
                img = plt.imread(self.files[index]['image'])
                depth_map = np.load(self.files[index]['depth'])
                
                # Display RGB image
                ax.imshow(img)
                
                # Get depth range
                vmin, vmax = self.depth_range
                if self.auto_range_var.get():
                    valid_depths = depth_map[depth_map > 0]
                    if len(valid_depths) > 0:
                        vmin = np.percentile(valid_depths, 1)
                        vmax = np.percentile(valid_depths, 99)
                
                # Create depth overlay with transparency
                opacity = self.opacity_var.get()
                depth_colored = plt.cm.get_cmap(self.colormap_var.get())(
                    (depth_map - vmin) / (vmax - vmin) if vmax > vmin else depth_map)
                depth_colored[..., 3] = opacity  # Set alpha channel
                
                # Only show depth where it's valid (non-zero)
                depth_mask = depth_map > 0
                depth_colored[~depth_mask, 3] = 0  # Make invalid depth transparent
                
                ax.imshow(depth_colored, alpha=opacity)
                ax.set_title(f"Overlay: {self.files[index]['image'].stem}")
                ax.axis('off')
                
                # Add colorbar
                sm = plt.cm.ScalarMappable(cmap=self.colormap_var.get(), 
                                          norm=plt.Normalize(vmin=vmin, vmax=vmax))
                sm.set_array([])
                cbar = self.fig.colorbar(sm, ax=ax)
                cbar.set_label('Depth (mm)')
            
            # Update canvas
            self.fig.tight_layout()
            self.canvas.draw()
            
        def on_slider_change(self, value):
            """Handle slider value change"""
            index = int(float(value))
            if index != self.current_index:
                self.current_index = index
                self.load_frame(index)
                
        def prev_frame(self):
            """Go to previous frame"""
            if self.current_index > 0:
                self.current_index -= 1
                self.slider.set(self.current_index)
                self.load_frame(self.current_index)
                
        def next_frame(self):
            """Go to next frame"""
            if self.current_index < len(self.files) - 1:
                self.current_index += 1
                self.slider.set(self.current_index)
                self.load_frame(self.current_index)
                
        def update_colormap(self, event=None):
            """Update colormap when selection changes"""
            if self.files:
                self.load_frame(self.current_index)
                
        def update_view_mode(self, event=None):
            """Update view mode when selection changes"""
            if self.files:
                self.load_frame(self.current_index)
                
        def toggle_auto_range(self):
            """Toggle between auto and manual depth range"""
            if self.auto_range_var.get():
                self.min_entry.configure(state="disabled")
                self.max_entry.configure(state="disabled")
                self.apply_button.configure(state="disabled")
                self.depth_range = (None, None)
            else:
                self.min_entry.configure(state="normal")
                self.max_entry.configure(state="normal")
                self.apply_button.configure(state="normal")
            
            if self.files:
                self.load_frame(self.current_index)
        
        def apply_depth_range(self):
            """Apply manual depth range"""
            try:
                min_depth = float(self.min_depth_var.get())
                max_depth = float(self.max_depth_var.get())
                if min_depth >= max_depth:
                    raise ValueError("Min depth must be less than max depth")
                    
                self.depth_range = (min_depth, max_depth)
                if self.files:
                    self.load_frame(self.current_index)
                    
            except ValueError as e:
                self.status_var.set(f"Error: {str(e)}")
                
        def update_display(self, event=None):
            """Update display when opacity changes"""
            if self.files:
                self.load_frame(self.current_index)

# Create and run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = DepthViewer(root)
    root.mainloop()

    if hasattr(app, 'traj_fig'):
        plt.show()
