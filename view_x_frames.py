import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.ndimage import label, generate_binary_structure, zoom

print("Loading 100 random frames from line_masks.h5...")

num_frames = 100

with h5py.File('line_masks.h5', 'r') as f:
    total_frames = len(f['frames'])
    print(f"Total frames in file: {total_frames}")
    
    # Select 100 random frames
    np.random.seed(42)  # For reproducibility
    random_frame_indices = np.sort(np.random.choice(total_frames, num_frames, replace=False))
    print(f"Selected {num_frames} random frames")
    
    # Determine dimensions
    max_height = 0
    max_width = 0
    
    for i in random_frame_indices:
        heights = f['heights'][i]
        widths = f['widths'][i]
        
        if len(heights) > 0:
            max_height = max(max_height, np.max(heights))
            max_width = max(max_width, np.max(widths))
    
    img_height = max_height + 1
    img_width = max_width + 1
    print(f"Image dimensions: {img_height} x {img_width}")
    
    # Create colored masks with clusters
    all_colored_masks = []
    structure = generate_binary_structure(2, 2)  # 8-connectivity
    
    for i in random_frame_indices:
        heights = f['heights'][i]
        widths = f['widths'][i]
        
        mask = np.zeros((img_height, img_width), dtype=np.uint8)
        if len(heights) > 0:
            mask[heights, widths] = 1
        
        # Label connected components
        labeled_mask, num_clusters = label(mask, structure=structure)
        
        # Upscale to 640x480 using nearest neighbor (preserves labels)
        target_height, target_width = 480, 640
        scale_y = target_height / img_height
        scale_x = target_width / img_width
        upscaled_mask = zoom(labeled_mask, (scale_y, scale_x), order=0)
        upscaled_mask = upscaled_mask[:target_height, :target_width]
        
        all_colored_masks.append({
            'frame_idx': i,
            'labeled_mask': upscaled_mask,
            'num_clusters': num_clusters,
            'num_pixels': np.sum(mask)
        })
        print(f"Frame {i}: {num_clusters} clusters, {np.sum(mask)} pixels")

# Create a colormap with enough colors for the maximum number of clusters
max_clusters_in_any_frame = max(data['num_clusters'] for data in all_colored_masks)
print(f"\nMax clusters in any frame: {max_clusters_in_any_frame}")

# Create colormap with distinct colors
cmap = plt.cm.get_cmap('tab20', max_clusters_in_any_frame + 1)
colors = [cmap(i) for i in range(max_clusters_in_any_frame + 1)]
colors[0] = (0, 0, 0, 1)  # Black background
custom_cmap = ListedColormap(colors)

# Create visualization with 10 rows x 10 columns in portrait format
# Portrait: 1080x1920 pixels total
fig = plt.figure(figsize=(10.8, 19.2), dpi=100)
gs = fig.add_gridspec(10, 10, wspace=0, hspace=0, left=0, right=1, top=1, bottom=0)

for idx, data in enumerate(all_colored_masks):
    row = idx // 10
    col = idx % 10
    ax = fig.add_subplot(gs[row, col])
    
    frame_idx = data['frame_idx']
    labeled_mask = data['labeled_mask']
    num_clusters = data['num_clusters']
    num_pixels = data['num_pixels']
    
    ax.imshow(labeled_mask, cmap=custom_cmap, interpolation='none', vmin=0, vmax=max_clusters_in_any_frame, aspect='auto')
    ax.axis('off')
    ax.margins(0)

output_file = 'random_100_frames_colored_clusters.svg'
plt.savefig(output_file, format='svg', bbox_inches=None, pad_inches=0, dpi=100)
print(f"\n✓ Visualization saved to {output_file}")

plt.show()
