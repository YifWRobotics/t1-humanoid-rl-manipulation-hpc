import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
import open3d as o3d

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
DATA_PATH = REPO_ROOT / "source/isaaclab_assets/data/feasible_poses_dataset.npz"
IMG_PATH = SCRIPT_DIR / "feasible_poses_pointcloud.png"
IMG_PATH_3D = SCRIPT_DIR / "feasible_poses_pointcloud_3d.png"


def normalize(arr, min_val, max_val, img_size, margin=20):
    """Normalize array values to image coordinates."""
    range_val = max_val - min_val
    if range_val == 0:
        range_val = 1
    return ((arr - min_val) / range_val * (img_size - 2 * margin) + margin).astype(int)


def create_2d_views(left, right):
    """Create 2D projection views using PIL with grid and axis labels."""
    img_size = 500
    margin = 60  

    all_x = np.concatenate([left[:, 0], right[:, 0]])
    all_y = np.concatenate([left[:, 1], right[:, 1]])
    all_z = np.concatenate([left[:, 2], right[:, 2]])

    x_min, x_max = all_x.min(), all_x.max()
    y_min, y_max = all_y.min(), all_y.max()
    z_min, z_max = all_z.min(), all_z.max()

    total_width = img_size * 3 + 60
    total_height = img_size + 100
    img = Image.new('RGB', (total_width, total_height), 'white')
    draw = ImageDraw.Draw(img)

    step = max(1, len(left) // 10000)
    left_sub = left[::step]
    right_sub = right[::step]

    def draw_grid_and_axes(draw, offset_x, h_min, h_max, v_min, v_max, h_label, v_label, title):
        """Draw grid lines and axis labels for a subplot."""
        plot_left = offset_x + margin
        plot_right = offset_x + img_size - 20
        plot_top = margin
        plot_bottom = img_size - 20
        plot_width = plot_right - plot_left
        plot_height = plot_bottom - plot_top

        draw.rectangle([plot_left, plot_top, plot_right, plot_bottom], outline='gray')

        num_divisions = 5
        for i in range(num_divisions + 1):
            x = plot_left + i * plot_width // num_divisions
            draw.line([(x, plot_top), (x, plot_bottom)], fill='lightgray', width=1)
            val = h_min + i * (h_max - h_min) / num_divisions
            draw.text((x - 15, plot_bottom + 5), f"{val:.2f}", fill='darkgreen')

            y = plot_top + i * plot_height // num_divisions
            draw.line([(plot_left, y), (plot_right, y)], fill='lightgray', width=1)
            val = v_max - i * (v_max - v_min) / num_divisions
            draw.text((plot_left - 40, y - 5), f"{val:.2f}", fill='darkgreen')

        draw.text((offset_x + img_size // 2 - 40, 10), title, fill='black')
        draw.text((offset_x + img_size // 2 - 10, img_size - 5), h_label, fill='blue')
        draw.text((offset_x + 5, img_size // 2), v_label, fill='blue')

        return plot_left, plot_right, plot_top, plot_bottom

    offset_x = 10
    plot_left, plot_right, plot_top, plot_bottom = draw_grid_and_axes(
        draw, offset_x, x_min, x_max, y_min, y_max, "X", "Y", "Top View (XY)"
    )
    plot_width = plot_right - plot_left
    plot_height = plot_bottom - plot_top

    lx = ((left_sub[:, 0] - x_min) / (x_max - x_min) * plot_width + plot_left).astype(int)
    ly = (plot_bottom - (left_sub[:, 1] - y_min) / (y_max - y_min) * plot_height).astype(int)
    rx = ((right_sub[:, 0] - x_min) / (x_max - x_min) * plot_width + plot_left).astype(int)
    ry = (plot_bottom - (right_sub[:, 1] - y_min) / (y_max - y_min) * plot_height).astype(int)

    for x, y in zip(lx, ly):
        if plot_left <= x <= plot_right and plot_top <= y <= plot_bottom:
            draw.ellipse([x - 1, y - 1, x + 1, y + 1], fill='blue')
    for x, y in zip(rx, ry):
        if plot_left <= x <= plot_right and plot_top <= y <= plot_bottom:
            draw.ellipse([x - 1, y - 1, x + 1, y + 1], fill='red')

    offset_x = img_size + 20
    plot_left, plot_right, plot_top, plot_bottom = draw_grid_and_axes(
        draw, offset_x, x_min, x_max, z_min, z_max, "X", "Z", "Side View (XZ)"
    )
    plot_width = plot_right - plot_left
    plot_height = plot_bottom - plot_top

    lx = ((left_sub[:, 0] - x_min) / (x_max - x_min) * plot_width + plot_left).astype(int)
    lz = (plot_bottom - (left_sub[:, 2] - z_min) / (z_max - z_min) * plot_height).astype(int)
    rx = ((right_sub[:, 0] - x_min) / (x_max - x_min) * plot_width + plot_left).astype(int)
    rz = (plot_bottom - (right_sub[:, 2] - z_min) / (z_max - z_min) * plot_height).astype(int)

    for x, z in zip(lx, lz):
        if plot_left <= x <= plot_right and plot_top <= z <= plot_bottom:
            draw.ellipse([x - 1, z - 1, x + 1, z + 1], fill='blue')
    for x, z in zip(rx, rz):
        if plot_left <= x <= plot_right and plot_top <= z <= plot_bottom:
            draw.ellipse([x-1, z-1, x+1, z+1], fill='red')

    offset_x = img_size * 2 + 30
    plot_left, plot_right, plot_top, plot_bottom = draw_grid_and_axes(
        draw, offset_x, y_min, y_max, z_min, z_max, "Y", "Z", "Front View (YZ)"
    )
    plot_width = plot_right - plot_left
    plot_height = plot_bottom - plot_top

    ly = ((left_sub[:, 1] - y_min) / (y_max - y_min) * plot_width + plot_left).astype(int)
    lz = (plot_bottom - (left_sub[:, 2] - z_min) / (z_max - z_min) * plot_height).astype(int)
    ry = ((right_sub[:, 1] - y_min) / (y_max - y_min) * plot_width + plot_left).astype(int)
    rz = (plot_bottom - (right_sub[:, 2] - z_min) / (z_max - z_min) * plot_height).astype(int)

    for y, z in zip(ly, lz):
        if plot_left <= y <= plot_right and plot_top <= z <= plot_bottom:
            draw.ellipse([y-1, z-1, y+1, z+1], fill='blue')
    for y, z in zip(ry, rz):
        if plot_left <= y <= plot_right and plot_top <= z <= plot_bottom:
            draw.ellipse([y-1, z-1, y+1, z+1], fill='red')

    draw.text((total_width // 2 - 100, total_height - 20), "Blue = Left Hand, Red = Right Hand", fill='black')

    img.save(IMG_PATH)
    print(f"Saved 2D pointcloud image to {IMG_PATH}")


def create_3d_view(left, right):
    """Create interactive 3D view using Open3D with grid planes showing ranges."""
    step = max(1, len(left) // 20000)
    left_sub = left[::step]
    right_sub = right[::step]

    all_points = np.concatenate([left, right], axis=0)
    x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
    y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
    z_min, z_max = all_points[:, 2].min(), all_points[:, 2].max()

    pcd_left = o3d.geometry.PointCloud()
    pcd_left.points = o3d.utility.Vector3dVector(left_sub)
    pcd_left.paint_uniform_color([0, 0, 1])  # Blue

    pcd_right = o3d.geometry.PointCloud()
    pcd_right.points = o3d.utility.Vector3dVector(right_sub)
    pcd_right.paint_uniform_color([1, 0, 0])  # Red

    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])

    # Create bounding box wireframe
    bbox_points = [
        [x_min, y_min, z_min],
        [x_max, y_min, z_min],
        [x_max, y_max, z_min],
        [x_min, y_max, z_min],
        [x_min, y_min, z_max],
        [x_max, y_min, z_max],
        [x_max, y_max, z_max],
        [x_min, y_max, z_max],
    ]
    bbox_lines = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
        [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
        [0, 4], [1, 5], [2, 6], [3, 7],  # Vertical edges
    ]
    bbox_colors = [[0.3, 0.3, 0.3] for _ in range(len(bbox_lines))]  # Dark gray

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(bbox_points)
    line_set.lines = o3d.utility.Vector2iVector(bbox_lines)
    line_set.colors = o3d.utility.Vector3dVector(bbox_colors)

    grid_lines_points = []
    grid_lines_indices = []
    num_grid = 5

    for i in range(num_grid + 1):
        x = x_min + i * (x_max - x_min) / num_grid
        idx = len(grid_lines_points)
        grid_lines_points.append([x, y_min, z_min])
        grid_lines_points.append([x, y_max, z_min])
        grid_lines_indices.append([idx, idx + 1])

        y = y_min + i * (y_max - y_min) / num_grid
        idx = len(grid_lines_points)
        grid_lines_points.append([x_min, y, z_min])
        grid_lines_points.append([x_max, y, z_min])
        grid_lines_indices.append([idx, idx + 1])

    for i in range(num_grid + 1):
        x = x_min + i * (x_max - x_min) / num_grid
        idx = len(grid_lines_points)
        grid_lines_points.append([x, y_min, z_min])
        grid_lines_points.append([x, y_min, z_max])
        grid_lines_indices.append([idx, idx + 1])

        z = z_min + i * (z_max - z_min) / num_grid
        idx = len(grid_lines_points)
        grid_lines_points.append([x_min, y_min, z])
        grid_lines_points.append([x_max, y_min, z])
        grid_lines_indices.append([idx, idx + 1])

    for i in range(num_grid + 1):
        y = y_min + i * (y_max - y_min) / num_grid
        idx = len(grid_lines_points)
        grid_lines_points.append([x_min, y, z_min])
        grid_lines_points.append([x_min, y, z_max])
        grid_lines_indices.append([idx, idx + 1])

        z = z_min + i * (z_max - z_min) / num_grid
        idx = len(grid_lines_points)
        grid_lines_points.append([x_min, y_min, z])
        grid_lines_points.append([x_min, y_max, z])
        grid_lines_indices.append([idx, idx + 1])

    grid_colors = [[0.7, 0.7, 0.7] for _ in range(len(grid_lines_indices))]  # Light gray

    grid_line_set = o3d.geometry.LineSet()
    grid_line_set.points = o3d.utility.Vector3dVector(grid_lines_points)
    grid_line_set.lines = o3d.utility.Vector2iVector(grid_lines_indices)
    grid_line_set.colors = o3d.utility.Vector3dVector(grid_colors)

    print("Opening 3D viewer...")
    print("  - Blue = Left Hand")
    print("  - Red = Right Hand")
    print("  - Gray grid = Coordinate planes")
    print(f"  - X range: [{x_min:.3f}, {x_max:.3f}]")
    print(f"  - Y range: [{y_min:.3f}, {y_max:.3f}]")
    print(f"  - Z range: [{z_min:.3f}, {z_max:.3f}]")
    print("  - Press Q or close window to exit")
    print("  - Use mouse to rotate/zoom/pan")

    o3d.visualization.draw_geometries(
        [pcd_left, pcd_right, coord_frame, line_set, grid_line_set],
        window_name=f"Hand Positions | X:[{x_min:.2f},{x_max:.2f}] Y:[{y_min:.2f},{y_max:.2f}] Z:[{z_min:.2f},{z_max:.2f}]",
        width=1200,
        height=800,
    )


def main():
    data = np.load(DATA_PATH)
    left = data["xyzwxyz_BLH"][:, :3]
    right = data["xyzwxyz_BRH"][:, :3]

    print(f"Loaded {len(left)} samples")
    print(f"Left hand X range: [{left[:, 0].min():.3f}, {left[:, 0].max():.3f}]")
    print(f"Left hand Y range: [{left[:, 1].min():.3f}, {left[:, 1].max():.3f}]")
    print(f"Left hand Z range: [{left[:, 2].min():.3f}, {left[:, 2].max():.3f}]")
    print(f"Right hand X range: [{right[:, 0].min():.3f}, {right[:, 0].max():.3f}]")
    print(f"Right hand Y range: [{right[:, 1].min():.3f}, {right[:, 1].max():.3f}]")
    print(f"Right hand Z range: [{right[:, 2].min():.3f}, {right[:, 2].max():.3f}]")

    create_2d_views(left, right)
    create_3d_view(left, right)


if __name__ == "__main__":
    main()
