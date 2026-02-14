"""
Neural Organ Composite + Logic Shell - PyVista GPU Edition
Full 3D neon organism with stability tubes, logic shell, fiber halo, and pulsating finale.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.linalg import eig
from tqdm import tqdm
import imageio
import warnings
warnings.filterwarnings("ignore")

import pyvista as pv
pv.start_xvfb()   # critical for Colab

# ============================================================
# Configuration
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
A_MATRIX_DIR = SCRIPT_DIR / "models" / "a_matrix"
OUTPUT_DIR = SCRIPT_DIR / "visualizations"
OUTPUT_DIR.mkdir(exist_ok=True)

LOGIC_SHELL_MODE = 1  # 1=thin, 2=double, 3=volumetric
NEON_COLORMAP = "plasma"  # 'plasma', 'viridis', 'inferno', 'magma'

# Render / video settings
RESOLUTION = (1920, 1080)
FPS = 24
SLOW_FACTOR = 3
INTERPOLATE = True

# Geometry complexity
MAX_EIGENVALUES = None  # None = all
FIBERS_PER_POINT = 12
TENSOR_ARROW_COUNT = 40
LOGIC_SHELL_RESOLUTION = 50
LOGIC_SHELL_RADIUS = 3.0
THRESHOLD = 0.01

PULSE_FPS = 24
PULSE_DURATION = 5

print("=" * 60)
print("Neural Organ Composite Visualization - PyVista GPU Edition")
print("=" * 60)

# ============================================================
# Section 1: Load A-matrices
# ============================================================

print("\n[1/18] Loading A-matrices...")


def load_a_matrices(directory: Path):
    matrices = []
    epoch_files = sorted(directory.glob("Epoch*_A_Matrix.csv"))
    if not epoch_files:
        print(f"✗ No A-matrix files found in {directory}")
        return None

    for csv_file in epoch_files:
        try:
            df = pd.read_csv(csv_file, header=None)
            matrix = df.values
            matrices.append(matrix)
            print(f"  Loaded {csv_file.name}: shape {matrix.shape}")
        except Exception as e:
            print(f"  ✗ Failed to load {csv_file.name}: {e}")
    return matrices


A_matrices = load_a_matrices(A_MATRIX_DIR)
if not A_matrices:
    print("✗ No valid A-matrices loaded. Exiting.")
    raise SystemExit(1)

print(f"✓ Loaded {len(A_matrices)} A-matrices")

# ============================================================
# Section 2: Stability Analysis
# ============================================================

print("\n[2/18] Computing stability metrics...")


def analyze_matrix(A: np.ndarray):
    eigenvalues, eigenvectors = eig(A)
    diag = np.diag(A)
    row_sums = np.sum(np.abs(A), axis=1)
    radii = row_sums - np.abs(diag)
    spectral_radius = np.max(np.abs(eigenvalues))
    gershgorin_bound = np.max(diag + radii)
    return {
        "matrix": A,
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors.T,  # row-wise access
        "diagonal": diag,
        "radii": radii,
        "spectral_radius": spectral_radius,
        "gershgorin_bound": gershgorin_bound,
    }


results = [analyze_matrix(A) for A in A_matrices]

rho_all = np.array([r["spectral_radius"] for r in results])
gersh_all = np.array([r["gershgorin_bound"] for r in results])

min_rho, max_rho = float(np.min(rho_all)), float(np.max(rho_all))
min_gersh, max_gersh = float(np.min(gersh_all)), float(np.max(gersh_all))

print(f"  Spectral radius range: [{min_rho:.6f}, {max_rho:.6f}]")
print(f"  Gershgorin range: [{min_gersh:.6f}, {max_gersh:.6f}]")

# ============================================================
# Section 3: Color Mapping Helpers
# ============================================================

print("\n[3/18] Defining neon color maps...")

import matplotlib.pyplot as plt


def neon_color(value, vmin, vmax, cmap_name=NEON_COLORMAP):
    norm = (value - vmin) / (vmax - vmin + 1e-10)
    norm = np.clip(norm, 0.0, 1.0)
    cmap = plt.cm.get_cmap(cmap_name)
    rgba = cmap(norm)
    return rgba[:3]  # drop alpha


def neon_eigen(v):
    return neon_color(v, min_rho, max_rho)


def neon_shell(v):
    return neon_color(v, min_gersh, max_gersh)


def neon_fiber(v):
    return neon_color(v, 0.0, 1.0)


def neon_tensor(v):
    return neon_color(v, 0.0, 1.0)


def neon_tube(v):
    return neon_color(v, min_rho, max_rho)


# ============================================================
# Section 4: Composite Geometry
# ============================================================

print("\n[4/18] Computing composite geometry...")


def compute_composite(result, epoch: int):
    A = result["matrix"]
    eigenvalues = result["eigenvalues"]
    eigenvectors = result["eigenvectors"]
    radii = result["radii"]
    rho = result["spectral_radius"]
    gersh = result["gershgorin_bound"]

    if MAX_EIGENVALUES is not None:
        n = min(len(eigenvalues), MAX_EIGENVALUES)
        eigenvalues = eigenvalues[:n]
        eigenvectors = eigenvectors[:n]
        radii = radii[:n]
    else:
        n = len(eigenvalues)

    re_vals = np.real(eigenvalues)
    im_vals = np.imag(eigenvalues)
    pts = np.column_stack([re_vals, im_vals, radii])

    center_shift = np.mean(pts, axis=0)
    pts = pts - center_shift

    spheres = [(pts[i], radii[i]) for i in range(n)]

    fibers = []
    for i in range(n):
        p = pts[i]
        vec = eigenvectors[i]
        vec3 = vec[: min(3, len(vec))]
        vec3 = np.pad(vec3, (0, max(0, 3 - len(vec3))), constant_values=0)
        vec3 = np.real(vec3)
        if np.linalg.norm(vec3) == 0:
            direction = np.array([0.0, 0.0, 1.0])
        else:
            direction = vec3 / np.linalg.norm(vec3)

        for t in np.linspace(0, 2 * np.pi, FIBERS_PER_POINT, endpoint=False):
            perturb = direction + 0.2 * np.array([np.cos(t), np.sin(t), 0.0])
            perturb = perturb / np.linalg.norm(perturb)
            end_point = p + 0.15 * perturb
            fibers.append((p, end_point))

    tensor_arrows = []
    tensor_rows = A[:TENSOR_ARROW_COUNT, : min(3, A.shape[1])]
    for i, row in enumerate(tensor_rows):
        base = np.array([float(i), -1.5, 0.0])
        row_padded = np.pad(row, (0, max(0, 3 - len(row))), constant_values=0)
        if np.linalg.norm(row_padded) == 0:
            direction = np.array([0.0, 0.0, 1.0])
        else:
            direction = row_padded / np.linalg.norm(row_padded)
        end = base + 0.3 * direction
        tensor_arrows.append((base, end))

    return {
        "points": pts,
        "spheres": spheres,
        "fibers": fibers,
        "tensor_arrows": tensor_arrows,
        "rho": rho,
        "gersh": gersh,
        "epoch": epoch,
    }


composites = [compute_composite(results[i], i + 1) for i in range(len(results))]

# ============================================================
# Section 5: Stability Tubes
# ============================================================

print("\n[5/18] Building stability tubes...")


def stability_tubes(epoch, rho, gersh):
    e = float(epoch)

    rho_start = np.array([e, 2.0, 0.0])
    rho_height = 0.2 * (rho - min_rho) / (max_rho - min_rho + 1e-10)
    rho_end = np.array([e + 0.1, 2.0 + rho_height, 0.0])

    gersh_start = np.array([e, 2.0, -0.5])
    gersh_height = 0.2 * (gersh - min_gersh) / (max_gersh - min_gersh + 1e-10)
    gersh_end = np.array([e + 0.1, 2.0 + gersh_height, -0.5])

    return {"rho": (rho_start, rho_end), "gersh": (gersh_start, gersh_end)}


tubes_per_epoch = [
    stability_tubes(c["epoch"], c["rho"], c["gersh"]) for c in composites
]

# ============================================================
# Section 6: Logic Shell Geometry
# ============================================================

print("\n[6/18] Preparing logic-shell geometry...")

logic_maps = [
    np.sign(np.clip(r["matrix"], -THRESHOLD, THRESHOLD)) for r in results
]


def logic_shell_points(logic_matrix, mode):
    n = logic_matrix.shape[0]
    resolution = LOGIC_SHELL_RESOLUTION

    coords = []
    vals = []

    for i in range(resolution):
        theta = i / resolution * np.pi
        for j in range(resolution):
            phi = j / resolution * 2 * np.pi
            x = LOGIC_SHELL_RADIUS * np.sin(theta) * np.cos(phi)
            y = LOGIC_SHELL_RADIUS * np.sin(theta) * np.sin(phi)
            z = LOGIC_SHELL_RADIUS * np.cos(theta)
            coords.append([x, y, z])

            row = int((i / resolution) * n)
            col = int((j / resolution) * n)
            row = min(row, n - 1)
            col = min(col, n - 1)
            vals.append(logic_matrix[row, col])

    return np.array(coords), np.array(vals)


# ============================================================
# Section 7: PyVista Scene Construction
# ============================================================

print("\n[7/18] Building 3D frames with PyVista...")

pv.global_theme.background = (0.02, 0.02, 0.05)
pv.global_theme.window_size = RESOLUTION


def build_scene_for_epoch(idx: int, pulse_scale: float = 1.0, pulse_angle: float = None):
    c = composites[idx]
    tubes = tubes_per_epoch[idx]
    logic = logic_maps[idx]

    pts = c["points"] * pulse_scale
    spheres = c["spheres"]
    fibers = c["fibers"]
    tensor_arrows = c["tensor_arrows"]
    rho = c["rho"]

    coords, vals = logic_shell_points(logic, LOGIC_SHELL_MODE)
    coords = coords * pulse_scale

    plotter = pv.Plotter(off_screen=True, window_size=RESOLUTION)
    plotter.set_background(0.02, 0.02, 0.05)

    # Logic shell
    if LOGIC_SHELL_MODE in (1, 2, 3):
        shell_points = pv.PolyData(coords)
        colors = []
        for v in vals:
            if v == 1:
                colors.append(neon_shell(1.0))
            elif v == -1:
                colors.append(neon_shell(0.0))
            else:
                colors.append((0.0, 0.0, 0.0))
        colors = np.array(colors)
        mask = np.any(colors > 0, axis=1)
        shell_points = shell_points.extract_points(mask)
        shell_points["colors"] = colors[mask]
        plotter.add_mesh(
            shell_points,
            scalars="colors",
            rgb=True,
            point_size=5.0,
            opacity=0.12 if LOGIC_SHELL_MODE == 1 else 0.08,
            render_points_as_spheres=True,
        )

    # Gershgorin spheres
    for pos, radius in spheres:
        sphere = pv.Sphere(radius=radius, center=pos, theta_resolution=24, phi_resolution=16)
        color = neon_shell(radius)
        plotter.add_mesh(
            sphere,
            color=color,
            opacity=0.10,
            smooth_shading=True,
        )

    # Eigenvalue points
    eigen_cloud = pv.PolyData(pts)
    eigen_color = neon_eigen(rho)
    plotter.add_mesh(
        eigen_cloud,
        color=eigen_color,
        point_size=12.0,
        opacity=0.30,
        render_points_as_spheres=True,
    )

    # Fiber halo
    for start, end in fibers:
        line = pv.Line(start, end)
        plotter.add_mesh(
            line,
            color=neon_fiber(0.7),
            line_width=2.0,
            opacity=0.45,
        )

    # Tensor arrows
    for start, end in tensor_arrows:
        line = pv.Line(start, end)
        plotter.add_mesh(
            line,
            color=neon_tensor(0.3),
            line_width=3.0,
            opacity=0.20,
        )

    # Stability tubes
    rho_tube = tubes["rho"]
    gersh_tube = tubes["gersh"]

    rho_line = pv.Line(rho_tube[0], rho_tube[1])
    gersh_line = pv.Line(gersh_tube[0], gersh_tube[1])

    plotter.add_mesh(
        rho_line,
        color=neon_tube(rho),
        line_width=10.0,
        opacity=0.20,
    )
    plotter.add_mesh(
        gersh_line,
        color=neon_tube(rho),
        line_width=10.0,
        opacity=0.20,
    )

    # Camera
    plotter.set_focus((0.0, 0.0, 0.0))
    plotter.set_viewup((0.0, 0.0, 1.0))
    if pulse_angle is None:
        az = 45.0 + idx * 10.0
    else:
        az = pulse_angle
    plotter.camera.azimuth = az
    plotter.camera.elevation = 20.0
    plotter.camera.zoom(1.2)

    return plotter


# ============================================================
# Section 8: Generate Frames
# ============================================================

print("  Generating frames...")
frames = []
for i in tqdm(range(len(composites)), desc="Creating frames"):
    pl = build_scene_for_epoch(i)
    img = pl.screenshot(return_img=True)
    pl.close()
    frames.append(img)

# ============================================================
# Section 9–11: Interpolation + Slowdown
# ============================================================

print("\n[8/18] Reversing frames...")
rev_frames = frames[::-1]

if INTERPOLATE:
    print("\n[9/18] Interpolating between frames...")
    smooth_frames = []
    for i in range(len(rev_frames) - 1):
        smooth_frames.append(rev_frames[i])
        blend = (
            0.5 * rev_frames[i].astype(float)
            + 0.5 * rev_frames[i + 1].astype(float)
        ).astype(np.uint8)
        smooth_frames.append(blend)
    smooth_frames.append(rev_frames[-1])
else:
    print("\n[9/18] Skipping interpolation...")
    smooth_frames = rev_frames

print("\n[10/18] Slowing down animation...")
slow_frames = []
for frame in smooth_frames:
    slow_frames.extend([frame] * SLOW_FACTOR)

print(f"  Total frames: {len(slow_frames)}")

# ============================================================
# Section 12–14: Export Main Video
# ============================================================

print("\n[12/18] Exporting main MP4...")

main_video_path = OUTPUT_DIR / f"neural_organ_logic_shell_mode{LOGIC_SHELL_MODE}_pyvista.mp4"

with imageio.get_writer(
    main_video_path, fps=FPS, codec="libx264", quality=8, pixelformat="yuv420p"
) as writer:
    for frame in tqdm(slow_frames, desc="Writing main video"):
        writer.append_data(frame)

print(f"✓ Main video saved: {main_video_path}")

# ============================================================
# Section 17–18: Pulsating Final Shot
# ============================================================

print("\n[17/18] Building 5-second pulsating final shot...")

pulse_total_frames = PULSE_FPS * PULSE_DURATION
pulsating_frames = []

for t in tqdm(range(pulse_total_frames), desc="Pulsating frames"):
    scale = 1.0 + 0.02 * np.sin(2 * np.pi * t / pulse_total_frames)
    angle = 45.0 + t * 2.0
    pl = build_scene_for_epoch(len(composites) - 1, pulse_scale=scale, pulse_angle=angle)
    img = pl.screenshot(return_img=True)
    pl.close()
    pulsating_frames.append(img)

print("\n[18/18] Exporting pulsating final MP4...")

pulse_video_path = OUTPUT_DIR / "neural_organ_final_pulse_pyvista.mp4"

with imageio.get_writer(
    pulse_video_path, fps=PULSE_FPS, codec="libx264", quality=8, pixelformat="yuv420p"
) as writer:
    for frame in tqdm(pulsating_frames, desc="Writing pulse video"):
        writer.append_data(frame)

print(f"✓ Pulse video saved: {pulse_video_path}")

# ============================================================
# Complete
# ============================================================

print("\n" + "=" * 60)
print("✓ PyVista Script complete!")
print("=" * 60)
print(f"Main video:  {main_video_path}")
print(f"Pulse video: {pulse_video_path}")
print(f"Total frames generated: {len(slow_frames) + len(pulsating_frames)}")
print("=" * 60)
