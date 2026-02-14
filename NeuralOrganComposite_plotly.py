"""
Neural Organ Composite + Logic Shell — Plotly WebGL Edition
===========================================================
Identical geometry pipeline to the PyVista GPU edition, but renders via
Plotly's WebGL backend for fast, interactive 3D on Google Colab (or any
browser).  No VTK/OpenGL context required.

Outputs:
  visualizations/NeuralOrgan_Plotly.html   (interactive 3D)
  visualizations/NeuralOrgan_Plotly.png    (static snapshot)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.linalg import eig
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

import plotly.graph_objects as go

# ============================================================
# Configuration
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
A_MATRIX_DIR = SCRIPT_DIR / "models" / "a_matrix"
OUTPUT_DIR = SCRIPT_DIR / "visualizations"
OUTPUT_DIR.mkdir(exist_ok=True)

LOGIC_SHELL_MODE = 1        # 1=thin, 2=double, 3=volumetric
NEON_COLORMAP = "Plasma"    # Plotly built-in: Plasma, Viridis, Inferno, Magma

# Geometry complexity
MAX_EIGENVALUES = None      # None = all
FIBERS_PER_POINT = 12
TENSOR_ARROW_COUNT = 40
LOGIC_SHELL_RESOLUTION = 50
LOGIC_SHELL_RADIUS = 3.0
THRESHOLD = 0.01

# Epoch spacing (Z-axis offset per epoch for 3D layering)
EPOCH_Z_SPACING = 8.0

print("=" * 60)
print("Neural Organ Composite Visualization — Plotly WebGL Edition")
print("=" * 60)

# ============================================================
# Section 1: Load A-matrices
# ============================================================

print("\n[1/8] Loading A-matrices...")


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

print("\n[2/8] Computing stability metrics...")


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
        "eigenvectors": eigenvectors.T,
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
# Section 3: Color Helpers (Plotly-native)
# ============================================================

print("\n[3/8] Defining neon color maps...")

import matplotlib.pyplot as plt


def _to_plotly_rgb(value, vmin, vmax, cmap_name="plasma"):
    """Return 'rgb(r,g,b)' string for Plotly."""
    norm = (value - vmin) / (vmax - vmin + 1e-10)
    norm = np.clip(norm, 0.0, 1.0)
    cmap = plt.cm.get_cmap(cmap_name)
    r, g, b, _ = cmap(float(norm))
    return f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"


def _to_plotly_rgba(value, vmin, vmax, alpha=0.5, cmap_name="plasma"):
    """Return 'rgba(r,g,b,a)' string for Plotly."""
    norm = (value - vmin) / (vmax - vmin + 1e-10)
    norm = np.clip(norm, 0.0, 1.0)
    cmap = plt.cm.get_cmap(cmap_name)
    r, g, b, _ = cmap(float(norm))
    return f"rgba({int(r*255)},{int(g*255)},{int(b*255)},{alpha})"


# ============================================================
# Section 4: Composite Geometry (identical pipeline)
# ============================================================

print("\n[4/8] Computing composite geometry...")


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
            fibers.append((p.copy(), end_point))

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
# Section 5: Logic Shell Geometry
# ============================================================

print("\n[5/8] Preparing logic-shell geometry...")

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
# Section 6: Build Plotly Traces per Epoch
# ============================================================

print("\n[6/8] Building Plotly WebGL traces...")


def build_epoch_traces(idx: int):
    """Build all Plotly traces for one epoch, Z-offset by epoch index."""
    c = composites[idx]
    logic = logic_maps[idx]
    epoch = c["epoch"]
    rho = c["rho"]

    z_off = idx * EPOCH_Z_SPACING
    traces = []

    # --- Logic Shell ---
    coords, vals = logic_shell_points(logic, LOGIC_SHELL_MODE)
    mask = vals != 0
    sc = coords[mask]
    sv = vals[mask]
    shell_colors = [
        _to_plotly_rgba(1.0, 0, 1, alpha=0.15) if v == 1
        else _to_plotly_rgba(0.0, 0, 1, alpha=0.15)
        for v in sv
    ]
    traces.append(go.Scatter3d(
        x=sc[:, 0], y=sc[:, 1], z=sc[:, 2] + z_off,
        mode="markers",
        marker=dict(size=1.8, color=shell_colors),
        name=f"Epoch {epoch} Logic Shell",
        legendgroup=f"e{epoch}",
        showlegend=False,
        hoverinfo="skip",
    ))

    # --- Gershgorin Spheres (wireframe approximation) ---
    pts = c["points"]
    for i, (pos, radius) in enumerate(c["spheres"]):
        # Draw a latitude/longitude wireframe
        u = np.linspace(0, 2 * np.pi, 16)
        v = np.linspace(0, np.pi, 8)
        # Equator and meridians as line traces
        for vi in range(len(v)):
            cx = pos[0] + radius * np.sin(v[vi]) * np.cos(u)
            cy = pos[1] + radius * np.sin(v[vi]) * np.sin(u)
            cz = np.full_like(cx, pos[2] + radius * np.cos(v[vi]) + z_off)
            color = _to_plotly_rgba(radius, min_gersh, max_gersh, alpha=0.08)
            traces.append(go.Scatter3d(
                x=cx, y=cy, z=cz,
                mode="lines",
                line=dict(width=1, color=color),
                showlegend=False, hoverinfo="skip",
                legendgroup=f"e{epoch}",
            ))
        if i >= 15:  # limit sphere wireframes for performance
            break

    # --- Eigenvalue Cloud ---
    eigen_colors = [_to_plotly_rgb(rho, min_rho, max_rho)] * len(pts)
    traces.append(go.Scatter3d(
        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2] + z_off,
        mode="markers",
        marker=dict(size=4, color=eigen_colors, opacity=0.7),
        name=f"Epoch {epoch} (ρ={rho:.4f})",
        legendgroup=f"e{epoch}",
        showlegend=True,
    ))

    # --- Fiber Halo ---
    fx, fy, fz = [], [], []
    for start, end in c["fibers"]:
        fx.extend([start[0], end[0], None])
        fy.extend([start[1], end[1], None])
        fz.extend([start[2] + z_off, end[2] + z_off, None])
    fiber_color = _to_plotly_rgba(0.7, 0, 1, alpha=0.35)
    traces.append(go.Scatter3d(
        x=fx, y=fy, z=fz,
        mode="lines",
        line=dict(width=1.5, color=fiber_color),
        name=f"Epoch {epoch} Fibers",
        legendgroup=f"e{epoch}",
        showlegend=False,
        hoverinfo="skip",
    ))

    # --- Tensor Arrows ---
    tx, ty, tz = [], [], []
    for start, end in c["tensor_arrows"]:
        tx.extend([start[0], end[0], None])
        ty.extend([start[1], end[1], None])
        tz.extend([start[2] + z_off, end[2] + z_off, None])
    tensor_color = _to_plotly_rgba(0.3, 0, 1, alpha=0.25)
    traces.append(go.Scatter3d(
        x=tx, y=ty, z=tz,
        mode="lines",
        line=dict(width=2.5, color=tensor_color),
        name=f"Epoch {epoch} Tensor",
        legendgroup=f"e{epoch}",
        showlegend=False,
        hoverinfo="skip",
    ))

    # --- Stability Tubes (vertical bars) ---
    gersh = c["gersh"]
    rho_height = 0.2 * (rho - min_rho) / (max_rho - min_rho + 1e-10)
    gersh_height = 0.2 * (gersh - min_gersh) / (max_gersh - min_gersh + 1e-10)

    bar_x_base = float(epoch)
    rho_color = _to_plotly_rgb(rho, min_rho, max_rho)
    traces.append(go.Scatter3d(
        x=[bar_x_base, bar_x_base + 0.1],
        y=[2.0, 2.0 + rho_height],
        z=[z_off, z_off],
        mode="lines",
        line=dict(width=8, color=rho_color),
        name=f"Epoch {epoch} ρ-tube",
        legendgroup=f"e{epoch}",
        showlegend=False,
    ))
    traces.append(go.Scatter3d(
        x=[bar_x_base, bar_x_base + 0.1],
        y=[2.0, 2.0 + gersh_height],
        z=[-0.5 + z_off, -0.5 + z_off],
        mode="lines",
        line=dict(width=8, color=rho_color),
        name=f"Epoch {epoch} G-tube",
        legendgroup=f"e{epoch}",
        showlegend=False,
    ))

    return traces


# ============================================================
# Section 7: Assemble Full Scene
# ============================================================

print("\n[7/8] Assembling full 3D scene...")

all_traces = []
for i in tqdm(range(len(composites)), desc="Building epochs"):
    all_traces.extend(build_epoch_traces(i))

fig = go.Figure(data=all_traces)

fig.update_layout(
    title=dict(
        text="Neural Organ Composite — Epoch Spectral Evolution",
        font=dict(size=20, color="#e0e0ff", family="Courier New, monospace"),
        x=0.5,
    ),
    paper_bgcolor="#0a0a14",
    plot_bgcolor="#0a0a14",
    font=dict(color="#c0c0e0"),
    legend=dict(
        bgcolor="rgba(10,10,20,0.8)",
        bordercolor="#404080",
        borderwidth=1,
        font=dict(size=11, color="#c0c0e0"),
    ),
    scene=dict(
        bgcolor="#0a0a14",
        xaxis=dict(
            title="Re(λ)",
            backgroundcolor="#0a0a14",
            gridcolor="#1a1a3a",
            showspikes=False,
            color="#8080c0",
        ),
        yaxis=dict(
            title="Im(λ)",
            backgroundcolor="#0a0a14",
            gridcolor="#1a1a3a",
            showspikes=False,
            color="#8080c0",
        ),
        zaxis=dict(
            title="Epoch Layer",
            backgroundcolor="#0a0a14",
            gridcolor="#1a1a3a",
            showspikes=False,
            color="#8080c0",
        ),
        camera=dict(
            eye=dict(x=1.8, y=1.8, z=1.2),
            up=dict(x=0, y=0, z=1),
        ),
        aspectmode="data",
    ),
    margin=dict(l=0, r=0, t=50, b=0),
    width=1920,
    height=1080,
)

# ============================================================
# Section 8: Export
# ============================================================

print("\n[8/8] Exporting...")

html_path = OUTPUT_DIR / "NeuralOrgan_Plotly.html"
fig.write_html(str(html_path), include_plotlyjs="cdn")
print(f"✓ Interactive HTML saved: {html_path}")

try:
    png_path = OUTPUT_DIR / "NeuralOrgan_Plotly.png"
    fig.write_image(str(png_path), width=1920, height=1080, scale=2)
    print(f"✓ Static PNG saved:      {png_path}")
except Exception as e:
    print(f"  ⚠ PNG export skipped (install kaleido): {e}")

# Show inline in Colab/Jupyter
try:
    fig.show()
except Exception:
    pass

print("\n" + "=" * 60)
print("✓ Plotly WebGL Script complete!")
print("=" * 60)
print(f"HTML: {html_path}")
print(f"Traces: {len(all_traces)} | Epochs: {len(composites)}")
print("=" * 60)
