"""
Neural Organ Composite — Ultimate Edition
==========================================
All five visualization modes in one script:
  1. Interactive 3D  — Plotly WebGL (HTML)
  2. Camera Orbit    — 360° cinematic MP4
  3. Pulsating Finale— breathing organism MP4
  4. Multi-Epoch Fade— spectral evolution MP4
  5. Volumetric Shell— layered glow (enhanced HTML)
  6. Three.js Cathedral— bloom/glow HTML export

Required Colab installs:
  pip install plotly kaleido==0.2.1 imageio[ffmpeg]

Everything else (numpy, pandas, scipy, matplotlib, tqdm) is pre-installed.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.linalg import eig
from tqdm import tqdm
import json
import warnings
warnings.filterwarnings("ignore")

import plotly.graph_objects as go
import matplotlib.pyplot as plt

# ============================================================
# INSTALL GUARD — auto-install missing deps in Colab
# ============================================================

def _ensure_imports():
    """Try importing optional deps; install if missing."""
    try:
        import imageio
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "imageio[ffmpeg]"])
        import imageio
    return imageio

imageio = _ensure_imports()

# ============================================================
# CONFIGURATION
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
A_MATRIX_DIR = SCRIPT_DIR / "models" / "a_matrix"
OUTPUT_DIR = SCRIPT_DIR / "visualizations"
OUTPUT_DIR.mkdir(exist_ok=True)

# ---- Mode toggles (set to True to enable) ----
ENABLE_INTERACTIVE_HTML  = True    # Mode 1: static interactive 3D HTML
ENABLE_CAMERA_ORBIT_MP4  = True    # Mode 2: 360° orbit MP4
ENABLE_PULSATING_MP4     = True    # Mode 3: pulsating finale MP4
ENABLE_EPOCH_FADE_MP4    = True    # Mode 4: multi-epoch fade MP4
ENABLE_VOLUMETRIC_HTML   = True    # Mode 5: volumetric shell HTML
ENABLE_THREEJS_CATHEDRAL = True    # Mode 6: Three.js bloom/glow HTML

# ---- Geometry ----
LOGIC_SHELL_MODE       = 1
NEON_COLORMAP          = "plasma"
MAX_EIGENVALUES        = None
FIBERS_PER_POINT       = 12
TENSOR_ARROW_COUNT     = 40
LOGIC_SHELL_RESOLUTION = 50
LOGIC_SHELL_RADIUS     = 3.0
THRESHOLD              = 0.01
EPOCH_Z_SPACING        = 8.0

# ---- Video ----
RESOLUTION = (1920, 1080)
FPS        = 24

# ---- Orbit ----
ORBIT_FRAMES    = 120   # 5 seconds at 24fps
ORBIT_ELEVATION = 20.0

# ---- Pulsating ----
PULSE_FRAMES   = 120
PULSE_AMPLITUDE = 0.15
PULSE_CYCLES    = 3

# ---- Epoch Fade ----
FADE_FRAMES_PER_EPOCH = 36  # 1.5 sec per epoch at 24fps
FADE_HOLD_FRAMES      = 12  # 0.5 sec hold

# ---- Volumetric ----
VOLUMETRIC_LAYERS = 5
VOLUMETRIC_OPACITY_RANGE = (0.02, 0.12)

print("=" * 64)
print("  Neural Organ Composite — Ultimate Edition")
print("=" * 64)

# ============================================================
# SECTION 1: LOAD A-MATRICES
# ============================================================

print("\n[1/8] Loading A-matrices...")


def load_a_matrices(directory: Path):
    matrices = []
    epoch_files = sorted(directory.glob("Epoch*_A_Matrix.csv"))
    if not epoch_files:
        print(f"  ✗ No A-matrix files found in {directory}")
        return None
    for csv_file in epoch_files:
        try:
            df = pd.read_csv(csv_file, header=None)
            matrices.append(df.values)
            print(f"  ✓ {csv_file.name}: shape {df.values.shape}")
        except Exception as e:
            print(f"  ✗ {csv_file.name}: {e}")
    return matrices


A_matrices = load_a_matrices(A_MATRIX_DIR)
if not A_matrices:
    print("✗ No valid A-matrices loaded. Exiting.")
    raise SystemExit(1)
print(f"  Total: {len(A_matrices)} matrices")

# ============================================================
# SECTION 2: STABILITY ANALYSIS
# ============================================================

print("\n[2/8] Computing spectral analysis...")


def analyze_matrix(A):
    eigenvalues, eigenvectors = eig(A)
    diag = np.diag(A)
    row_sums = np.sum(np.abs(A), axis=1)
    radii = row_sums - np.abs(diag)
    return {
        "matrix": A,
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors.T,
        "diagonal": diag,
        "radii": radii,
        "spectral_radius": float(np.max(np.abs(eigenvalues))),
        "gershgorin_bound": float(np.max(diag + radii)),
    }


results = [analyze_matrix(A) for A in A_matrices]
rho_all   = np.array([r["spectral_radius"]  for r in results])
gersh_all = np.array([r["gershgorin_bound"] for r in results])
min_rho,   max_rho   = float(rho_all.min()),   float(rho_all.max())
min_gersh, max_gersh = float(gersh_all.min()), float(gersh_all.max())
print(f"  ρ range:  [{min_rho:.6f}, {max_rho:.6f}]")
print(f"  G range:  [{min_gersh:.6f}, {max_gersh:.6f}]")

# ============================================================
# SECTION 3: COLOR HELPERS
# ============================================================

print("\n[3/8] Initializing color system...")


def _cmap_rgb(value, vmin, vmax):
    """Return (r,g,b) floats 0-1 from matplotlib colormap."""
    norm = np.clip((value - vmin) / (vmax - vmin + 1e-10), 0.0, 1.0)
    cmap = plt.cm.get_cmap(NEON_COLORMAP)
    r, g, b, _ = cmap(float(norm))
    return r, g, b


def plotly_rgb(value, vmin, vmax):
    r, g, b = _cmap_rgb(value, vmin, vmax)
    return f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"


def plotly_rgba(value, vmin, vmax, alpha=0.5):
    r, g, b = _cmap_rgb(value, vmin, vmax)
    return f"rgba({int(r*255)},{int(g*255)},{int(b*255)},{alpha})"


def hex_color(value, vmin, vmax):
    r, g, b = _cmap_rgb(value, vmin, vmax)
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"


# ============================================================
# SECTION 4: GEOMETRY PIPELINE
# ============================================================

print("\n[4/8] Building geometry pipeline...")


def compute_composite(result, epoch):
    A = result["matrix"]
    eigenvalues = result["eigenvalues"]
    eigenvectors = result["eigenvectors"]
    radii = result["radii"]

    n = len(eigenvalues) if MAX_EIGENVALUES is None else min(len(eigenvalues), MAX_EIGENVALUES)
    eigenvalues  = eigenvalues[:n]
    eigenvectors = eigenvectors[:n]
    radii        = radii[:n]

    re_vals = np.real(eigenvalues)
    im_vals = np.imag(eigenvalues)
    pts = np.column_stack([re_vals, im_vals, radii])
    pts -= np.mean(pts, axis=0)

    spheres = [(pts[i].copy(), float(radii[i])) for i in range(n)]

    fibers = []
    for i in range(n):
        p = pts[i]
        vec = np.real(eigenvectors[i][:min(3, len(eigenvectors[i]))])
        vec = np.pad(vec, (0, max(0, 3 - len(vec))))
        d = vec / (np.linalg.norm(vec) + 1e-10) if np.linalg.norm(vec) > 0 else np.array([0,0,1.0])
        for t in np.linspace(0, 2*np.pi, FIBERS_PER_POINT, endpoint=False):
            perturb = d + 0.2 * np.array([np.cos(t), np.sin(t), 0.0])
            perturb /= np.linalg.norm(perturb)
            fibers.append((p.copy(), p + 0.15 * perturb))

    tensor_arrows = []
    rows = A[:TENSOR_ARROW_COUNT, :min(3, A.shape[1])]
    for i, row in enumerate(rows):
        base = np.array([float(i), -1.5, 0.0])
        rp = np.pad(row, (0, max(0, 3 - len(row))))
        d = rp / (np.linalg.norm(rp) + 1e-10) if np.linalg.norm(rp) > 0 else np.array([0,0,1.0])
        tensor_arrows.append((base, base + 0.3 * d))

    return {
        "points": pts, "spheres": spheres, "fibers": fibers,
        "tensor_arrows": tensor_arrows,
        "rho": result["spectral_radius"], "gersh": result["gershgorin_bound"],
        "epoch": epoch,
    }


composites = [compute_composite(results[i], i+1) for i in range(len(results))]

logic_maps = [np.sign(np.clip(r["matrix"], -THRESHOLD, THRESHOLD)) for r in results]


def logic_shell_points(logic_matrix):
    n = logic_matrix.shape[0]
    res = LOGIC_SHELL_RESOLUTION
    coords, vals = [], []
    for i in range(res):
        theta = i / res * np.pi
        for j in range(res):
            phi = j / res * 2 * np.pi
            x = LOGIC_SHELL_RADIUS * np.sin(theta) * np.cos(phi)
            y = LOGIC_SHELL_RADIUS * np.sin(theta) * np.sin(phi)
            z = LOGIC_SHELL_RADIUS * np.cos(theta)
            coords.append([x, y, z])
            r = min(int((i / res) * n), n-1)
            c = min(int((j / res) * n), n-1)
            vals.append(logic_matrix[r, c])
    return np.array(coords), np.array(vals)


print(f"  {len(composites)} epoch composites ready")

# ============================================================
# SECTION 5: PLOTLY TRACE BUILDER
# ============================================================

print("\n[5/8] Building Plotly traces...")


def build_epoch_traces(idx, z_off=None, opacity_scale=1.0, point_scale=1.0):
    """Build Plotly traces for one epoch with configurable offsets and opacity."""
    c = composites[idx]
    if z_off is None:
        z_off = idx * EPOCH_Z_SPACING
    rho = c["rho"]
    epoch = c["epoch"]
    traces = []

    # Logic shell
    coords, vals = logic_shell_points(logic_maps[idx])
    coords = coords * point_scale
    mask = vals != 0
    sc, sv = coords[mask], vals[mask]
    shell_colors = [
        plotly_rgba(1.0, 0, 1, alpha=0.15 * opacity_scale) if v == 1
        else plotly_rgba(0.0, 0, 1, alpha=0.15 * opacity_scale)
        for v in sv
    ]
    traces.append(go.Scatter3d(
        x=sc[:,0], y=sc[:,1], z=sc[:,2] + z_off,
        mode="markers", marker=dict(size=1.8, color=shell_colors),
        name=f"E{epoch} Shell", legendgroup=f"e{epoch}",
        showlegend=False, hoverinfo="skip",
    ))

    # Gershgorin spheres (wireframe)
    u = np.linspace(0, 2*np.pi, 16)
    v = np.linspace(0, np.pi, 8)
    for si, (pos, radius) in enumerate(c["spheres"][:16]):
        pos = pos * point_scale
        radius = radius * point_scale
        for vi in range(len(v)):
            cx = pos[0] + radius * np.sin(v[vi]) * np.cos(u)
            cy = pos[1] + radius * np.sin(v[vi]) * np.sin(u)
            cz = np.full_like(cx, pos[2] + radius * np.cos(v[vi]) + z_off)
            traces.append(go.Scatter3d(
                x=cx, y=cy, z=cz, mode="lines",
                line=dict(width=1, color=plotly_rgba(radius, min_gersh, max_gersh, alpha=0.08 * opacity_scale)),
                showlegend=False, hoverinfo="skip", legendgroup=f"e{epoch}",
            ))

    # Eigenvalue cloud
    pts = c["points"] * point_scale
    traces.append(go.Scatter3d(
        x=pts[:,0], y=pts[:,1], z=pts[:,2] + z_off,
        mode="markers",
        marker=dict(size=4, color=plotly_rgb(rho, min_rho, max_rho), opacity=0.7 * opacity_scale),
        name=f"Epoch {epoch} (ρ={rho:.4f})", legendgroup=f"e{epoch}", showlegend=True,
    ))

    # Fibers
    fx, fy, fz = [], [], []
    for s, e in c["fibers"]:
        s2, e2 = s * point_scale, e * point_scale
        fx.extend([s2[0], e2[0], None])
        fy.extend([s2[1], e2[1], None])
        fz.extend([s2[2]+z_off, e2[2]+z_off, None])
    traces.append(go.Scatter3d(
        x=fx, y=fy, z=fz, mode="lines",
        line=dict(width=1.5, color=plotly_rgba(0.7, 0, 1, alpha=0.35 * opacity_scale)),
        showlegend=False, hoverinfo="skip", legendgroup=f"e{epoch}",
    ))

    # Tensor arrows
    tx, ty, tz = [], [], []
    for s, e in c["tensor_arrows"]:
        tx.extend([s[0], e[0], None])
        ty.extend([s[1], e[1], None])
        tz.extend([s[2]+z_off, e[2]+z_off, None])
    traces.append(go.Scatter3d(
        x=tx, y=ty, z=tz, mode="lines",
        line=dict(width=2.5, color=plotly_rgba(0.3, 0, 1, alpha=0.25 * opacity_scale)),
        showlegend=False, hoverinfo="skip", legendgroup=f"e{epoch}",
    ))

    # Stability tubes
    gersh = c["gersh"]
    rho_h = 0.2 * (rho - min_rho) / (max_rho - min_rho + 1e-10) * point_scale
    gersh_h = 0.2 * (gersh - min_gersh) / (max_gersh - min_gersh + 1e-10) * point_scale
    bx = float(epoch)
    rc = plotly_rgb(rho, min_rho, max_rho)
    traces.append(go.Scatter3d(
        x=[bx, bx+0.1], y=[2.0, 2.0+rho_h], z=[z_off, z_off],
        mode="lines", line=dict(width=8, color=rc),
        showlegend=False, legendgroup=f"e{epoch}",
    ))
    traces.append(go.Scatter3d(
        x=[bx, bx+0.1], y=[2.0, 2.0+gersh_h], z=[-0.5+z_off, -0.5+z_off],
        mode="lines", line=dict(width=8, color=rc),
        showlegend=False, legendgroup=f"e{epoch}",
    ))

    return traces


def make_layout(title="Neural Organ Composite", camera=None):
    """Shared dark-neon Plotly layout."""
    if camera is None:
        camera = dict(eye=dict(x=1.8, y=1.8, z=1.2), up=dict(x=0, y=0, z=1))
    return go.Layout(
        title=dict(text=title, font=dict(size=20, color="#e0e0ff", family="Courier New, monospace"), x=0.5),
        paper_bgcolor="#0a0a14", plot_bgcolor="#0a0a14",
        font=dict(color="#c0c0e0"),
        legend=dict(bgcolor="rgba(10,10,20,0.8)", bordercolor="#404080", borderwidth=1,
                    font=dict(size=11, color="#c0c0e0")),
        scene=dict(
            bgcolor="#0a0a14",
            xaxis=dict(title="Re(λ)", backgroundcolor="#0a0a14", gridcolor="#1a1a3a", showspikes=False, color="#8080c0"),
            yaxis=dict(title="Im(λ)", backgroundcolor="#0a0a14", gridcolor="#1a1a3a", showspikes=False, color="#8080c0"),
            zaxis=dict(title="Epoch",  backgroundcolor="#0a0a14", gridcolor="#1a1a3a", showspikes=False, color="#8080c0"),
            camera=camera, aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=50, b=0), width=RESOLUTION[0], height=RESOLUTION[1],
    )


def export_fig_frame(fig, path):
    """Export a single frame as PNG via kaleido."""
    fig.write_image(str(path), width=RESOLUTION[0], height=RESOLUTION[1], scale=1)


def frames_to_mp4(frames_dir, output_path, fps=FPS):
    """Assemble numbered PNGs into MP4."""
    pngs = sorted(frames_dir.glob("frame_*.png"))
    if not pngs:
        print(f"  ✗ No frames found in {frames_dir}")
        return
    with imageio.get_writer(str(output_path), fps=fps, codec="libx264",
                            quality=8, pixelformat="yuv420p") as w:
        for p in tqdm(pngs, desc="  Encoding MP4"):
            w.append_data(imageio.imread(str(p)))
    # Cleanup frames
    for p in pngs:
        p.unlink()
    print(f"  ✓ {output_path.name} ({len(pngs)} frames, {fps}fps)")

# ============================================================
# MODE 1: INTERACTIVE HTML
# ============================================================

if ENABLE_INTERACTIVE_HTML:
    print("\n" + "─" * 64)
    print("  MODE 1: Interactive 3D HTML")
    print("─" * 64)

    all_traces = []
    for i in range(len(composites)):
        all_traces.extend(build_epoch_traces(i))

    fig = go.Figure(data=all_traces, layout=make_layout("Neural Organ — Epoch Spectral Evolution"))

    html_path = OUTPUT_DIR / "NeuralOrgan_Interactive.html"
    fig.write_html(str(html_path), include_plotlyjs="cdn")
    print(f"  ✓ {html_path.name}")
    try:
        fig.show()
    except Exception:
        pass

# ============================================================
# MODE 2: CAMERA ORBIT MP4
# ============================================================

if ENABLE_CAMERA_ORBIT_MP4:
    print("\n" + "─" * 64)
    print("  MODE 2: 360° Camera Orbit MP4")
    print("─" * 64)

    orbit_dir = OUTPUT_DIR / "_orbit_frames"
    orbit_dir.mkdir(exist_ok=True)

    # Build traces once (all epochs stacked)
    orbit_traces = []
    for i in range(len(composites)):
        orbit_traces.extend(build_epoch_traces(i))

    for f in tqdm(range(ORBIT_FRAMES), desc="  Rendering orbit"):
        angle = (f / ORBIT_FRAMES) * 360.0
        rad = np.radians(angle)
        dist = 2.5
        cam = dict(
            eye=dict(x=dist*np.cos(rad), y=dist*np.sin(rad), z=0.8 + 0.3*np.sin(rad*2)),
            up=dict(x=0, y=0, z=1),
        )
        fig = go.Figure(data=orbit_traces, layout=make_layout("Neural Organ — 360° Orbit", camera=cam))
        export_fig_frame(fig, orbit_dir / f"frame_{f:04d}.png")

    frames_to_mp4(orbit_dir, OUTPUT_DIR / "NeuralOrgan_Orbit360.mp4")
    orbit_dir.rmdir()

# ============================================================
# MODE 3: PULSATING FINALE MP4
# ============================================================

if ENABLE_PULSATING_MP4:
    print("\n" + "─" * 64)
    print("  MODE 3: Pulsating Finale MP4")
    print("─" * 64)

    pulse_dir = OUTPUT_DIR / "_pulse_frames"
    pulse_dir.mkdir(exist_ok=True)

    last_idx = len(composites) - 1

    for f in tqdm(range(PULSE_FRAMES), desc="  Rendering pulse"):
        t = f / PULSE_FRAMES
        scale = 1.0 + PULSE_AMPLITUDE * np.sin(2 * np.pi * PULSE_CYCLES * t)
        angle = 45.0 + t * 360.0
        rad = np.radians(angle)
        elev_rad = np.radians(ORBIT_ELEVATION)
        dist = 2.5

        pulse_traces = build_epoch_traces(last_idx, z_off=0, point_scale=scale)
        cam = dict(
            eye=dict(x=dist*np.cos(rad)*np.cos(elev_rad),
                     y=dist*np.sin(rad)*np.cos(elev_rad),
                     z=dist*np.sin(elev_rad)),
            up=dict(x=0, y=0, z=1),
        )
        fig = go.Figure(data=pulse_traces, layout=make_layout(
            f"Neural Organ — Pulsating (scale {scale:.2f})", camera=cam))
        export_fig_frame(fig, pulse_dir / f"frame_{f:04d}.png")

    frames_to_mp4(pulse_dir, OUTPUT_DIR / "NeuralOrgan_Pulse.mp4")
    pulse_dir.rmdir()

# ============================================================
# MODE 4: MULTI-EPOCH FADE MP4
# ============================================================

if ENABLE_EPOCH_FADE_MP4:
    print("\n" + "─" * 64)
    print("  MODE 4: Multi-Epoch Fade Animation MP4")
    print("─" * 64)

    fade_dir = OUTPUT_DIR / "_fade_frames"
    fade_dir.mkdir(exist_ok=True)

    n_epochs = len(composites)
    total_fade_frames = n_epochs * (FADE_FRAMES_PER_EPOCH + FADE_HOLD_FRAMES)
    frame_num = 0

    for ei in tqdm(range(n_epochs), desc="  Rendering epochs"):
        # Fade in
        for fi in range(FADE_FRAMES_PER_EPOCH):
            alpha = (fi + 1) / FADE_FRAMES_PER_EPOCH
            traces = []
            # Previous epochs at full opacity
            for prev in range(ei):
                traces.extend(build_epoch_traces(prev, z_off=prev * EPOCH_Z_SPACING))
            # Current epoch fading in
            traces.extend(build_epoch_traces(ei, z_off=ei * EPOCH_Z_SPACING, opacity_scale=alpha))

            angle = 45.0 + frame_num * 1.5
            rad = np.radians(angle)
            cam = dict(
                eye=dict(x=2.5*np.cos(rad), y=2.5*np.sin(rad), z=1.0 + ei*0.3),
                up=dict(x=0, y=0, z=1),
            )
            fig = go.Figure(data=traces, layout=make_layout(
                f"Spectral Evolution — Epoch {ei+1}/{n_epochs}", camera=cam))
            export_fig_frame(fig, fade_dir / f"frame_{frame_num:04d}.png")
            frame_num += 1

        # Hold
        for _ in range(FADE_HOLD_FRAMES):
            traces = []
            for prev in range(ei + 1):
                traces.extend(build_epoch_traces(prev, z_off=prev * EPOCH_Z_SPACING))
            angle = 45.0 + frame_num * 1.5
            rad = np.radians(angle)
            cam = dict(
                eye=dict(x=2.5*np.cos(rad), y=2.5*np.sin(rad), z=1.0 + ei*0.3),
                up=dict(x=0, y=0, z=1),
            )
            fig = go.Figure(data=traces, layout=make_layout(
                f"Spectral Evolution — Epoch {ei+1}/{n_epochs}", camera=cam))
            export_fig_frame(fig, fade_dir / f"frame_{frame_num:04d}.png")
            frame_num += 1

    frames_to_mp4(fade_dir, OUTPUT_DIR / "NeuralOrgan_EpochFade.mp4")
    fade_dir.rmdir()

# ============================================================
# MODE 5: VOLUMETRIC SHELL HTML
# ============================================================

if ENABLE_VOLUMETRIC_HTML:
    print("\n" + "─" * 64)
    print("  MODE 5: Volumetric Shell HTML")
    print("─" * 64)

    vol_traces = []
    # Use last epoch for the volumetric showcase
    last = len(composites) - 1

    # Standard traces (eigenvalues, fibers, tensor, tubes)
    vol_traces.extend(build_epoch_traces(last, z_off=0))

    # Layered volumetric shells with radial glow
    logic = logic_maps[last]
    for layer in range(VOLUMETRIC_LAYERS):
        t = layer / (VOLUMETRIC_LAYERS - 1)
        radius_mult = 0.6 + t * 0.6   # inner to outer
        opacity = VOLUMETRIC_OPACITY_RANGE[1] - t * (VOLUMETRIC_OPACITY_RANGE[1] - VOLUMETRIC_OPACITY_RANGE[0])

        coords, vals = logic_shell_points(logic)
        coords = coords * radius_mult
        mask = vals != 0
        sc, sv = coords[mask], vals[mask]

        # Hue shift per layer
        hue_shift = t * 0.3
        colors = [
            plotly_rgba(1.0 - hue_shift, 0, 1, alpha=opacity) if v == 1
            else plotly_rgba(0.0 + hue_shift, 0, 1, alpha=opacity)
            for v in sv
        ]

        vol_traces.append(go.Scatter3d(
            x=sc[:,0], y=sc[:,1], z=sc[:,2],
            mode="markers", marker=dict(size=2.5 - t*0.8, color=colors),
            showlegend=False, hoverinfo="skip",
        ))

    fig = go.Figure(data=vol_traces, layout=make_layout("Neural Organ — Volumetric Cathedral"))
    vol_path = OUTPUT_DIR / "NeuralOrgan_Volumetric.html"
    fig.write_html(str(vol_path), include_plotlyjs="cdn")
    print(f"  ✓ {vol_path.name}")
    try:
        fig.show()
    except Exception:
        pass

# ============================================================
# MODE 6: THREE.JS NEON CATHEDRAL
# ============================================================

if ENABLE_THREEJS_CATHEDRAL:
    print("\n" + "─" * 64)
    print("  MODE 6: Three.js Neon Cathedral HTML")
    print("─" * 64)

    # Serialize geometry to JSON for Three.js
    last = len(composites) - 1
    c = composites[last]
    r = results[last]

    # Eigenvalue points
    eigen_pts = c["points"].tolist()
    eigen_color = hex_color(c["rho"], min_rho, max_rho)

    # Fiber lines
    fiber_lines = [(s.tolist(), e.tolist()) for s, e in c["fibers"]]
    fiber_color = hex_color(0.7, 0, 1)

    # Tensor arrows
    tensor_lines = [(s.tolist(), e.tolist()) for s, e in c["tensor_arrows"]]
    tensor_color = hex_color(0.3, 0, 1)

    # Gershgorin sphere data
    gersh_spheres = [(pos.tolist(), float(rad)) for pos, rad in c["spheres"][:32]]
    sphere_colors = [hex_color(rad, min_gersh, max_gersh) for _, rad in gersh_spheres]

    # Logic shell
    coords, vals = logic_shell_points(logic_maps[last])
    shell_mask = vals != 0
    shell_pts = coords[shell_mask].tolist()
    shell_vals = vals[shell_mask].tolist()
    shell_pos_color = hex_color(1.0, 0, 1)
    shell_neg_color = hex_color(0.0, 0, 1)

    geo_json = json.dumps({
        "eigen_pts": eigen_pts, "eigen_color": eigen_color,
        "fibers": fiber_lines, "fiber_color": fiber_color,
        "tensors": tensor_lines, "tensor_color": tensor_color,
        "spheres": gersh_spheres, "sphere_colors": sphere_colors,
        "shell_pts": shell_pts, "shell_vals": shell_vals,
        "shell_pos_color": shell_pos_color, "shell_neg_color": shell_neg_color,
        "rho": c["rho"], "gersh": c["gersh"], "epoch": c["epoch"],
    })

    threejs_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Neural Organ — Neon Cathedral</title>
<style>
  * {{ margin: 0; padding: 0; }}
  body {{ background: #050510; overflow: hidden; font-family: 'Courier New', monospace; }}
  canvas {{ display: block; }}
  #info {{
    position: absolute; top: 15px; left: 50%; transform: translateX(-50%);
    color: #b0b0ff; font-size: 14px; text-align: center;
    text-shadow: 0 0 10px #6060ff;
    pointer-events: none; z-index: 10;
  }}
  #stats {{
    position: absolute; bottom: 15px; left: 15px;
    color: #707090; font-size: 11px; z-index: 10;
    pointer-events: none;
  }}
</style>
</head>
<body>
<div id="info">
  Neural Organ — Neon Cathedral<br>
  <span style="font-size:11px; color:#8080a0">
    Epoch {{}} | ρ = {{:.6f}} | Gershgorin = {{:.6f}}
  </span>
</div>
<div id="stats"></div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/postprocessing/EffectComposer.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/postprocessing/RenderPass.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/postprocessing/UnrealBloomPass.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/shaders/CopyShader.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/shaders/LuminosityHighPassShader.js"></script>

<script>
const GEO = {geo_json};

// ---- Renderer ----
const renderer = new THREE.WebGLRenderer({{ antialias: true }});
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.2;
document.body.appendChild(renderer.domElement);

// ---- Scene ----
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x050510);
scene.fog = new THREE.FogExp2(0x050510, 0.04);

// ---- Camera ----
const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 200);
camera.position.set(5, 5, 4);

// ---- Controls ----
const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.05;
controls.autoRotate = true;
controls.autoRotateSpeed = 0.8;

// ---- Bloom Post-Processing ----
const composer = new THREE.EffectComposer(renderer);
composer.addPass(new THREE.RenderPass(scene, camera));

const bloomPass = new THREE.UnrealBloomPass(
    new THREE.Vector2(window.innerWidth, window.innerHeight),
    1.5,   // strength
    0.4,   // radius
    0.85   // threshold
);
composer.addPass(bloomPass);

// ---- Ambient Light ----
scene.add(new THREE.AmbientLight(0x303060, 0.5));
const pointLight = new THREE.PointLight(0x8080ff, 2, 50);
pointLight.position.set(0, 0, 5);
scene.add(pointLight);

// ---- Build Geometry ----

// Eigenvalue cloud — glowing spheres
const eigenGroup = new THREE.Group();
GEO.eigen_pts.forEach(pt => {{
    const geo = new THREE.SphereGeometry(0.04, 12, 8);
    const mat = new THREE.MeshStandardMaterial({{
        color: new THREE.Color(GEO.eigen_color),
        emissive: new THREE.Color(GEO.eigen_color),
        emissiveIntensity: 2.0,
        metalness: 0.3,
        roughness: 0.5,
    }});
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.set(pt[0], pt[1], pt[2]);
    eigenGroup.add(mesh);
}});
scene.add(eigenGroup);

// Gershgorin spheres — emissive wireframes
GEO.spheres.forEach((sph, i) => {{
    const geo = new THREE.SphereGeometry(sph[1], 16, 12);
    const mat = new THREE.MeshStandardMaterial({{
        color: new THREE.Color(GEO.sphere_colors[i] || '#4040a0'),
        emissive: new THREE.Color(GEO.sphere_colors[i] || '#4040a0'),
        emissiveIntensity: 0.8,
        transparent: true,
        opacity: 0.06,
        wireframe: true,
    }});
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.set(sph[0][0], sph[0][1], sph[0][2]);
    scene.add(mesh);
}});

// Fiber halo — emissive lines
const fiberGeo = new THREE.BufferGeometry();
const fiberVerts = [];
GEO.fibers.forEach(pair => {{
    fiberVerts.push(pair[0][0], pair[0][1], pair[0][2]);
    fiberVerts.push(pair[1][0], pair[1][1], pair[1][2]);
}});
fiberGeo.setAttribute('position', new THREE.Float32BufferAttribute(fiberVerts, 3));
const fiberMat = new THREE.LineBasicMaterial({{
    color: new THREE.Color(GEO.fiber_color),
    transparent: true,
    opacity: 0.5,
    linewidth: 1,
}});
scene.add(new THREE.LineSegments(fiberGeo, fiberMat));

// Tensor arrows
const tensorGeo = new THREE.BufferGeometry();
const tensorVerts = [];
GEO.tensors.forEach(pair => {{
    tensorVerts.push(pair[0][0], pair[0][1], pair[0][2]);
    tensorVerts.push(pair[1][0], pair[1][1], pair[1][2]);
}});
tensorGeo.setAttribute('position', new THREE.Float32BufferAttribute(tensorVerts, 3));
const tensorMat = new THREE.LineBasicMaterial({{
    color: new THREE.Color(GEO.tensor_color),
    transparent: true,
    opacity: 0.3,
    linewidth: 1,
}});
scene.add(new THREE.LineSegments(tensorGeo, tensorMat));

// Logic shell — point cloud with emissive colors
const shellGeo = new THREE.BufferGeometry();
const shellVerts = [];
const shellColors = [];
GEO.shell_pts.forEach((pt, i) => {{
    shellVerts.push(pt[0], pt[1], pt[2]);
    const c = new THREE.Color(GEO.shell_vals[i] === 1 ? GEO.shell_pos_color : GEO.shell_neg_color);
    shellColors.push(c.r, c.g, c.b);
}});
shellGeo.setAttribute('position', new THREE.Float32BufferAttribute(shellVerts, 3));
shellGeo.setAttribute('color', new THREE.Float32BufferAttribute(shellColors, 3));
const shellMat = new THREE.PointsMaterial({{
    size: 0.04,
    vertexColors: true,
    transparent: true,
    opacity: 0.15,
    sizeAttenuation: true,
}});
scene.add(new THREE.Points(shellGeo, shellMat));

// ---- Animation Loop ----
let time = 0;
const statsDiv = document.getElementById('stats');

function animate() {{
    requestAnimationFrame(animate);
    time += 0.01;

    // Pulsate eigenvalue cloud
    const pulse = 1.0 + 0.03 * Math.sin(time * 3.0);
    eigenGroup.scale.set(pulse, pulse, pulse);

    // Drift point light
    pointLight.position.x = 3 * Math.cos(time * 0.5);
    pointLight.position.y = 3 * Math.sin(time * 0.5);
    pointLight.intensity = 2.0 + 0.5 * Math.sin(time * 2.0);

    // Subtle bloom breathing
    bloomPass.strength = 1.5 + 0.3 * Math.sin(time * 1.5);

    controls.update();
    composer.render();

    statsDiv.textContent = 'FPS: ' + Math.round(1.0 / (performance.now() / 1000 / (time * 100 + 1)));
}}

animate();

// ---- Resize ----
window.addEventListener('resize', () => {{
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
    composer.setSize(window.innerWidth, window.innerHeight);
}});
</script>
</body>
</html>""".replace(
        "Epoch {} | ρ = {:.6f} | Gershgorin = {:.6f}",
        f"Epoch {c['epoch']} | ρ = {c['rho']:.6f} | Gershgorin = {c['gersh']:.6f}"
    )

    threejs_path = OUTPUT_DIR / "NeuralOrgan_Cathedral.html"
    with open(threejs_path, "w", encoding="utf-8") as f:
        f.write(threejs_html)
    print(f"  ✓ {threejs_path.name}")

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 64)
print("  ✓ Ultimate Edition complete!")
print("=" * 64)
outputs = []
if ENABLE_INTERACTIVE_HTML:  outputs.append("NeuralOrgan_Interactive.html")
if ENABLE_CAMERA_ORBIT_MP4:  outputs.append("NeuralOrgan_Orbit360.mp4")
if ENABLE_PULSATING_MP4:     outputs.append("NeuralOrgan_Pulse.mp4")
if ENABLE_EPOCH_FADE_MP4:    outputs.append("NeuralOrgan_EpochFade.mp4")
if ENABLE_VOLUMETRIC_HTML:   outputs.append("NeuralOrgan_Volumetric.html")
if ENABLE_THREEJS_CATHEDRAL: outputs.append("NeuralOrgan_Cathedral.html")
for o in outputs:
    print(f"  → visualizations/{o}")
print("=" * 64)
