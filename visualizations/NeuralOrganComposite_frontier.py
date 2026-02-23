"""
Neural Organ Composite — Next Frontier Edition
================================================
Extends the Ultimate Edition with 5 new capabilities:
  A. OpenCV Bloom/Glow post-processing on MP4 frames
  B. Enhanced Three.js Neon Cathedral (god rays, particles, chromatic aberration)
  C. Blender Cycles GPU export (.py script)
  D. Interactive Epoch Scrubber (Plotly slider)
  E. Multi-Organ Composite (multiple training runs)

Required installs (Colab):
  pip install plotly kaleido==0.2.1 imageio[ffmpeg] opencv-python-headless

Pre-installed: numpy, pandas, scipy, matplotlib, tqdm
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
# INSTALL GUARD
# ============================================================

def _ensure_imports():
    mods = {}
    for pkg, pip_name in [("imageio", "imageio[ffmpeg]"), ("cv2", "opencv-python-headless")]:
        try:
            mods[pkg] = __import__(pkg)
        except ImportError:
            import subprocess, sys
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip_name])
            mods[pkg] = __import__(pkg)
    return mods

_mods = _ensure_imports()
imageio = _mods["imageio"]
cv2 = _mods["cv2"]

# ============================================================
# CONFIGURATION
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
A_MATRIX_DIR = SCRIPT_DIR / "models" / "a_matrix"
OUTPUT_DIR = SCRIPT_DIR / "visualizations"
OUTPUT_DIR.mkdir(exist_ok=True)

# ---- Mode toggles ----
ENABLE_BLOOM_MP4         = False   # A: OpenCV bloom orbit MP4
ENABLE_THREEJS_CATHEDRAL = True    # B: Enhanced Three.js
ENABLE_BLENDER_EXPORT    = False   # C: Blender Cycles .py
ENABLE_EPOCH_SCRUBBER    = False   # D: Interactive slider HTML
ENABLE_MULTI_ORGAN       = False   # E: Multi-organ (needs extra dirs)

# ---- Multi-Organ directories (Mode E) ----
MULTI_ORGAN_DIRS = [
    SCRIPT_DIR / "models" / "a_matrix",
    # Add more: SCRIPT_DIR / "models" / "a_matrix_v2",
]
MULTI_ORGAN_LABELS = ["Run 1"]  # One label per dir

# ---- Geometry ----
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
FPS = 24

# ---- Bloom ----
BLOOM_FRAMES    = 120
BLOOM_INTENSITY = 0.6
BLOOM_KERNEL    = 51
CHROMATIC_SHIFT = 3       # pixel offset for RGB split
VIGNETTE_STRENGTH = 0.4

print("=" * 64)
print("  Neural Organ Composite — Next Frontier Edition")
print("=" * 64)

# ============================================================
# SECTION 1: LOAD A-MATRICES
# ============================================================

print("\n[1/6] Loading A-matrices...")

def load_a_matrices(directory: Path):
    matrices = []
    epoch_files = sorted(directory.glob("Epoch*_A_Matrix.csv"))
    if not epoch_files:
        print(f"  ✗ No files in {directory}")
        return None
    for f in epoch_files:
        try:
            df = pd.read_csv(f, header=None)
            matrices.append(df.values)
            print(f"  ✓ {f.name}: {df.values.shape}")
        except Exception as e:
            print(f"  ✗ {f.name}: {e}")
    return matrices

A_matrices = load_a_matrices(A_MATRIX_DIR)
if not A_matrices:
    raise SystemExit("✗ No valid A-matrices. Exiting.")
print(f"  Total: {len(A_matrices)} matrices")

# ============================================================
# SECTION 2: SPECTRAL ANALYSIS
# ============================================================

print("\n[2/6] Spectral analysis...")

def analyze_matrix(A):
    eigenvalues, eigenvectors = eig(A)
    diag = np.diag(A)
    row_sums = np.sum(np.abs(A), axis=1)
    radii = row_sums - np.abs(diag)
    return {
        "matrix": A, "eigenvalues": eigenvalues, "eigenvectors": eigenvectors.T,
        "diagonal": diag, "radii": radii,
        "spectral_radius": float(np.max(np.abs(eigenvalues))),
        "gershgorin_bound": float(np.max(diag + radii)),
    }

results = [analyze_matrix(A) for A in A_matrices]
rho_all   = np.array([r["spectral_radius"]  for r in results])
gersh_all = np.array([r["gershgorin_bound"] for r in results])
min_rho, max_rho     = float(rho_all.min()), float(rho_all.max())
min_gersh, max_gersh = float(gersh_all.min()), float(gersh_all.max())
print(f"  ρ: [{min_rho:.6f}, {max_rho:.6f}]  G: [{min_gersh:.6f}, {max_gersh:.6f}]")

# ============================================================
# SECTION 3: COLOR SYSTEM
# ============================================================

def _cmap_rgb(value, vmin, vmax):
    norm = np.clip((value - vmin) / (vmax - vmin + 1e-10), 0.0, 1.0)
    r, g, b, _ = plt.cm.get_cmap(NEON_COLORMAP)(float(norm))
    return r, g, b

def plotly_rgb(v, lo, hi):
    r,g,b = _cmap_rgb(v,lo,hi); return f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"

def plotly_rgba(v, lo, hi, a=0.5):
    r,g,b = _cmap_rgb(v,lo,hi); return f"rgba({int(r*255)},{int(g*255)},{int(b*255)},{a})"

def hex_color(v, lo, hi):
    r,g,b = _cmap_rgb(v,lo,hi); return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"

# ============================================================
# SECTION 4: GEOMETRY PIPELINE
# ============================================================

print("\n[3/6] Geometry pipeline...")

def compute_composite(result, epoch):
    A = result["matrix"]
    ev = result["eigenvalues"]
    evec = result["eigenvectors"]
    radii = result["radii"]
    n = len(ev) if MAX_EIGENVALUES is None else min(len(ev), MAX_EIGENVALUES)
    ev, evec, radii = ev[:n], evec[:n], radii[:n]

    pts = np.column_stack([np.real(ev), np.imag(ev), radii])
    pts -= np.mean(pts, axis=0)
    spheres = [(pts[i].copy(), float(radii[i])) for i in range(n)]

    fibers = []
    for i in range(n):
        p = pts[i]
        v3 = np.real(evec[i][:min(3, len(evec[i]))])
        v3 = np.pad(v3, (0, max(0, 3-len(v3))))
        d = v3/(np.linalg.norm(v3)+1e-10) if np.linalg.norm(v3)>0 else np.array([0,0,1.0])
        for t in np.linspace(0, 2*np.pi, FIBERS_PER_POINT, endpoint=False):
            perturb = d + 0.2*np.array([np.cos(t), np.sin(t), 0.0])
            perturb /= np.linalg.norm(perturb)
            fibers.append((p.copy(), p + 0.15*perturb))

    tensor_arrows = []
    rows = A[:TENSOR_ARROW_COUNT, :min(3, A.shape[1])]
    for i, row in enumerate(rows):
        base = np.array([float(i), -1.5, 0.0])
        rp = np.pad(row, (0, max(0, 3-len(row))))
        d = rp/(np.linalg.norm(rp)+1e-10) if np.linalg.norm(rp)>0 else np.array([0,0,1.0])
        tensor_arrows.append((base, base + 0.3*d))

    return {"points": pts, "spheres": spheres, "fibers": fibers,
            "tensor_arrows": tensor_arrows, "rho": result["spectral_radius"],
            "gersh": result["gershgorin_bound"], "epoch": epoch}

composites = [compute_composite(results[i], i+1) for i in range(len(results))]
logic_maps = [np.sign(np.clip(r["matrix"], -THRESHOLD, THRESHOLD)) for r in results]

def logic_shell_points(logic_matrix):
    n = logic_matrix.shape[0]
    res = LOGIC_SHELL_RESOLUTION
    coords, vals = [], []
    for i in range(res):
        theta = i/res * np.pi
        for j in range(res):
            phi = j/res * 2*np.pi
            coords.append([LOGIC_SHELL_RADIUS*np.sin(theta)*np.cos(phi),
                           LOGIC_SHELL_RADIUS*np.sin(theta)*np.sin(phi),
                           LOGIC_SHELL_RADIUS*np.cos(theta)])
            r = min(int((i/res)*n), n-1)
            c = min(int((j/res)*n), n-1)
            vals.append(logic_matrix[r, c])
    return np.array(coords), np.array(vals)

print(f"  {len(composites)} composites ready")

# ============================================================
# SECTION 5: SHARED PLOTLY HELPERS
# ============================================================

print("\n[4/6] Initializing renderers...")

def build_epoch_traces(idx, z_off=None, opacity_scale=1.0, point_scale=1.0):
    c = composites[idx]
    if z_off is None: z_off = idx * EPOCH_Z_SPACING
    rho, epoch = c["rho"], c["epoch"]
    traces = []

    # Logic shell
    coords, vals = logic_shell_points(logic_maps[idx])
    coords *= point_scale
    mask = vals != 0
    sc, sv = coords[mask], vals[mask]
    colors = [plotly_rgba(1.0,0,1,a=0.15*opacity_scale) if v==1
              else plotly_rgba(0.0,0,1,a=0.15*opacity_scale) for v in sv]
    traces.append(go.Scatter3d(x=sc[:,0], y=sc[:,1], z=sc[:,2]+z_off,
        mode="markers", marker=dict(size=1.8, color=colors),
        showlegend=False, hoverinfo="skip", legendgroup=f"e{epoch}"))

    # Gershgorin wireframes (limited for perf)
    u = np.linspace(0, 2*np.pi, 16)
    v = np.linspace(0, np.pi, 8)
    for si, (pos, rad) in enumerate(c["spheres"][:16]):
        pos, rad = pos*point_scale, rad*point_scale
        for vi in range(len(v)):
            cx = pos[0]+rad*np.sin(v[vi])*np.cos(u)
            cy = pos[1]+rad*np.sin(v[vi])*np.sin(u)
            cz = np.full_like(cx, pos[2]+rad*np.cos(v[vi])+z_off)
            traces.append(go.Scatter3d(x=cx, y=cy, z=cz, mode="lines",
                line=dict(width=1, color=plotly_rgba(rad,min_gersh,max_gersh,a=0.08*opacity_scale)),
                showlegend=False, hoverinfo="skip", legendgroup=f"e{epoch}"))

    # Eigenvalue cloud
    pts = c["points"]*point_scale
    traces.append(go.Scatter3d(x=pts[:,0], y=pts[:,1], z=pts[:,2]+z_off,
        mode="markers", marker=dict(size=4, color=plotly_rgb(rho,min_rho,max_rho),
        opacity=0.7*opacity_scale),
        name=f"Epoch {epoch} (ρ={rho:.4f})", legendgroup=f"e{epoch}", showlegend=True))

    # Fibers
    fx,fy,fz = [],[],[]
    for s,e in c["fibers"]:
        s2,e2 = s*point_scale, e*point_scale
        fx.extend([s2[0],e2[0],None]); fy.extend([s2[1],e2[1],None])
        fz.extend([s2[2]+z_off,e2[2]+z_off,None])
    traces.append(go.Scatter3d(x=fx, y=fy, z=fz, mode="lines",
        line=dict(width=1.5, color=plotly_rgba(0.7,0,1,a=0.35*opacity_scale)),
        showlegend=False, hoverinfo="skip", legendgroup=f"e{epoch}"))

    # Tensor arrows
    tx,ty,tz = [],[],[]
    for s,e in c["tensor_arrows"]:
        tx.extend([s[0],e[0],None]); ty.extend([s[1],e[1],None])
        tz.extend([s[2]+z_off,e[2]+z_off,None])
    traces.append(go.Scatter3d(x=tx, y=ty, z=tz, mode="lines",
        line=dict(width=2.5, color=plotly_rgba(0.3,0,1,a=0.25*opacity_scale)),
        showlegend=False, hoverinfo="skip", legendgroup=f"e{epoch}"))

    return traces

def make_layout(title="Neural Organ", camera=None):
    if camera is None:
        camera = dict(eye=dict(x=1.8,y=1.8,z=1.2), up=dict(x=0,y=0,z=1))
    return go.Layout(
        title=dict(text=title, font=dict(size=20, color="#e0e0ff", family="Courier New"), x=0.5),
        paper_bgcolor="#0a0a14", plot_bgcolor="#0a0a14", font=dict(color="#c0c0e0"),
        legend=dict(bgcolor="rgba(10,10,20,0.8)", bordercolor="#404080", borderwidth=1,
                    font=dict(size=11, color="#c0c0e0")),
        scene=dict(bgcolor="#0a0a14",
            xaxis=dict(title="Re(λ)", backgroundcolor="#0a0a14", gridcolor="#1a1a3a", color="#8080c0"),
            yaxis=dict(title="Im(λ)", backgroundcolor="#0a0a14", gridcolor="#1a1a3a", color="#8080c0"),
            zaxis=dict(title="Epoch",  backgroundcolor="#0a0a14", gridcolor="#1a1a3a", color="#8080c0"),
            camera=camera, aspectmode="data"),
        margin=dict(l=0,r=0,t=50,b=0), width=RESOLUTION[0], height=RESOLUTION[1])

# ============================================================
# A: OPENCV BLOOM POST-PROCESSING
# ============================================================

def bloom_postprocess(frame_bgr):
    """Apply neon bloom, chromatic aberration, vignette, ACES tone mapping."""
    h, w = frame_bgr.shape[:2]
    img = frame_bgr.astype(np.float32) / 255.0

    # 1) Extract bright regions
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    bright_mask = np.clip((gray - 0.3) / 0.7, 0.0, 1.0)
    bright = img * bright_mask[:, :, np.newaxis]

    # 2) Gaussian blur for bloom
    bloom = cv2.GaussianBlur(bright, (BLOOM_KERNEL, BLOOM_KERNEL), 0)
    img = img + BLOOM_INTENSITY * bloom

    # 3) Chromatic aberration (RGB channel shift)
    s = CHROMATIC_SHIFT
    result = np.zeros_like(img)
    result[:, s:, 2]    = img[:, :-s, 2] if s > 0 else img[:, :, 2]   # R shift right
    result[:, :, 1]     = img[:, :, 1]                                  # G stays
    result[:, :-s, 0]   = img[:, s:, 0] if s > 0 else img[:, :, 0]    # B shift left
    img = result

    # 4) ACES filmic tone mapping
    a, b, c_t, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    img = np.clip((img*(a*img+b)) / (img*(c_t*img+d)+e), 0.0, 1.0)

    # 5) Vignette
    Y, X = np.ogrid[:h, :w]
    cx, cy = w/2, h/2
    dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
    max_dist = np.sqrt(cx**2 + cy**2)
    vignette = 1.0 - VIGNETTE_STRENGTH * (dist / max_dist)**2
    img = img * vignette[:, :, np.newaxis]

    return np.clip(img * 255, 0, 255).astype(np.uint8)

# ============================================================
# MODE A: BLOOM ORBIT MP4
# ============================================================

if ENABLE_BLOOM_MP4:
    print("\n" + "─"*64)
    print("  MODE A: OpenCV Bloom 360° Orbit MP4")
    print("─"*64)

    bloom_dir = OUTPUT_DIR / "_bloom_frames"
    bloom_dir.mkdir(exist_ok=True)

    all_traces = []
    for i in range(len(composites)):
        all_traces.extend(build_epoch_traces(i))

    for f in tqdm(range(BLOOM_FRAMES), desc="  Rendering bloom orbit"):
        angle = (f / BLOOM_FRAMES) * 360.0
        rad = np.radians(angle)
        dist = 2.5
        cam = dict(eye=dict(x=dist*np.cos(rad), y=dist*np.sin(rad),
                            z=0.8+0.3*np.sin(rad*2)), up=dict(x=0,y=0,z=1))
        fig = go.Figure(data=all_traces, layout=make_layout("Neural Organ — Bloom Orbit", camera=cam))
        png_path = bloom_dir / f"frame_{f:04d}.png"
        fig.write_image(str(png_path), width=RESOLUTION[0], height=RESOLUTION[1])

        # Apply bloom post-processing
        raw = cv2.imread(str(png_path))
        processed = bloom_postprocess(raw)
        cv2.imwrite(str(png_path), processed)

    # Assemble MP4
    pngs = sorted(bloom_dir.glob("frame_*.png"))
    out_path = OUTPUT_DIR / "NeuralOrgan_BloomOrbit.mp4"
    with imageio.get_writer(str(out_path), fps=FPS, codec="libx264",
                            quality=8, pixelformat="yuv420p") as w:
        for p in tqdm(pngs, desc="  Encoding MP4"):
            w.append_data(imageio.imread(str(p)))
    for p in pngs: p.unlink()
    bloom_dir.rmdir()
    print(f"  ✓ {out_path.name}")

# ============================================================
# MODE D: INTERACTIVE EPOCH SCRUBBER
# ============================================================

if ENABLE_EPOCH_SCRUBBER:
    print("\n" + "─"*64)
    print("  MODE D: Interactive Epoch Scrubber HTML")
    print("─"*64)

    # Build frames for animation slider
    fig = go.Figure()
    # Add all epoch traces as initial (epoch 1)
    initial_traces = build_epoch_traces(0, z_off=0)
    for t in initial_traces:
        fig.add_trace(t)
    n_traces = len(initial_traces)

    # Build frames for each epoch
    frames = []
    for ei in range(len(composites)):
        frame_traces = build_epoch_traces(ei, z_off=0)
        frame_data = []
        for t in frame_traces:
            frame_data.append(go.Scatter3d(x=t.x, y=t.y, z=t.z,
                mode=t.mode, marker=t.marker, line=t.line,
                showlegend=t.showlegend, hoverinfo=t.hoverinfo))
        frames.append(go.Frame(data=frame_data, name=f"Epoch {ei+1}",
                               traces=list(range(n_traces))))

    fig.frames = frames

    # Slider
    sliders = [dict(
        active=0, pad=dict(b=10, t=50),
        steps=[dict(args=[[f"Epoch {ei+1}"],
                          dict(frame=dict(duration=300, redraw=True), mode="immediate")],
                    label=f"E{ei+1}", method="animate")
               for ei in range(len(composites))],
        currentvalue=dict(prefix="Epoch: ", font=dict(size=14, color="#c0c0ff")),
        font=dict(color="#8080c0"),
        bgcolor="#1a1a3a", activebgcolor="#4040a0", bordercolor="#404080",
    )]

    # Play/Pause buttons
    updatemenus = [dict(type="buttons", showactive=False, x=0.05, y=0.05,
        buttons=[
            dict(label="▶ Play", method="animate",
                 args=[None, dict(frame=dict(duration=800, redraw=True),
                                  fromcurrent=True, mode="immediate")]),
            dict(label="⏸ Pause", method="animate",
                 args=[[None], dict(frame=dict(duration=0, redraw=False),
                                    mode="immediate")]),
        ],
        font=dict(color="#c0c0ff"), bgcolor="#1a1a3a", bordercolor="#404080",
    )]

    layout = make_layout("Neural Organ — Epoch Scrubber")
    layout.update(sliders=sliders, updatemenus=updatemenus)
    fig.update_layout(layout)

    # Add epoch stats annotation
    for ei in range(len(composites)):
        c = composites[ei]
        # Stats will show via title per frame
        pass

    scrub_path = OUTPUT_DIR / "NeuralOrgan_Scrubber.html"
    fig.write_html(str(scrub_path), include_plotlyjs="cdn", auto_play=False)
    print(f"  ✓ {scrub_path.name}")
    try:
        fig.show()
    except Exception:
        pass

# ============================================================
# MODE E: MULTI-ORGAN COMPOSITE
# ============================================================

if ENABLE_MULTI_ORGAN and len(MULTI_ORGAN_DIRS) > 1:
    print("\n" + "─"*64)
    print("  MODE E: Multi-Organ Composite HTML")
    print("─"*64)

    multi_traces = []
    organ_colors = ["plasma", "viridis", "inferno", "magma", "cividis"]

    for run_idx, (run_dir, label) in enumerate(zip(MULTI_ORGAN_DIRS, MULTI_ORGAN_LABELS)):
        run_matrices = load_a_matrices(run_dir)
        if not run_matrices:
            print(f"  ✗ Skipping {label}: no matrices")
            continue

        old_cmap = NEON_COLORMAP
        # Use different colormap per run for visual separation
        run_results = [analyze_matrix(A) for A in run_matrices]
        run_composites = [compute_composite(run_results[i], i+1) for i in range(len(run_results))]

        x_offset = run_idx * 15.0  # separate runs along X
        for ei in range(len(run_composites)):
            traces = build_epoch_traces(ei, z_off=ei*EPOCH_Z_SPACING)
            # Shift all X coords
            for t in traces:
                if t.x is not None:
                    x_arr = np.array(t.x, dtype=float)
                    x_arr[~np.isnan(x_arr)] += x_offset
                    t.x = x_arr.tolist()
            multi_traces.extend(traces)
        print(f"  ✓ {label}: {len(run_composites)} epochs loaded")

    if multi_traces:
        fig = go.Figure(data=multi_traces, layout=make_layout("Multi-Organ Spectral Atlas"))
        multi_path = OUTPUT_DIR / "NeuralOrgan_MultiOrgan.html"
        fig.write_html(str(multi_path), include_plotlyjs="cdn")
        print(f"  ✓ {multi_path.name}")

elif ENABLE_MULTI_ORGAN:
    print("\n  ⚠ Multi-Organ: Add more dirs to MULTI_ORGAN_DIRS to enable")

print("\n[5/6] Building advanced HTML exports...")

# ============================================================
# MODE B: ENHANCED THREE.JS NEON CATHEDRAL
# ============================================================

if ENABLE_THREEJS_CATHEDRAL:
    print("\n" + "─"*64)
    print("  MODE B: Three.js Neon Cathedral HTML")
    print("─"*64)

    last = len(composites) - 1
    c = composites[last]

    geo = json.dumps({
        "eigen_pts": c["points"].tolist(),
        "eigen_color": hex_color(c["rho"], min_rho, max_rho),
        "fibers": [(s.tolist(), e.tolist()) for s,e in c["fibers"]],
        "fiber_color": hex_color(0.7, 0, 1),
        "tensors": [(s.tolist(), e.tolist()) for s,e in c["tensor_arrows"]],
        "tensor_color": hex_color(0.3, 0, 1),
        "spheres": [(p.tolist(), float(r)) for p,r in c["spheres"][:32]],
        "sphere_colors": [hex_color(r, min_gersh, max_gersh) for _,r in c["spheres"][:32]],
        "shell_pts": (lambda co,va: co[va!=0].tolist())(*logic_shell_points(logic_maps[last])),
        "shell_vals": (lambda co,va: va[va!=0].tolist())(*logic_shell_points(logic_maps[last])),
        "pos_color": hex_color(1.0, 0, 1),
        "neg_color": hex_color(0.0, 0, 1),
        "rho": c["rho"], "gersh": c["gersh"], "epoch": c["epoch"],
    })

    html = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Neural Organ — Neon Cathedral</title>
<style>*{margin:0;padding:0}body{background:#050510;overflow:hidden;font-family:'Courier New',monospace}
canvas{display:block}
#hud{position:absolute;top:12px;left:50%;transform:translateX(-50%);color:#b0b0ff;font-size:14px;
text-align:center;text-shadow:0 0 12px #6060ff;pointer-events:none;z-index:10}
#stats{position:absolute;bottom:12px;right:12px;color:#505070;font-size:11px;z-index:10;pointer-events:none}
</style></head><body>
<div id="hud">Neural Organ — Neon Cathedral<br>
<span style="font-size:11px;color:#8080a0">EPOCH_INFO</span></div>
<div id="stats"></div>

<script async src="https://ga.jspm.io/npm:es-module-shims@1.8.0/dist/es-module-shims.js"></script>
<script type="importmap">{"imports":{
"three":"https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
"three/addons/":"https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"}}</script>

<script type="module">
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {EffectComposer} from 'three/addons/postprocessing/EffectComposer.js';
import {RenderPass} from 'three/addons/postprocessing/RenderPass.js';
import {UnrealBloomPass} from 'three/addons/postprocessing/UnrealBloomPass.js';
import {ShaderPass} from 'three/addons/postprocessing/ShaderPass.js';

const G = GEO_JSON_PLACEHOLDER;

// --- Renderer ---
const renderer = new THREE.WebGLRenderer({antialias:true});
renderer.setSize(innerWidth, innerHeight);
renderer.setPixelRatio(devicePixelRatio);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.3;
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x050510);
scene.fog = new THREE.FogExp2(0x050510, 0.03);

const camera = new THREE.PerspectiveCamera(60, innerWidth/innerHeight, 0.1, 200);
camera.position.set(5, 5, 4);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.05;
controls.autoRotate = true;
controls.autoRotateSpeed = 0.6;

// --- Post-processing ---
const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(scene, camera));
const bloom = new UnrealBloomPass(new THREE.Vector2(innerWidth, innerHeight), 1.8, 0.4, 0.8);
composer.addPass(bloom);

// Chromatic Aberration shader
const ChromaShader = {
    uniforms: {tDiffuse:{value:null}, amount:{value:0.003}},
    vertexShader: `varying vec2 vUv; void main(){vUv=uv;gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0);}`,
    fragmentShader: `uniform sampler2D tDiffuse; uniform float amount; varying vec2 vUv;
    void main(){vec2 offset=amount*vec2(1.0,0.0);
    float r=texture2D(tDiffuse,vUv+offset).r;
    float g=texture2D(tDiffuse,vUv).g;
    float b=texture2D(tDiffuse,vUv-offset).b;
    gl_FragColor=vec4(r,g,b,1.0);}`
};
composer.addPass(new ShaderPass(ChromaShader));

// --- Lights ---
scene.add(new THREE.AmbientLight(0x303060, 0.4));
const pLight = new THREE.PointLight(0x8080ff, 2, 50);
pLight.position.set(0, 0, 5);
scene.add(pLight);
const pLight2 = new THREE.PointLight(0xff4080, 1.5, 40);
pLight2.position.set(-3, 2, -2);
scene.add(pLight2);

// --- Eigenvalue spheres ---
const eigenGroup = new THREE.Group();
G.eigen_pts.forEach(pt => {
    const geo = new THREE.SphereGeometry(0.05, 16, 10);
    const mat = new THREE.MeshStandardMaterial({
        color: new THREE.Color(G.eigen_color),
        emissive: new THREE.Color(G.eigen_color),
        emissiveIntensity: 2.5, metalness: 0.3, roughness: 0.4});
    const m = new THREE.Mesh(geo, mat);
    m.position.set(pt[0], pt[1], pt[2]);
    eigenGroup.add(m);
});
scene.add(eigenGroup);

// --- Gershgorin spheres ---
G.spheres.forEach((sph, i) => {
    const geo = new THREE.SphereGeometry(sph[1], 20, 14);
    const mat = new THREE.MeshStandardMaterial({
        color: new THREE.Color(G.sphere_colors[i]||'#4040a0'),
        emissive: new THREE.Color(G.sphere_colors[i]||'#4040a0'),
        emissiveIntensity: 1.0, transparent: true, opacity: 0.05, wireframe: true});
    const m = new THREE.Mesh(geo, mat);
    m.position.set(sph[0][0], sph[0][1], sph[0][2]);
    scene.add(m);
});

// --- Fiber halo ---
const fGeo = new THREE.BufferGeometry();
const fV = [];
G.fibers.forEach(p => {fV.push(p[0][0],p[0][1],p[0][2],p[1][0],p[1][1],p[1][2]);});
fGeo.setAttribute('position', new THREE.Float32BufferAttribute(fV, 3));
scene.add(new THREE.LineSegments(fGeo,
    new THREE.LineBasicMaterial({color:G.fiber_color, transparent:true, opacity:0.5})));

// --- Tensor arrows ---
const tGeo = new THREE.BufferGeometry();
const tV = [];
G.tensors.forEach(p => {tV.push(p[0][0],p[0][1],p[0][2],p[1][0],p[1][1],p[1][2]);});
tGeo.setAttribute('position', new THREE.Float32BufferAttribute(tV, 3));
scene.add(new THREE.LineSegments(tGeo,
    new THREE.LineBasicMaterial({color:G.tensor_color, transparent:true, opacity:0.3})));

// --- Logic shell point cloud ---
const sGeo = new THREE.BufferGeometry();
const sV = [], sC = [];
G.shell_pts.forEach((pt, i) => {
    sV.push(pt[0], pt[1], pt[2]);
    const c = new THREE.Color(G.shell_vals[i]===1 ? G.pos_color : G.neg_color);
    sC.push(c.r, c.g, c.b);
});
sGeo.setAttribute('position', new THREE.Float32BufferAttribute(sV, 3));
sGeo.setAttribute('color', new THREE.Float32BufferAttribute(sC, 3));
scene.add(new THREE.Points(sGeo,
    new THREE.PointsMaterial({size:0.04, vertexColors:true, transparent:true, opacity:0.15, sizeAttenuation:true})));

// --- Particle dust field ---
const dustGeo = new THREE.BufferGeometry();
const dustV = [];
for(let i=0; i<2000; i++){
    dustV.push((Math.random()-0.5)*12, (Math.random()-0.5)*12, (Math.random()-0.5)*12);
}
dustGeo.setAttribute('position', new THREE.Float32BufferAttribute(dustV, 3));
const dustPts = new THREE.Points(dustGeo,
    new THREE.PointsMaterial({size:0.02, color:0x404080, transparent:true, opacity:0.3}));
scene.add(dustPts);

// --- God ray volume (central glow cone) ---
const godRayGeo = new THREE.ConeGeometry(2, 8, 32, 1, true);
const godRayMat = new THREE.MeshBasicMaterial({
    color: 0x2020ff, transparent: true, opacity: 0.015, side: THREE.DoubleSide});
const godRay = new THREE.Mesh(godRayGeo, godRayMat);
godRay.rotation.x = Math.PI / 2;
scene.add(godRay);

// --- Animation ---
let time = 0;
const statsDiv = document.getElementById('stats');
let lastT = performance.now();

function animate(){
    requestAnimationFrame(animate);
    time += 0.01;

    // Pulsate
    const pulse = 1.0 + 0.04 * Math.sin(time * 2.5);
    eigenGroup.scale.set(pulse, pulse, pulse);

    // Light drift
    pLight.position.x = 3*Math.cos(time*0.4);
    pLight.position.y = 3*Math.sin(time*0.4);
    pLight.intensity = 2.0 + 0.6*Math.sin(time*1.8);
    pLight2.position.x = -3*Math.cos(time*0.3);
    pLight2.position.z = 3*Math.sin(time*0.3);

    // Bloom breathing
    bloom.strength = 1.8 + 0.4*Math.sin(time*1.2);

    // Dust rotation
    dustPts.rotation.y += 0.0005;
    dustPts.rotation.x += 0.0002;

    // God ray pulse
    godRay.material.opacity = 0.015 + 0.008*Math.sin(time*1.5);
    godRay.rotation.z += 0.002;

    controls.update();
    composer.render();

    const now = performance.now();
    statsDiv.textContent = `${Math.round(1000/(now-lastT))} FPS`;
    lastT = now;
}
animate();

addEventListener('resize', () => {
    camera.aspect = innerWidth/innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(innerWidth, innerHeight);
    composer.setSize(innerWidth, innerHeight);
});
</script></body></html>"""

    html = html.replace("EPOCH_INFO", f"Epoch {c['epoch']} | ρ = {c['rho']:.6f} | G = {c['gersh']:.6f}")
    html = html.replace("GEO_JSON_PLACEHOLDER", geo)

    cathedral_path = OUTPUT_DIR / "NeuralOrgan_Cathedral.html"
    with open(cathedral_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  ✓ {cathedral_path.name}")

# ============================================================
# MODE C: BLENDER CYCLES GPU EXPORT
# ============================================================

if ENABLE_BLENDER_EXPORT:
    print("\n" + "─"*64)
    print("  MODE C: Blender Cycles GPU Script Export")
    print("─"*64)

    last = len(composites) - 1
    c = composites[last]

    blender_script = f'''"""
Blender Cycles GPU — Neural Organ Cathedral
Run: blender --background --python NeuralOrgan_Blender.py
"""
import bpy
import math
import mathutils
import json

# ---- Clear scene ----
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# ---- Render settings ----
scene = bpy.context.scene
scene.render.engine = 'CYCLES'
scene.cycles.device = 'GPU'
scene.render.resolution_x = {RESOLUTION[0]}
scene.render.resolution_y = {RESOLUTION[1]}
scene.cycles.samples = 256
scene.cycles.use_denoising = True
scene.render.film_transparent = False
scene.cycles.use_motion_blur = True
scene.cycles.motion_blur_shutter = 0.5
scene.frame_start = 1
scene.frame_end = 120

# ---- World: volumetric atmosphere ----
world = bpy.data.worlds['World']
world.use_nodes = True
wn = world.node_tree.nodes
wn.clear()
bg = wn.new('ShaderNodeBackground')
bg.inputs['Color'].default_value = (0.02, 0.02, 0.05, 1.0)
bg.inputs['Strength'].default_value = 0.1
vol_scatter = wn.new('ShaderNodeVolumeScatter')
vol_scatter.inputs['Color'].default_value = (0.1, 0.1, 0.3, 1.0)
vol_scatter.inputs['Density'].default_value = 0.02
out = wn.new('ShaderNodeOutputWorld')
world.node_tree.links.new(bg.outputs['Background'], out.inputs['Surface'])
world.node_tree.links.new(vol_scatter.outputs['Volume'], out.inputs['Volume'])

# ---- Geometry data (pure Python — no numpy dependency) ----
eigen_pts = {json.dumps([[round(float(x), 8) for x in pt] for pt in c["points"]])}
eigen_color = {json.dumps([round(float(x), 6) for x in _cmap_rgb(c["rho"], min_rho, max_rho)])}
fibers = {json.dumps([([round(float(x), 8) for x in s], [round(float(x), 8) for x in e]) for s,e in c["fibers"][:500]])}
spheres = {json.dumps([([round(float(x), 8) for x in p], round(float(r), 8)) for p,r in c["spheres"][:32]])}

# ---- Helper: create emissive material ----
def make_emissive_mat(name, rgb, strength=5.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    emission = nodes.new('ShaderNodeEmission')
    emission.inputs['Color'].default_value = (*rgb, 1.0)
    emission.inputs['Strength'].default_value = strength
    output = nodes.new('ShaderNodeOutputMaterial')
    mat.node_tree.links.new(emission.outputs['Emission'], output.inputs['Surface'])
    return mat

# ---- Eigenvalue spheres ----
eigen_mat = make_emissive_mat("EigenMat", eigen_color, strength=8.0)
for i, pt in enumerate(eigen_pts):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.04, location=pt, segments=12, ring_count=8)
    obj = bpy.context.active_object
    obj.name = f"Eigen_{{i}}"
    obj.data.materials.append(eigen_mat)

# ---- Gershgorin spheres (wireframe) ----
for i, (pos, rad) in enumerate(spheres):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=rad, location=pos, segments=16, ring_count=12)
    obj = bpy.context.active_object
    obj.name = f"Gersh_{{i}}"
    mat = make_emissive_mat(f"GershMat_{{i}}", (0.3, 0.3, 0.8), strength=2.0)
    obj.data.materials.append(mat)
    mod = obj.modifiers.new("Wire", 'WIREFRAME')
    mod.thickness = 0.003

# ---- Fibers as curves ----
fiber_mat = make_emissive_mat("FiberMat", (0.8, 0.4, 1.0), strength=4.0)
for i, (start, end) in enumerate(fibers):
    curve = bpy.data.curves.new(f"Fiber_{{i}}", 'CURVE')
    curve.dimensions = '3D'
    curve.bevel_depth = 0.003
    spline = curve.splines.new('POLY')
    spline.points.add(1)
    spline.points[0].co = (*start, 1)
    spline.points[1].co = (*end, 1)
    obj = bpy.data.objects.new(f"Fiber_{{i}}", curve)
    obj.data.materials.append(fiber_mat)
    bpy.context.collection.objects.link(obj)

# ---- Camera with orbit path ----
bpy.ops.curve.primitive_bezier_circle_add(radius=6, location=(0, 0, 2))
circle = bpy.context.active_object
circle.name = "CameraPath"

bpy.ops.object.camera_add(location=(6, 0, 2))
cam = bpy.context.active_object
cam.name = "OrbitalCamera"
scene.camera = cam

# Track-to constraint
track = cam.constraints.new('TRACK_TO')
bpy.ops.object.empty_add(location=(0, 0, 0))
target = bpy.context.active_object
target.name = "CameraTarget"
track.target = target
track.track_axis = 'TRACK_NEGATIVE_Z'
track.up_axis = 'UP_Y'

# Follow path
follow = cam.constraints.new('FOLLOW_PATH')
follow.target = circle
follow.use_curve_follow = True

# Animate the path
circle.data.path_duration = 120
circle.data.use_path = True
circle.data.eval_time = 0
circle.data.keyframe_insert('eval_time', frame=1)
circle.data.eval_time = 120
circle.data.keyframe_insert('eval_time', frame=120)

# ---- Output path ----
scene.render.filepath = "//NeuralOrgan_Cycles_"
scene.render.image_settings.file_format = 'FFMPEG'
scene.render.ffmpeg.format = 'MPEG4'
scene.render.ffmpeg.codec = 'H264'
scene.render.ffmpeg.constant_rate_factor = 'HIGH'

print("✓ Blender scene ready. Render with: bpy.ops.render.render(animation=True)")
'''

    blender_path = OUTPUT_DIR / "NeuralOrgan_Blender.py"
    with open(blender_path, "w", encoding="utf-8") as f:
        f.write(blender_script)
    print(f"  ✓ {blender_path.name}")
    print(f"    Run: blender --background --python {blender_path}")

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 64)
print("  ✓ Next Frontier Edition complete!")
print("=" * 64)
outputs = []
if ENABLE_BLOOM_MP4:         outputs.append("NeuralOrgan_BloomOrbit.mp4    (A: OpenCV bloom)")
if ENABLE_THREEJS_CATHEDRAL: outputs.append("NeuralOrgan_Cathedral.html    (B: Three.js)")
if ENABLE_BLENDER_EXPORT:    outputs.append("NeuralOrgan_Blender.py        (C: Blender Cycles)")
if ENABLE_EPOCH_SCRUBBER:    outputs.append("NeuralOrgan_Scrubber.html     (D: Epoch slider)")
if ENABLE_MULTI_ORGAN:       outputs.append("NeuralOrgan_MultiOrgan.html   (E: Multi-organ)")
for o in outputs:
    print(f"  → visualizations/{o}")
print("=" * 64)
