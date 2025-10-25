# app.py
# Streamlit app: 2x2 Matrix Linear Transformation Visualizer
# Requirements: streamlit, numpy, matplotlib

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.collections import LineCollection

st.set_page_config(page_title="Matrix-Based Linear Transformation", layout="wide")

st.title("Matrix-Based Linear Transformation (2 × 2)")
st.write("Enter matrix elements to see how the plane, basis, and vectors transform.")

# Matrix inputs (nice defaults)
col1, col2 = st.columns(2)
with col1:
    a11 = st.number_input("a₁₁ (row 0, col 0)", value=1.0, format="%.3f")
    a12 = st.number_input("a₁₂ (row 0, col 1)", value=0.0, format="%.3f")
with col2:
    a21 = st.number_input("a₂₁ (row 1, col 0)", value=0.0, format="%.3f")
    a22 = st.number_input("a₂₂ (row 1, col 1)", value=1.0, format="%.3f")

A = np.array([[a11, a12],
              [a21, a22]], dtype=float)

st.markdown("### Matrix")
st.latex(r"\begin{bmatrix} %.3f & %.3f \\ %.3f & %.3f \end{bmatrix}" % (a11, a12, a21, a22))

# Derived properties
det = np.linalg.det(A)
st.metric("Determinant", f"{det:.4f}")

try:
    eigvals, eigvecs = np.linalg.eig(A)
    eig_text = ", ".join([f"{v:.3f}" for v in eigvals])
    st.write("Eigenvalues:", eig_text)
except Exception:
    st.write("Eigenvalues: (could not compute)")

# Optional sample vectors
st.write("---")
st.write("Add vectors to see their transformation:")
vec_input = st.text_input("Vectors (comma-separated pairs). Example: 1,0; 0,1; 2,1", value="1,0; 0,1")
# parse vectors
vectors = []
for part in vec_input.split(";"):
    part = part.strip()
    if not part:
        continue
    try:
        x_str, y_str = part.split(",")
        vectors.append(np.array([float(x_str), float(y_str)]))
    except Exception:
        pass

# Plot settings
grid_extent = st.slider("Grid extent (±)", min_value=2, max_value=20, value=5)
grid_density = st.slider("Grid density (lines per axis)", min_value=5, max_value=40, value=11)

# Prepare grid lines
xs = np.linspace(-grid_extent, grid_extent, grid_density)
ys = np.linspace(-grid_extent, grid_extent, grid_density)

# Function to build line segments for a set of parallel grid lines
def make_grid_lines(xs, ys):
    # vertical lines (constant x)
    vlines = [np.column_stack((np.full_like(ys, x), ys)) for x in xs]
    # horizontal lines (constant y)
    hlines = [np.column_stack((xs, np.full_like(xs, y))) for y in ys]
    return vlines + hlines

grid_lines = make_grid_lines(xs, ys)

# transform lines
def transform_segment(seg, M):
    return (M @ seg.T).T

trans_lines = [transform_segment(seg, A) for seg in grid_lines]

# Plot with matplotlib
fig, ax = plt.subplots(figsize=(7,7))
ax.set_aspect("equal", adjustable="box")
ax.set_xlim(-grid_extent, grid_extent)
ax.set_ylim(-grid_extent, grid_extent)

# draw original grid (light)
orig_lc = LineCollection(grid_lines, linewidths=0.6, alpha=0.5)
ax.add_collection(orig_lc)

# draw transformed grid (bolder)
trans_lc = LineCollection(trans_lines, linewidths=1.0, alpha=0.9)
ax.add_collection(trans_lc)

# draw basis vectors before and after
origin = np.array([0.0, 0.0])
e1 = np.array([1.0, 0.0])
e2 = np.array([0.0, 1.0])
Ae1 = A @ e1
Ae2 = A @ e2

def draw_arrow(ax, v, color="k", label=None, lw=2, style="-|>", alpha=1.0):
    ax.add_patch(FancyArrowPatch(posA=(0,0), posB=(v[0], v[1]),
                                 arrowstyle=style, mutation_scale=12,
                                 linewidth=lw, alpha=alpha))
    if label:
        ax.text(v[0]*1.05, v[1]*1.05, label, fontsize=12)

# original basis (dashed)
draw_arrow(ax, e1, color="gray", label=r"$e_1$ (original)", lw=1, alpha=0.6)
draw_arrow(ax, e2, color="gray", label=r"$e_2$ (original)", lw=1, alpha=0.6)

# transformed basis (solid)
draw_arrow(ax, Ae1, color="tab:blue", label=r"$A e_1$", lw=2)
draw_arrow(ax, Ae2, color="tab:orange", label=r"$A e_2$", lw=2)

# draw user vectors
for i, v in enumerate(vectors):
    Tv = A @ v
    draw_arrow(ax, v, color="green", label=f"v{i+1} (orig)")
    draw_arrow(ax, Tv, color="red", label=f"A v{i+1}")

# origin marker
ax.plot(0,0, "ko", markersize=3)
ax.grid(False)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Original grid (light) and transformed grid (solid)\nBlue/Orange = transformed basis")

st.pyplot(fig)

# show matrix inverse if invertible
st.write("---")
if abs(det) < 1e-9:
    st.warning("Matrix is singular (determinant ≈ 0); no inverse.")
else:
    invA = np.linalg.inv(A)
    st.subheader("Inverse matrix (A⁻¹)")
    st.latex(r"\begin{bmatrix} %.3f & %.3f \\ %.3f & %.3f \end{bmatrix}" % (invA[0,0], invA[0,1], invA[1,0], invA[1,1]))

# show composition slider (t from 0 to 1) to morph the transformation (optional)
if st.checkbox("Show continuous morph (interpolate to transformation)"):
    t = st.slider("t (0 = identity, 1 = A)", 0.0, 1.0, 1.0, step=0.01)
    M_t = (1 - t) * np.eye(2) + t * A
    trans_lines_t = [transform_segment(seg, M_t) for seg in grid_lines]
    fig2, ax2 = plt.subplots(figsize=(6,6))
    ax2.set_aspect("equal", adjustable="box")
    ax2.set_xlim(-grid_extent, grid_extent)
    ax2.set_ylim(-grid_extent, grid_extent)
    ax2.add_collection(LineCollection(grid_lines, linewidths=0.6, alpha=0.4))
    ax2.add_collection(LineCollection(trans_lines_t, linewidths=1.0, alpha=0.9))
    ax2.set_title(f"Interpolation t={t:.2f}")
    ax2.plot(0,0,"ko",markersize=3)
    st.pyplot(fig2)

st.write("---")
st.caption("Built to visualize 2×2 linear maps. You can copy this file and run: `pip install streamlit numpy matplotlib` then `streamlit run app.py`.")
