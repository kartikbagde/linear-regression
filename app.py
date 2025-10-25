import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.collections import LineCollection


st.set_page_config(page_title="Matrix Transformation Visualizer", layout="wide")

st.markdown("<h1 style='text-align: center;'>2×2 Matrix-Based Linear Transformation</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Explore how matrices transform the plane, basis, and vectors.</p>", unsafe_allow_html=True)
st.write("---")


col_left, col_right = st.columns([1, 2.5], gap="large")

with col_left:
    st.subheader("Matrix Input")

    c1, c2 = st.columns(2)
    with c1:
        a11 = st.number_input("a₁₁", value=1.0, format="%.3f")
        a21 = st.number_input("a₂₁", value=0.0, format="%.3f")
    with c2:
        a12 = st.number_input("a₁₂", value=0.0, format="%.3f")
        a22 = st.number_input("a₂₂", value=1.0, format="%.3f")

    A = np.array([[a11, a12], [a21, a22]], dtype=float)

    st.markdown("#### Matrix")
    st.latex(r"\begin{bmatrix} %.3f & %.3f \\ %.3f & %.3f \end{bmatrix}" % (a11, a12, a21, a22))

    det = np.linalg.det(A)
    st.metric("Determinant", f"{det:.4f}")

    try:
        eigvals, eigvecs = np.linalg.eig(A)
        eig_text = ", ".join([f"{v:.3f}" for v in eigvals])
        st.write("**Eigenvalues:**", eig_text)
    except Exception:
        st.write("**Eigenvalues:** (could not compute)")

    st.write("---")
    st.subheader("Vector Input")
    vec_input = st.text_input("Vectors (comma-separated pairs)", value="1,0; 0,1", help="Example: 1,0; 0,1; 2,1")

    # Parse 
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

    st.write("---")
    grid_extent = st.slider("Grid extent (±)", 2, 20, 5)
    grid_density = st.slider("Grid density", 5, 40, 11)

with col_right:
    # --- Grid Preparation ---
    xs = np.linspace(-grid_extent, grid_extent, grid_density)
    ys = np.linspace(-grid_extent, grid_extent, grid_density)

    def make_grid_lines(xs, ys):
        vlines = [np.column_stack((np.full_like(ys, x), ys)) for x in xs]
        hlines = [np.column_stack((xs, np.full_like(xs, y))) for y in ys]
        return vlines + hlines

    grid_lines = make_grid_lines(xs, ys)

    def transform_segment(seg, M):
        return (M @ seg.T).T

    trans_lines = [transform_segment(seg, A) for seg in grid_lines]

    # --- Matplotlib  ---
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-grid_extent, grid_extent)
    ax.set_ylim(-grid_extent, grid_extent)

    # Original 
    orig_lc = LineCollection(grid_lines, linewidths=0.6, alpha=0.4, color="gray")
    ax.add_collection(orig_lc)

    # Transformed 
    trans_lc = LineCollection(trans_lines, linewidths=1.2, alpha=0.9, color="black")
    ax.add_collection(trans_lc)

    origin = np.array([0.0, 0.0])
    e1 = np.array([1.0, 0.0])
    e2 = np.array([0.0, 1.0])
    Ae1 = A @ e1
    Ae2 = A @ e2

    def draw_arrow(ax, v, color="k", label=None, lw=2, style="-|>", alpha=1.0):
        ax.add_patch(FancyArrowPatch(posA=(0,0), posB=(v[0], v[1]),
                                     arrowstyle=style, mutation_scale=12,
                                     linewidth=lw, color=color, alpha=alpha))
        if label:
            ax.text(v[0]*1.05, v[1]*1.05, label, fontsize=10)

    draw_arrow(ax, e1, color="gray", label=r"$e_1$")
    draw_arrow(ax, e2, color="gray", label=r"$e_2$")
    draw_arrow(ax, Ae1, color="tab:blue", label=r"$A e_1$")
    draw_arrow(ax, Ae2, color="tab:orange", label=r"$A e_2$")

   
    for i, v in enumerate(vectors):
        Tv = A @ v
        draw_arrow(ax, v, color="green", label=f"v{i+1}")
        draw_arrow(ax, Tv, color="red", label=f"A v{i+1}")

    ax.plot(0, 0, "ko", markersize=3)
    ax.grid(False)
    ax.set_xlabel("x-axis")
    ax.set_ylabel("y-axis")
    ax.set_title("Transformation Visualization", fontsize=13, pad=10)

    st.pyplot(fig, use_container_width=True)

# --- Inverse & Morph Section ---
st.write("---")
col3, col4 = st.columns(2)
with col3:
    if abs(det) < 1e-9:
        st.warning("Matrix is singular (determinant ≈ 0); no inverse.")
    else:
        invA = np.linalg.inv(A)
        st.subheader("Inverse Matrix (A⁻¹)")
        st.latex(r"\begin{bmatrix} %.3f & %.3f \\ %.3f & %.3f \end{bmatrix}" % 
                 (invA[0,0], invA[0,1], invA[1,0], invA[1,1]))

with col4:
    if st.checkbox("Show Morph (Interpolation)"):
        t = st.slider("t (0 = identity, 1 = A)", 0.0, 1.0, 1.0, step=0.01)
        M_t = (1 - t) * np.eye(2) + t * A
        trans_lines_t = [transform_segment(seg, M_t) for seg in grid_lines]
        fig2, ax2 = plt.subplots(figsize=(6,6))
        ax2.set_aspect("equal", adjustable="box")
        ax2.set_xlim(-grid_extent, grid_extent)
        ax2.set_ylim(-grid_extent, grid_extent)
        ax2.add_collection(LineCollection(grid_lines, linewidths=0.6, alpha=0.4))
        ax2.add_collection(LineCollection(trans_lines_t, linewidths=1.0, alpha=0.9))
        ax2.set_title(f"Interpolation (t={t:.2f})")
        ax2.plot(0,0,"ko",markersize=3)
        st.pyplot(fig2, use_container_width=True)

st.caption("🔹 Built to visualize 2×2 linear transformations interactively. Run locally with: `streamlit run app.py`")
