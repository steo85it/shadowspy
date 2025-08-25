# fluxbf/diagnostics.py
from __future__ import annotations
import os
import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt


def _ensure_outdir(outdir: str) -> str:
    os.makedirs(outdir, exist_ok=True)
    return outdir

def to_numpy_vec(v):
    return v.to_array() if hasattr(v, "to_array") else np.asarray(v)

def to_scipy_csr(F):
    """
    Convert Butterfly CSR to SciPy CSR using the new backend method.
    Falls back to best-effort paths if that method is absent.
    """
    try:
        return F.to_scipy_csr()
    except AttributeError:
        pass  # fall back below

    # Fallbacks, just in case
    for name in ("to_scipy", "to_csr", "to_scipy_sparse"):
        if hasattr(F, name):
            A = getattr(F, name)()
            return A

    # Raw buffers path (if wrapper exposes them as attributes)
    import scipy.sparse as sp
    data   = getattr(F, "data", None)
    indices= getattr(F, "indices", None)
    indptr = getattr(F, "indptr", None)
    shape  = getattr(F, "shape", None)
    if data is not None and indices is not None and indptr is not None and shape is not None:
        return sp.csr_matrix((np.asarray(data), np.asarray(indices), np.asarray(indptr)),
                             shape=tuple(shape))

    raise RuntimeError("No conversion path to SciPy CSR found for Butterfly matrix")

def plot_spy(A, out_png: str, title: str):
    import scipy.sparse as sp
    if not sp.isspmatrix(A): A = sp.csr_matrix(A)
    plt.figure(figsize=(6,6), dpi=150)
    plt.spy(A, markersize=0.5)
    plt.title(title)
    plt.xlabel("col"); plt.ylabel("row")
    plt.tight_layout()
    plt.savefig(out_png); plt.close()

def plot_hist_nnz_per_row(A, out_png: str, title: str):
    import scipy.sparse as sp
    if not sp.isspmatrix(A): A = sp.csr_matrix(A)
    nnz_row = np.diff(A.indptr)
    plt.figure(figsize=(6,4), dpi=150)
    plt.hist(nnz_row, bins=40)
    plt.title(title); plt.xlabel("nnz / row"); plt.ylabel("count")
    plt.tight_layout(); plt.savefig(out_png); plt.close()

def plot_hist_values(A, out_png: str, title: str):
    import scipy.sparse as sp
    if not sp.isspmatrix(A): A = sp.csr_matrix(A)
    vals = np.abs(A.data)
    vals = vals[np.isfinite(vals) & (vals>0)]
    if vals.size == 0: vals = np.array([1e-300])
    plt.figure(figsize=(6,4), dpi=150)
    plt.hist(vals, bins=60, log=True)
    plt.title(title); plt.xlabel("|value| (log bins)"); plt.ylabel("count (log)")
    plt.tight_layout(); plt.savefig(out_png); plt.close()

def plot_vector(y, out_png: str, title: str):
    y = to_numpy_vec(y)
    x = np.arange(y.size)
    plt.figure(figsize=(7,3), dpi=150)
    plt.plot(x, y, lw=0.7)
    plt.title(title); plt.xlabel("index"); plt.ylabel("value")
    plt.tight_layout(); plt.savefig(out_png); plt.close()

def plot_vector_semilogy(y, out_png: str, title: str):
    y = np.abs(to_numpy_vec(y))
    y = np.maximum(y, 1e-300)
    x = np.arange(y.size)
    plt.figure(figsize=(7,3), dpi=150)
    plt.semilogy(x, y, lw=0.7)
    plt.title(title); plt.xlabel("index"); plt.ylabel("|value|")
    plt.tight_layout(); plt.savefig(out_png); plt.close()

def plot_scatter_compare(y_ref, y_test, out_png: str, title: str):
    a = to_numpy_vec(y_ref).ravel()
    b = to_numpy_vec(y_test).ravel()
    n = min(a.size, b.size)
    a = a[:n]; b = b[:n]
    eps = 1e-16
    lim = max(np.max(np.abs(a)), np.max(np.abs(b))) + eps
    plt.figure(figsize=(4,4), dpi=150)
    plt.scatter(a, b, s=2, alpha=0.5)
    plt.plot([-lim, lim], [-lim, lim], 'k--', lw=0.8)
    plt.title(title); plt.xlabel("baseline"); plt.ylabel("bf")
    plt.tight_layout(); plt.savefig(out_png); plt.close()

def dump_basic_stats(A):
    import scipy.sparse as sp
    if not sp.isspmatrix(A): A = sp.csr_matrix(A)
    m, n = A.shape
    nnz_row = (A.indptr[1:] - A.indptr[:-1])
    return {
        "shape": (int(m), int(n)),
        "nnz": int(A.nnz),
        "density": float(A.nnz) / float(m*n) if m*n else float('nan'),
        "nnz_minmax": (int(nnz_row.min()), int(nnz_row.max())),
    }

# ---------- NEW: visibility diagnostics (centroid scatter) ----------

def face_centroids(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    """
    V: (nV,3) float
    F: (nF,3) int
    returns (nF,3) centroids
    """
    return (V[F[:,0]] + V[F[:,1]] + V[F[:,2]]) / 3.0

def visible_cols_from_row(A, row: int, thresh: float | None = None) -> np.ndarray:
    """
    From CSR row 'row', return column indices with nonzero entries.
    If thresh is given, filter by |value| >= thresh.
    """
    import scipy.sparse as sp
    if not sp.isspmatrix(A): A = sp.csr_matrix(A)
    start, end = A.indptr[row], A.indptr[row+1]
    cols = A.indices[start:end]
    if thresh is None:
        return cols
    vals = A.data[start:end]
    return cols[np.abs(vals) >= thresh]

def plot_visible_faces_centroids(V, F, A, row_idx: int, out_png: str,
                                 title: str = "Visible faces (centroids)",
                                 draw_lines: bool = False,
                                 thresh: float | None = None):
    """
    Scatter centroids, highlighting the source row and its visible columns.
    """
    C = face_centroids(V, F)
    vis = visible_cols_from_row(A, row_idx, thresh=thresh)

    plt.figure(figsize=(6,6), dpi=150)
    # all centroids in light gray
    plt.scatter(C[:,0], C[:,1], s=6, c="#cccccc", alpha=0.6, label="all")
    # visible in orange
    if vis.size:
        plt.scatter(C[vis,0], C[vis,1], s=10, c="#ff7f00", alpha=0.9, label=f"visible (n={vis.size})")
        if draw_lines:
            # simple 2D rays from source to each visible centroid
            p0 = C[row_idx, :2]
            for j in vis:
                p1 = C[j, :2]
                plt.plot([p0[0], p1[0]], [p0[1], p1[1]], lw=0.3, c="#999999", alpha=0.5)
    # source face in blue
    plt.scatter([C[row_idx,0]], [C[row_idx,1]], s=30, c="#1f77b4", label=f"row {row_idx}")

    plt.title(title)
    plt.axis("equal")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png); plt.close()

# ---------- 3D visibility plots ----------

def _visible_cols_from_row(A, row: int, thresh=None):
    import scipy.sparse as sp
    if not sp.isspmatrix(A):
        A = sp.csr_matrix(A)
    start, end = A.indptr[row], A.indptr[row+1]
    cols = A.indices[start:end]
    if thresh is None:
        return cols
    vals = A.data[start:end]
    return cols[np.abs(vals) >= float(thresh)]

def plot_visible_faces_3d_mpl(V, F, A, row_idx: int, out_png: str,
                              title="Visible faces (3D, mpl)",
                              draw_lines=False, thresh=None,
                              elev=28, azim=-60, figsize=(8,6)):
    """Matplotlib 3D trisurf with visible faces highlighted."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    vis = _visible_cols_from_row(A, row_idx, thresh=thresh)
    tris = V[F]  # (nF, 3, 3)
    nF = F.shape[0]

    # RGBA colors per face
    col = np.tile(np.array([0.80,0.80,0.80,0.35]), (nF,1))
    col[vis] = np.array([1.00,0.55,0.00,0.95])   # visible: orange
    col[row_idx] = np.array([0.10,0.35,0.95,1.0])# source: blue

    fig = plt.figure(figsize=figsize, dpi=150)
    ax = fig.add_subplot(111, projection='3d')
    coll = Poly3DCollection(tris, facecolors=col, edgecolor=(0,0,0,0.15), linewidths=0.2)
    ax.add_collection3d(coll)

    # autoscale
    mins = V.min(axis=0); maxs = V.max(axis=0)
    cen  = 0.5*(mins+maxs); rad = 0.55*np.max(maxs-mins)
    ax.set_xlim(cen[0]-rad, cen[0]+rad)
    ax.set_ylim(cen[1]-rad, cen[1]+rad)
    ax.set_zlim(cen[2]-rad, cen[2]+rad)
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title)

    if draw_lines and vis.size:
        C = (V[F[:,0]] + V[F[:,1]] + V[F[:,2]])/3.0
        p0 = C[row_idx]
        for j in vis[:2000]:  # safety cap
            p1 = C[j]
            ax.plot([p0[0],p1[0]],[p0[1],p1[1]],[p0[2],p1[2]], lw=0.3, c="#999999", alpha=0.5)

    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
    return vis

def plot_visible_faces_3d_pyvista(V, F, A, row_idx: int, out_png: str,
                                  title="Visible faces (3D, PyVista)",
                                  draw_lines=False, thresh=None,
                                  cpos="iso", window_size=(1024,768)):
    """PyVista 3D rendering with categorical face coloring; off-screen screenshot."""
    try:
        import pyvista as pv
    except Exception as e:
        raise RuntimeError(f"PyVista not available: {e}")

    vis = _visible_cols_from_row(A, row_idx, thresh=thresh)

    # PyVista expects faces as [3, i0, i1, i2, 3, ...] flat array
    faces = np.hstack([np.full((F.shape[0],1), 3, dtype=np.int32), F.astype(np.int32)]).ravel()
    mesh = pv.PolyData(V.astype(np.float64), faces)

    # cell scalars: 0=other, 1=visible, 2=source
    s = np.zeros(F.shape[0], dtype=np.int32)
    s[vis] = 1
    s[row_idx] = 2
    mesh.cell_data["vis"] = s

    # colors: lightgray, orange, royalblue
    cmap = [(0.82,0.82,0.82), (1.0,0.55,0.0), (0.10,0.35,0.95)]

    pv.OFF_SCREEN = True
    plotter = pv.Plotter(off_screen=True, window_size=window_size)
    plotter.add_mesh(mesh, scalars="vis", cmap=cmap, clim=[0,2], show_scalar_bar=False, opacity=0.8)

    if draw_lines and vis.size:
        C = (V[F[:,0]] + V[F[:,1]] + V[F[:,2]])/3.0
        p0 = C[row_idx]
        for j in vis[:5000]:   # cap for performance
            line = pv.Line(p0, C[j])
            plotter.add_mesh(line, color="#888888", line_width=1)

    plotter.add_text(title, font_size=10)
    plotter.background_color = "white"
    plotter.show(screenshot=out_png, cpos=cpos)
    plotter.close()
    return vis

def plot_visible_faces_3d(V, F, A, row_idx: int, out_png: str,
                          backend="auto", **kwargs):
    """
    Dispatch to pyvista (if available) or mpl.
    backend: "auto"|"pyvista"|"mpl"
    Returns: np.ndarray of visible column indices used for highlighting.
    """
    if backend == "pyvista":
        return plot_visible_faces_3d_pyvista(V, F, A, row_idx, out_png, **kwargs)
    if backend == "mpl":
        return plot_visible_faces_3d_mpl(V, F, A, row_idx, out_png, **kwargs)
    # auto
    try:
        import pyvista as _pv  # noqa: F401
        return plot_visible_faces_3d_pyvista(V, F, A, row_idx, out_png, **kwargs)
    except Exception:
        return plot_visible_faces_3d_mpl(V, F, A, row_idx, out_png, **kwargs)

# ---------- 3D visibility with per-face VALUES (|F[i,j]|) ----------

def _row_values(A, row, thresh=None):
    import scipy.sparse as sp
    if not sp.isspmatrix(A):
        A = sp.csr_matrix(A)
    i0, i1 = A.indptr[row], A.indptr[row+1]
    cols = A.indices[i0:i1]
    vals = A.data[i0:i1]
    if thresh is not None:
        m = np.abs(vals) >= float(thresh)
        cols, vals = cols[m], vals[m]
    return cols, vals

def plot_visible_faces_3d_pyvista_values(V, F, A, row_idx: int, out_png: str,
                                         title="|F[i,j]| on visible faces (PyVista)",
                                         log_colors=True, cmap="viridis",
                                         cpos="iso", window_size=(1024,768),
                                         add_lines=False, thresh=None):
    try:
        import pyvista as pv
    except Exception as e:
        raise RuntimeError(f"PyVista not available: {e}")

    cols, vals = _row_values(A, row_idx, thresh=thresh)

    # PyVista PolyData (triangulated surface)
    faces = np.hstack([np.full((F.shape[0],1), 3, dtype=np.int32), F.astype(np.int32)]).ravel()
    mesh = pv.PolyData(V.astype(np.float64), faces)

    # Cell scalars initialized to zero; fill visible faces with |F[i,j]|
    scal = np.zeros(F.shape[0], dtype=float)
    scal[cols] = np.abs(vals)
    if log_colors:
        # Avoid log(0) – shift by tiny epsilon
        eps = max(1e-16, float(scal[scal>0].min()) if np.any(scal>0) else 1e-16)
        scal = np.log10(scal + eps)
    mesh.cell_data["ff"] = scal

    pv.OFF_SCREEN = True
    p = pv.Plotter(off_screen=True, window_size=window_size)
    p.add_mesh(mesh, scalars="ff", cmap=cmap, opacity=0.9, show_scalar_bar=True)
    # Highlight source face outline
    p.add_mesh(mesh.extract_cells([row_idx]), color="#1f77b4", line_width=2, style="wireframe")
    if add_lines and cols.size:
        C = (V[F[:,0]] + V[F[:,1]] + V[F[:,2]])/3.0
        p0 = C[row_idx]
        for j in cols[:4000]:
            p.add_mesh(pv.Line(p0, C[j]), color="#888888", line_width=1)
    p.background_color = "white"
    p.add_text(title, font_size=10)
    p.show(screenshot=out_png, cpos=cpos)
    p.close()
    return cols, vals

# ---------- 3D elevation map (PyVista) ----------

def plot_elevation_pyvista(V, F, out_png: str,
                           title="Elevation (PyVista)", direction=(0,0,1),
                           cmap="terrain", cpos="iso", window_size=(1024,768)):
    try:
        import pyvista as pv
    except Exception as e:
        raise RuntimeError(f"PyVista not available: {e}")
    faces = np.hstack([np.full((F.shape[0],1), 3, dtype=np.int32), F.astype(np.int32)]).ravel()
    mesh = pv.PolyData(V.astype(np.float64), faces)
    # Elevation scalar along 'direction'
    direction = np.asarray(direction, dtype=float)
    direction = direction / (np.linalg.norm(direction) + 1e-16)
    # Project vertex coords onto the direction
    z = V @ direction
    # Convert to per-cell mean elevation
    elev_cell = (z[F[:,0]] + z[F[:,1]] + z[F[:,2]]) / 3.0
    mesh.cell_data["elev"] = elev_cell

    pv.OFF_SCREEN = True
    p = pv.Plotter(off_screen=True, window_size=window_size)
    p.add_mesh(mesh, scalars="elev", cmap=cmap, show_scalar_bar=True, opacity=0.95)
    p.background_color = "white"
    p.add_text(title, font_size=10)
    p.show(screenshot=out_png, cpos=cpos)
    p.close()
