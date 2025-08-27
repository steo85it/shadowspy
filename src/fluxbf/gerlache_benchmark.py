#!/usr/bin/env python3
from __future__ import annotations
import sys, time, os
from pathlib import Path
import numpy as np

from .mesh import TriMesh
from .ops import view_factor_csr, mv
from .diagnostics import (
    to_scipy_csr, dump_basic_stats,
    plot_spy, plot_hist_nnz_per_row, plot_hist_values,
    plot_vector, plot_vector_semilogy, plot_scatter_compare,
    plot_visible_faces_centroids, plot_visible_faces_3d,
)

def _timeit(fn, *a, **kw):
    t0 = time.perf_counter()
    out = fn(*a, **kw)
    return out, time.perf_counter() - t0

def _load_obj_numpy(path: Path):
    """Tiny OBJ loader (triangles only: first three indices per 'f' line)."""
    verts, faces = [], []
    with open(path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                _, xs, ys, zs = line.strip().split()[:4]
                verts.append((float(xs), float(ys), float(zs)))
            elif line.startswith('f '):
                tri = []
                for p in line.strip().split()[1:4]:
                    tri.append(int(p.split('/')[0]) - 1)
                faces.append(tri)
    V = np.asarray(verts, dtype=np.float64)
    F = np.asarray(faces, dtype=np.uintp)
    return V, F

def _subsample_faces_stride(V: np.ndarray, F: np.ndarray, frac: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Fast deterministic subsampling: keep ~frac of faces by striding.
    Re-index vertices to compact set used by kept faces.
    """
    frac = float(frac)
    if not (0 < frac <= 1.0):
        raise ValueError("frac must be in (0, 1]")
    if frac >= 0.999:
        return V, F

    step = max(1, int(round(1.0/frac)))
    keep = np.arange(0, F.shape[0], step, dtype=np.int64)
    if keep.size == 0:
        keep = np.array([0], dtype=np.int64)

    F_sub = F[keep]
    used = np.unique(F_sub.ravel())
    remap = -np.ones(V.shape[0], dtype=np.int64)
    remap[used] = np.arange(used.size, dtype=np.int64)

    V_sub = V[used]
    F_sub = remap[F_sub].astype(np.uintp, copy=False)
    return V_sub, F_sub

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("obj", type=str, help="path to de_gerlache.obj")
    ap.add_argument("--out", type=str, default="out/gerlache",
                    help="directory to save diagnostics")

    # Quick mode: subsample faces for fast runs
    ap.add_argument("--quick", action="store_true",
                    help="subsample faces for a faster run")
    ap.add_argument("--quick-frac", type=float, default=0.1,
                    help="fraction of faces to keep in quick mode (0<frac<=1)")

    # Visibility diagnostics
    ap.add_argument("--vis-face", type=int, default=0,
                    help="source face index to visualize visibility")
    ap.add_argument("--vis-thresh", type=float, default=None,
                    help="|value| threshold on CSR row when selecting visible faces")
    ap.add_argument("--vis-lines", action="store_true",
                    help="draw lines from source centroid to visibles in 2D")

    # 3D visibility
    ap.add_argument("--vis-3d", action="store_true",
                    help="render a 3D visibility snapshot (PyVista if available, else mpl)")
    ap.add_argument("--vis-3d-backend", type=str, default="auto",
                    choices=["auto","pyvista","mpl"],
                    help="3D backend to use")
    ap.add_argument("--vis-3d-lines", action="store_true",
                    help="draw lines from source to visibles in 3D view")
    ap.add_argument("--vis-3d-cpos", type=str, default="iso",
                    help='PyVista camera preset (e.g., "iso","xy","xz","yz")')

    ap.add_argument("--vis-3d-values", action="store_true",
                    help="color visible faces by |F[i,j]| (PyVista)")
    ap.add_argument("--vis-3d-values-log", action="store_true",
                    help="log10 coloring for |F[i,j]|")
    ap.add_argument("--elev-3d", action="store_true",
                    help="render elevation map (PyVista)")
    ap.add_argument("--elev-dir", type=float, nargs=3, default=(0, 0, 1),
                    help="elevation direction vector (default +Z)")

    args = ap.parse_args()
    outdir = Path(args.out).expanduser().resolve()
    os.makedirs(outdir, exist_ok=True)

    obj = Path(args.obj).expanduser().resolve()
    if not obj.exists():
        print(f"OBJ not found: {obj}")
        sys.exit(2)

    # Load (and optionally subsample) geometry
    V, F = _load_obj_numpy(obj)
    if args.quick:
        V, F = _subsample_faces_stride(V, F, args.quick_frac)
        print(f"[quick] using ~{args.quick_frac:.2%} faces -> V:{len(V)} F:{len(F)}")
    else:
        print(f"[full] V:{len(V)} F:{len(F)}")

    # Build BF mesh + Embree
    mesh = TriMesh.from_numpy(V, F)
    mesh.init_embree()

    # Assemble BF view-factor CSR
    F_bf, t = _timeit(view_factor_csr, mesh)
    # Safe shape reporting
    try:
        bf_shape = F_bf.shape
    except Exception:
        try:
            bf_shape = F_bf.shape()
        except Exception:
            bf_shape = None
    print(f"[bf] assembled view-factor CSR in {t:.2f}s (shape={bf_shape})")

    # Single RHS test: y = F e0
    e0 = np.zeros(mesh.num_faces); e0[0] = 1.0
    y_bf, t_mvp = _timeit(mv, F_bf, e0)
    print(f"[bf] MVP (single RHS) in {t_mvp:.3f}s")

    # Matrix diagnostics
    try:
        A_bf = to_scipy_csr(F_bf)
        stats = dump_basic_stats(A_bf)
        print(f"[bf] CSR stats: {stats}")
        plot_spy(A_bf,              str(outdir/"F_bf_spy.png"),           "BF: sparsity pattern")
        plot_hist_nnz_per_row(A_bf, str(outdir/"F_bf_nnz_row.png"),       "BF: nnz/row")
        plot_hist_values(A_bf,      str(outdir/"F_bf_values.png"),        "BF: |values|")
    except Exception as e:
        print(f"[bf] NOTE: matrix export unavailable: {e}")
        A_bf = None

    # Vector diagnostics
    plot_vector(y_bf,            str(outdir/"y_bf.png"),            "BF: y = F e0")
    plot_vector_semilogy(y_bf,   str(outdir/"y_bf_semilogy.png"),   "BF: |y| semilogy")

    # Visibility diagnostics (2D always; 3D optional)
    if A_bf is not None:
        vis2d_png = outdir / f"visible_row_{args.vis_face}.png"
        plot_visible_faces_centroids(
            V, F, A_bf, row_idx=args.vis_face, out_png=str(vis2d_png),
            title=f"Visible faces from row {args.vis_face}",
            draw_lines=args.vis_lines,
            thresh=args.vis_thresh
        )
        print(f"[diag] wrote {vis2d_png}")

        if args.vis_3d:
            vis3d_png = outdir / f"visible_row_{args.vis_face}_3d.png"
            vis = plot_visible_faces_3d(
                V, F, A_bf, row_idx=args.vis_face, out_png=str(vis3d_png),
                backend=args.vis_3d_backend,
                draw_lines=args.vis_3d_lines,
                thresh=args.vis_thresh,
                title=f"Visible faces (row {args.vis_face})",
                cpos=args.vis_3d_cpos
            )
            print(f"[diag] wrote {vis3d_png} (n_visible={vis.size})")

    # 3D visibility with values
    if args.vis_3d and args.vis_3d_values:
        out_png = os.path.join(outdir, f"visible_row_{args.vis_face}_3d_values.png")
        try:
            from .diagnostics import plot_visible_faces_3d_pyvista_values
            cols, vals = plot_visible_faces_3d_pyvista_values(
                V, F, A_bf, row_idx=args.vis_face, out_png=out_png,
                title=f"|F[i,j]| (row {args.vis_face})",
                log_colors=args.vis_3d_values_log,
                add_lines=args.vis_3d_lines,
                cpos=args.vis_3d_cpos,
                thresh=args.vis_thresh,
            )
            print(f"[diag] wrote {out_png} (n_visible={cols.size})")
        except RuntimeError as e:
            print(f"[diag] PyVista not available for value coloring: {e}")

    # 3D elevation
    if args.elev_3d:
        out_png = os.path.join(outdir, "elevation_3d.png")
        try:
            from .diagnostics import plot_elevation_pyvista
            plot_elevation_pyvista(V, F, out_png,
                                   title="Elevation",
                                   direction=tuple(args.elev_dir),
                                   cpos="iso")
            print(f"[diag] wrote {out_png}")
        except RuntimeError as e:
            print(f"[diag] PyVista not available for elevation: {e}")

    # Optional: flux baseline for comparison (if your legacy env imports)
    try:
        import flux.shape as fshape
        from flux.form_factors import get_form_factor_matrix
        import flux.config as fcfg

        sm = fshape.EmbreeTrimeshShapeModel(V.astype(np.float64), F.astype(np.uintp))
        sm._make_scene()
        eps = getattr(fcfg, 'DEFAULT_EPS', 0.0)
        F_fx, t2 = _timeit(get_form_factor_matrix, sm, None, None, eps)
        print(f"[flux] assembled CSR in {t2:.2f}s (shape={F_fx.shape})")

        y_fx, t_mvp_fx = _timeit(lambda A, x: A @ x, F_fx, e0)
        print(f"[flux] MVP (single RHS) in {t_mvp_fx:.3f}s")

        import scipy.sparse as sp
        A_fx = F_fx.tocsr() if not sp.isspmatrix(F_fx) else F_fx
        plot_spy(A_fx,              str(outdir/"F_flux_spy.png"),         "Flux: sparsity pattern")
        plot_hist_nnz_per_row(A_fx, str(outdir/"F_flux_nnz_row.png"),     "Flux: nnz/row")
        plot_hist_values(A_fx,      str(outdir/"F_flux_values.png"),      "Flux: |values|")
        plot_vector(y_fx,           str(outdir/"y_flux.png"),             "Flux: y = F e0")
        plot_vector_semilogy(y_fx,  str(outdir/"y_flux_semilogy.png"),    "Flux: |y| semilogy")

        if A_bf is not None and y_bf.shape[0] == y_fx.shape[0]:
            rel = np.linalg.norm(y_bf - y_fx) / (np.linalg.norm(y_fx) + 1e-16)
            print(f"[cmp] ||F_bf e0 - F_flux e0|| / ||F_flux e0|| = {rel:.3e}")
            plot_scatter_compare(y_fx, y_bf, str(outdir/"cmp_y_scatter.png"),
                                 "Compare: flux vs bf (y)")
    except Exception as e:
        print("Flux baseline not available:", e)

if __name__ == "__main__":
    main()
