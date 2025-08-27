#!/usr/bin/env python3
from __future__ import annotations
import sys, time
import numpy as np
import scipy.sparse as sp

from .mesh import TriMesh
from .ops import view_factor_csr, mv
from .utils import make_spherical_crater_mesh

import argparse, os
from .diagnostics import (
    to_scipy_csr, plot_spy, plot_hist_nnz_per_row, plot_hist_values,
    plot_vector, plot_vector_semilogy, plot_scatter_compare, dump_basic_stats,
    plot_visible_faces_centroids, plot_visible_faces_3d
)

# Optional: Python-only baseline from flux
def _try_import_flux():
    try:
        import flux.shape as fshape
        from flux.form_factors import get_form_factor_matrix
        import flux.config as fcfg
        return fshape, get_form_factor_matrix, fcfg
    except Exception as e:
        print("WARN: couldn't import flux baseline (skipping):", e)
        return None, None, None

def _timeit(fn, *a, **kw):
    t0 = time.perf_counter()
    out = fn(*a, **kw)
    return out, time.perf_counter() - t0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="out/crater",
                    help="directory to save diagnostics")
    ap.add_argument("--quick", action="store_true", help="use small mesh")
    ap.add_argument("--vis-face", type=int, default=0,
                    help="source face index to visualize visible faces")
    ap.add_argument("--vis-lines", action="store_true",
                    help="draw lines from source centroid to visibles")
    ap.add_argument("--vis-thresh", type=float, default=None,
                    help="optional |value| threshold when selecting visible cols from CSR row")
    ap.add_argument("--vis-3d", action="store_true",
                    help="render 3D visibility for the chosen --vis-face")
    ap.add_argument("--vis-3d-backend", type=str, default="auto",
                    choices=["auto", "pyvista", "mpl"],
                    help="3D backend for visibility plot")
    ap.add_argument("--vis-3d-lines", action="store_true",
                    help="draw lines from source to visible centroids in 3D view")
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
    outdir = args.out
    os.makedirs(outdir, exist_ok=True)

    QUICK = True if args.quick else False
    if QUICK:
        Rc, depth, Rdom = 200.0, 40.0, 600.0
        Nr, Nth = 24, 48
        eps = 1e-6
    else:
        Rc, depth, Rdom = 1000.0, 200.0, 3000.0
        Nr, Nth = 100, 200
        eps = 1e-12
    print(f"[crater] Rc={Rc} depth={depth} Rdom={Rdom} Nr={Nr} Nth={Nth}")

    V, F = make_spherical_crater_mesh(Rc, depth, Rdom, Nr, Nth)
    mesh = TriMesh.from_numpy(V, F)
    mesh.init_embree()

    F_bf, t = _timeit(view_factor_csr, mesh)
    print(f"[bf] assembled view-factor CSR in {t:.2f}s (shape={F_bf.shape})")

    e0 = np.zeros(mesh.num_faces); e0[0] = 1.0
    y_bf, t_mvp = _timeit(mv, F_bf, e0)
    print(f"[bf] MVP (single RHS) in {t_mvp:.3f}s")

    # ---- BF matrix diagnostics ----
    A_bf = to_scipy_csr(F_bf)  # uses the new backend method
    stats = dump_basic_stats(A_bf)
    print(f"[bf] CSR stats: {stats}")

    plot_spy(A_bf,              os.path.join(outdir, "F_bf_spy.png"), "BF: sparsity pattern")
    plot_hist_nnz_per_row(A_bf, os.path.join(outdir, "F_bf_nnz_row.png"), "BF: nnz/row")
    plot_hist_values(A_bf,      os.path.join(outdir, "F_bf_values.png"), "BF: |values|")

    plot_vector(y_bf,            os.path.join(outdir, "y_bf.png"),           "BF: y = F e0")
    plot_vector_semilogy(y_bf,   os.path.join(outdir, "y_bf_semilogy.png"),  "BF: |y| semilogy")

    # 2D centroids visibility already created earlier...
    vis2d_png = os.path.join(outdir, f"visible_row_{args.vis_face}.png")
    plot_visible_faces_centroids(
        V, F, A_bf, row_idx=args.vis_face, out_png=vis2d_png,
        title=f"Visible faces from row {args.vis_face}",
        draw_lines=args.vis_lines,
        thresh=args.vis_thresh
    )
    print(f"[diag] wrote {vis2d_png}")

    # 3D visibility (optional)
    if args.vis_3d:
        vis3d_png = os.path.join(outdir, f"visible_row_{args.vis_face}_3d.png")
        vis = plot_visible_faces_3d(
            V, F, A_bf, row_idx=args.vis_face, out_png=vis3d_png,
            backend=args.vis_3d_backend,
            draw_lines=args.vis_3d_lines,
            thresh=args.vis_thresh,
            title=f"Visible faces (row {args.vis_face})"
            # for PyVista you can also pass window_size=(W,H), cpos=args.vis_3d_cpos
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

    # ---- Optional: flux baseline (if available) ----
    fshape, get_ff, fcfg = _try_import_flux()
    if fshape is not None:
        sm = fshape.EmbreeTrimeshShapeModel(V.astype(np.float64), F.astype(np.uintp))
        sm._make_scene()
        eps = getattr(fcfg, 'DEFAULT_EPS', 0.0)
        F_fx, t2 = _timeit(get_ff, sm, None, None, eps)
        print(f"[flux] assembled CSR in {t2:.2f}s (shape={F_fx.shape})")
        y_fx, t_mvp_fx = _timeit(lambda A, x: A @ x, F_fx, e0)
        print(f"[flux] MVP (single RHS) in {t_mvp_fx:.3f}s")

        import scipy.sparse as sp
        A_fx = F_fx.tocsr() if not sp.isspmatrix(F_fx) else F_fx

        plot_spy(A_fx,              os.path.join(outdir, "F_flux_spy.png"), "Flux: sparsity pattern")
        plot_hist_nnz_per_row(A_fx, os.path.join(outdir, "F_flux_nnz_row.png"), "Flux: nnz/row")
        plot_hist_values(A_fx,      os.path.join(outdir, "F_flux_values.png"), "Flux: |values|")
        plot_vector(y_fx,           os.path.join(outdir, "y_flux.png"), "Flux: y = F e0")
        plot_vector_semilogy(y_fx,  os.path.join(outdir, "y_flux_semilogy.png"), "Flux: |y| semilogy")

        if y_bf.shape[0] == y_fx.shape[0]:
            rel = np.linalg.norm(y_bf - y_fx) / (np.linalg.norm(y_fx) + 1e-16)
            print(f"[cmp] ||F_bf e0 - F_flux e0|| / ||F_flux e0|| = {rel:.3e}")
            plot_scatter_compare(y_fx, y_bf, os.path.join(outdir, "cmp_y_scatter.png"),
                                 "Compare: flux vs bf (y)")

if __name__ == "__main__":
    main()
