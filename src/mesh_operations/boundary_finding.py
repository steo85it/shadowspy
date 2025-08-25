import time

import numpy as np
from scipy.spatial import Delaunay

def _boundary_loops_from_faces(faces: np.ndarray,
                               n_verts: int,
                               vertices_xy: np.ndarray = None,
                               seam_bbox: tuple = None,
                               faces_vertices: np.ndarray = None) -> list[np.ndarray]:
    """
    Return ordered boundary loops from a face list.

    - If seam_bbox is provided, we first prefilter faces to those intersecting
      a padded band around the bbox (to target seam-only edges).
      seam_bbox: (xmin, xmax, ymin, ymax, pad) in same units as vertices_xy.
    - Uses 1D 64-bit keys for edges (u<v) to count duplicates quickly.
    - Then builds loops by stitching degree-2 boundary vertices.
    """

    F = np.asarray(faces, dtype=np.int32, order='C')
    if seam_bbox is not None and vertices_xy is not None:
        xmin, xmax, ymin, ymax, pad = seam_bbox
        x = vertices_xy[:, 0]; y = vertices_xy[:, 1]
        in_band = (x >= xmin - pad) & (x <= xmax + pad) & (y >= ymin - pad) & (y <= ymax + pad)
        # keep faces that touch the band
        band_mask = in_band[F].any(axis=1)
        F = F[band_mask]

    if F.size == 0:
        return []

    # Build all undirected edges (u<v) for kept faces
    e = np.vstack([F[:, [0,1]], F[:, [1,2]], F[:, [2,0]]]).astype(np.int64, copy=False)
    e.sort(axis=1)
    u = e[:, 0]; v = e[:, 1]
    # 1-D 64-bit keys; base uses n_verts+1 to avoid collisions
    base = np.int64(n_verts + 1)
    keys = u * base + v

    # Sort keys and run-length count
    order = np.argsort(keys, kind='mergesort')  # stable
    sk = keys[order]
    # starts of runs
    run_starts = np.empty(0, dtype=np.int64)
    if sk.size:
        diff = np.concatenate(([True], sk[1:] != sk[:-1]))
        run_starts = np.flatnonzero(diff)
    run_ends = np.append(run_starts[1:], sk.size)
    run_counts = run_ends - run_starts

    # boundary edges = those with count == 1
    bsel = (run_counts == 1)
    if not np.any(bsel):
        return []

    b_keys = sk[run_starts[bsel]]
    bu = (b_keys // base).astype(np.int32, copy=False)
    bv = (b_keys %  base).astype(np.int32, copy=False)
    bedges = np.stack([bu, bv], axis=1)

    # Build adjacency for boundary graph and stitch loops
    # (boundary set is small => Python sets are fine)
    adj = {}
    for a, b in bedges:
        adj.setdefault(int(a), set()).add(int(b))
        adj.setdefault(int(b), set()).add(int(a))

    seen = set()
    loops = []
    for s in list(adj.keys()):
        if s in seen:
            continue
        comp = []
        stack = [s]
        seen.add(s)
        while stack:
            u0 = stack.pop()
            comp.append(u0)
            for v0 in adj[u0]:
                if v0 not in seen:
                    seen.add(v0)
                    stack.append(v0)
        # order
        start = comp[0]
        prev, cur = None, start
        ordered = [start]
        for _ in range(2 * len(comp)):
            nbrs = list(adj[cur])
            nxt = nbrs[0] if (prev is None or nbrs[0] != prev) else (nbrs[1] if len(nbrs) > 1 else None)
            if nxt is None or nxt == ordered[0]:
                break
            ordered.append(nxt)
            prev, cur = cur, nxt
        loops.append(np.asarray(ordered, dtype=np.int32))
    return loops

def _boundary_loops_from_faces_all(faces: np.ndarray) -> list[np.ndarray]:
    # Collect boundary edges (edges that occur exactly once)
    e = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    e = np.sort(e, axis=1)
    u, cnt = np.unique(e, axis=0, return_counts=True)
    bedges = u[cnt == 1]
    if bedges.size == 0:
        return []

    # Build small adjacency map on the boundary graph
    adj = {}
    for a, b in bedges:
        a = int(a); b = int(b)
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)

    # Traverse components and order each into a loop
    seen, loops = set(), []
    for s in list(adj.keys()):
        if s in seen:
            continue
        comp, stack = [], [s]
        seen.add(s)
        while stack:
            u0 = stack.pop()
            comp.append(u0)
            for v in adj[u0]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        # Order by walking neighbors (deg≈2 along proper boundaries)
        start = comp[0]
        prev, cur = None, start
        ordered = [start]
        for _ in range(2 * len(comp)):  # guard
            nbrs = list(adj[cur])
            nxt = nbrs[0] if (prev is None or nbrs[0] != prev) else (nbrs[1] if len(nbrs) > 1 else None)
            if nxt is None or nxt == ordered[0]:
                break
            ordered.append(nxt)
            prev, cur = cur, nxt
        loops.append(np.asarray(ordered, dtype=int))
    return loops

def _pick_outer_and_inner_loop(loops: list[np.ndarray], Vxy: np.ndarray, inner_bbox=None):
    if not loops:
        return None, None
    def poly_area(ids):
        xy = Vxy[ids]
        x, y = xy[:, 0], xy[:, 1]
        return 0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1))
    areas = np.array([abs(poly_area(ids)) for ids in loops])
    outer = loops[int(areas.argmax())]
    inner = None
    if inner_bbox is not None:
        xmin, xmax, ymin, ymax = inner_bbox
        for ids in loops:
            if ids is outer:
                continue
            c = Vxy[ids].mean(axis=0)
            if xmin <= c[0] <= xmax and ymin <= c[1] <= ymax:
                inner = ids
                break
    return outer, inner

def triangulate_and_find_boundaries_vectorized(faces, vertices, *, inner_bbox=None, seam_bbox=None):
    """
    Delaunay-free boundary finder.
    - faces: (N,3)
    - vertices: (M,>=2)
    - inner_bbox: if provided, we pick the loop whose centroid lies inside it as 'inner'
    - seam_bbox: optional (xmin,xmax,ymin,ymax,pad) to prefilter to a band
    """
    Vxy = vertices[:, :2]
    loops = _boundary_loops_from_faces(
        faces=faces,
        n_verts=vertices.shape[0],
        vertices_xy=Vxy,
        seam_bbox=seam_bbox
    )
    outer, inner = _pick_outer_and_inner_loop(loops, Vxy, inner_bbox=inner_bbox)
    outer_boundary_vertices = np.asarray(outer if outer is not None else [], dtype=np.int32)
    inner_boundary_vertices = np.asarray(inner if inner is not None else [], dtype=np.int32)
    return faces, outer_boundary_vertices, inner_boundary_vertices

def triangulate_and_find_boundaries(faces, vertices):
    """
    Perform Delaunay triangulation and identify the boundary vertices of the inner and outer regions.

    Args:
    - vertices: An array of vertices for the mesh.

    Returns:
    - delaunay: The Delaunay triangulation object.
    - outer_boundary_vertices: The vertices on the outer boundary.
    - inner_boundary_vertices: The vertices on the inner boundary.
    """

    # Find boundary edges (edges that appear exactly once)
    edges = {}
    for face in faces:
        for i in range(3):
            edge = tuple(sorted([face[i], face[(i + 1) % 3]]))
            if edge in edges:
                edges[edge] += 1
            else:
                edges[edge] = 1

    boundary_edges = [edge for edge, count in edges.items() if count == 1]
    boundary_vertices = list(set([vertex for edge in boundary_edges for vertex in edge]))

    # Assuming the inner boundary has vertices with higher indices due to the generation method
    delaunay = Delaunay(vertices)
    outer_boundary_vertices = [vertex for vertex in boundary_vertices if vertex in delaunay.convex_hull.flatten()]
    inner_boundary_vertices = list(set(boundary_vertices) - set(outer_boundary_vertices))

    return faces, outer_boundary_vertices, inner_boundary_vertices



def triangulate_and_find_boundaries_slow(faces, vertices):
    """
    Perform Delaunay triangulation and identify the boundary vertices of the inner and outer regions.

    Args:
    - vertices: An array of vertices for the mesh.

    Returns:
    - faces: The input faces of the mesh.
    - outer_boundary_vertices: The vertices on the outer boundary.
    - inner_boundary_vertices: The vertices on the inner boundary.
    """

    # Create edges
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges = np.sort(edges, axis=1)  # Sort edges to make identical ones adjacent
    edges, counts = np.unique(edges, axis=0, return_counts=True)

    # Boundary edges are those that appear exactly once
    boundary_edges = edges[counts == 1]
    boundary_vertices = np.unique(boundary_edges)

    # Delaunay triangulation and identification of convex hull vertices
    delaunay = Delaunay(vertices) # taking 80% of the whole merge_inout
    convex_hull_vertices = np.unique(delaunay.convex_hull)

    # Identify outer and inner boundary vertices based on convex hull
    outer_boundary_vertices = np.intersect1d(boundary_vertices, convex_hull_vertices)
    inner_boundary_vertices = np.setdiff1d(boundary_vertices, outer_boundary_vertices)

    return faces, outer_boundary_vertices, inner_boundary_vertices
