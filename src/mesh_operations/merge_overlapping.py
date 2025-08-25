import time

import meshio
from matplotlib import pyplot as plt
import numpy as np

from mesh_operations.boundary_finding import triangulate_and_find_boundaries_vectorized
from mesh_operations.mesh_generation import generate_terrain_mesh, stack_meshes
from mesh_operations.mesh_utils import remove_inner_from_outer, filter_faces, load_mesh
from mesh_operations.boundary_finding import _boundary_loops_from_faces, _pick_outer_and_inner_loop

def _grid_outer_loop_from_shape(mesh):
    """
    Return the outer boundary vertex loop for a rectangular grid mesh.
    If mesh['shape'] is missing, return None (fallback to generic path).
    """
    if "shape" not in mesh:
        return None
    nrows = int(mesh["shape"][0]) + 1
    ncols = int(mesh["shape"][1]) + 1
    idx_grid = np.arange(nrows * ncols, dtype=np.int32).reshape(nrows, ncols)
    top    = idx_grid[0, :]
    right  = idx_grid[1:-1, -1]
    bottom = idx_grid[-1, ::-1]
    left   = idx_grid[-2:0:-1, 0]
    loop = np.concatenate([top, right, bottom, left])
    return loop


def _triangulate_bridge_earcut(outer_loop_ids, inner_loop_ids, V):
    """
    Triangulate the annulus defined by 'outer' (CCW) with one hole='inner' (CW).
    Works with mapbox_earcut (Nx2 vertices + ring END indices).
    Falls back to the pure-python 'earcut' package if mapbox_earcut isn't present.
    """
    import numpy as np

    # Concatenate the XY of outer then inner loop
    outer_xy = V[outer_loop_ids, :2]
    inner_xy = V[inner_loop_ids, :2]
    verts = np.vstack([outer_xy, inner_xy]).astype(np.float64, copy=False)

    # Guards: both rings must have >= 3 unique points for a proper annulus
    if outer_xy.shape[0] < 3 or inner_xy.shape[0] < 3:
        raise ValueError("Transition triangulation: need >=3 vertices on both outer and inner loops.")

    try:
        import mapbox_earcut as earcut
        # RING **END** indices (outer ends at len(outer), hole ends at total)
        rings = np.asarray([len(outer_xy), len(outer_xy) + len(inner_xy)], dtype=np.uint32)
        # mapbox_earcut returns indices into verts
        tris = earcut.triangulate_float64(verts, rings)
        return tris.astype(np.int32, copy=False)

    except ImportError:
        # Fallback to the pure-Python 'earcut' API (different signature!)
        # It expects a **flat** coords array and hole **START** indices.
        import earcut as earcut_py  # pip install earcut
        coords_flat = verts.reshape(-1)  # [x0,y0, x1,y1, ...]
        hole_starts = [len(outer_xy)]
        idx = earcut_py.earcut(coords_flat, hole_starts, 2)  # returns flat indices
        return np.asarray(idx, dtype=np.int32).reshape(-1, 3)

def merge_inout(inner_mesh, outer_mesh, output_path, debug=False):

    # get inner and outer mesh vertices
    x_in, y_in, z_in = inner_mesh['V'].T
    x_out, y_out, z_out = outer_mesh['V'].T
    # get inner box
    bbox_in = [np.min(x_in), np.max(x_in), np.min(y_in), np.max(y_in)]

    # removing inner part from the outer mesh
    vertices_out = np.vstack((x_out, y_out, z_out)).T
    vertices_with_hole, mask = remove_inner_from_outer(np.vstack((x_out, y_out, z_out)).T, bbox_in)

    faces = filter_faces(mask, outer_mesh['F'])

    start = time.time()
    # Trimmed OUTER mesh boundaries: ask for the loop that surrounds the inner bbox
    # faces_out, _, inner_vertices_outer = triangulate_and_find_boundaries_vectorized(
    #     faces, vertices_out, inner_bbox=bbox_in
    # )
    # seam padding ~ half a cell size (or use a small absolute value).
    # Estimate grid spacing from medians of dx, dy on outer mesh:
    dx = np.median(np.diff(np.unique(vertices_out[:, 0])))
    dy = np.median(np.diff(np.unique(vertices_out[:, 1])))
    pad = 1.5 * np.sqrt(dx * dx + dy * dy)  # generous but still small band

    # find loops using seam prefilter + fast 1-D key unique
    loops = _boundary_loops_from_faces(
        faces=faces,
        n_verts=vertices_out.shape[0],
        vertices_xy=vertices_out[:, :2],
        seam_bbox=(bbox_in[0], bbox_in[1], bbox_in[2], bbox_in[3], pad)
    )

    # choose the loop whose centroid lies inside the inner bbox
    _, inner_vertices_outer = _pick_outer_and_inner_loop(loops, vertices_out[:, :2], inner_bbox=bbox_in)
    if inner_vertices_outer is None or inner_vertices_outer.size == 0:
        # fallback: no prefilter (extremely rare; expand pad)
        loops = _boundary_loops_from_faces(faces, n_verts=vertices_out.shape[0])
        _, inner_vertices_outer = _pick_outer_and_inner_loop(loops, vertices_out[:, :2], inner_bbox=bbox_in)
    faces = filter_faces(mask, outer_mesh['F'])
    faces_out = faces  # <-- add this

    # INNER mesh boundaries: just get the outer loop
    vertices_in = np.vstack((x_in, y_in, z_in)).T
    loop_inner = _grid_outer_loop_from_shape(inner_mesh)
    if loop_inner is None:
        # generic fallback when 'shape' is not present (e.g., VTK-loaded mesh)
        _, loop_inner, _ = triangulate_and_find_boundaries_vectorized(inner_mesh['F'], vertices_in)
    outer_vertices_inner = loop_inner.astype(np.int32, copy=False)
    faces_in = inner_mesh['F']
    print(f"Found boundaries after {round(time.time()-start,2)} seconds")

    # Print results
    if debug:
        print("Outer Boundary Vertices:", outer_vertices_inner)
        print("Inner Boundary Vertices:", inner_vertices_outer)

    # Build transition vertex cloud (no duplicates removed here; optional dedupe below)
    transition_vertices = np.vstack([
        vertices_in[outer_vertices_inner],
        vertices_out[inner_vertices_outer],
    ]).astype(np.float64, copy=False)

    start = time.time()
    # Triangulate annulus (outer with a hole=inner)
    transition_faces = _triangulate_bridge_earcut(
        np.arange(len(outer_vertices_inner), dtype=np.int32),
        np.arange(len(outer_vertices_inner), len(outer_vertices_inner) + len(inner_vertices_outer), dtype=np.int32),
        transition_vertices
    )
    transition_vertices = transition_vertices.astype(np.float32, copy=False) # float64 only needed by earcut
    print(f"Triangulated transition after {round(time.time()-start,2)} seconds")

    # Plot
    if debug:
        plt.triplot(vertices_out[:, 0], vertices_out[:, 1], faces_out)
        plt.triplot(vertices_in[:, 0], vertices_in[:, 1], faces_in)
        plt.triplot(transition_vertices[:, 0], transition_vertices[:, 1], transition_faces)
        plt.plot(vertices_in[outer_vertices_inner, 0], vertices_in[outer_vertices_inner, 1], 'ro', label='Outer Boundary')
        plt.plot(vertices_out[inner_vertices_outer, 0], vertices_out[inner_vertices_outer, 1], 'go', label='Inner Boundary')
        plt.legend()
        plt.show()

    start = time.time()
    # Stack them
    combined_vertices, combined_faces = stack_meshes([
        (vertices_in, faces_in),
        (transition_vertices, transition_faces),
        (vertices_out, faces_out)
    ])
    print(f"Stacked after {round(time.time()-start,2)} seconds")

    labels_dict = {
                    'inner': len(faces_in),
                    'transition': len(transition_faces),
                    'outer': len(faces_out),
                    'total': len(combined_faces)
                }
    print(labels_dict)

    start = time.time()
    # Write the combined mesh to a file
    # Create mesh objects using meshio and write to VTK files
    final_stacked_mesh = meshio.Mesh(points=combined_vertices,
                                     cells=[("triangle", combined_faces)])
    # Write to VTK files
    final_stacked_mesh.write(output_path)
    print(f"Wrote to VTK after {round(time.time()-start,2)} seconds")

    if debug:
        plt.triplot(combined_vertices[:, 0], combined_vertices[:, 1], combined_faces)
        plt.legend()
        plt.show()

    return output_path, labels_dict

if __name__ == '__main__':

    debug = False

    if debug:
        # Parameters for the inner (high-res) and outer (low-res) meshes
        bbox_in = [-500, 500, -500, 500]
        dx_in = 5
        bbox_out = [-2500, 2500, -2500, 2500]
        dx_out = 40

        # Generate terrain meshes
        x_in, y_in, z_in = generate_terrain_mesh((bbox_in[0], bbox_in[1]), (bbox_in[2], bbox_in[3]), dx_in)
        vert_in = np.vstack((x_in, y_in, z_in)).T
        inner_mesh = {'V': vert_in, 'F': Delaunay(vert_in).simplices}
        x_out, y_out, z_out = generate_terrain_mesh((bbox_out[0], bbox_out[1]), (bbox_out[2], bbox_out[3]), dx_out)
        vert_out = np.vstack((x_out, y_out, z_out)).T
        outer_mesh = {'V': vert_out, 'F': Delaunay(vert_out).simplices}

    else:
        inner_mesh = load_mesh('/home/sberton2/Lavoro/code/shadowspy/examples/aux/IM05_GLDELEV_001_st.vtk')
        outer_mesh = load_mesh('/home/sberton2/Lavoro/code/shadowspy/examples/aux/IM1_ldem_large_st.vtk')

    output_path = 'final_stacked.vtk'
    print(merge_inout(inner_mesh, outer_mesh, output_path, debug=debug))
