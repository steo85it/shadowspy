import logging
import time

import meshio
import numpy as np
import rioxarray as rio

from shadowspy.coord_tools import unproject_stereographic, sph2cart
from mesh_operations.mesh_tools import get_uniform_triangle_mesh
from mesh_operations.mesh_utils import remove_degenerate_faces


def generate_square_with_hole_vertices(outer_square_size=10, hole_size=2, spacing=1):
    """
    Generate vertices for a square region with a square hole in the center.

    Args:
    - outer_square_size: The size of the outer square (edge length).
    - hole_size: The size of the hole in the center (edge length).
    - spacing: The distance between adjacent vertices.

    Returns:
    - vertices: An array of vertices for the combined shape.
    """
    # Generate outer square vertices
    x_outer = np.arange(0, outer_square_size + spacing, spacing)
    y_outer = np.arange(0, outer_square_size + spacing, spacing)
    outer_grid = np.transpose([np.tile(x_outer, len(y_outer)), np.repeat(y_outer, len(x_outer))])

    # Generate hole vertices
    offset = (outer_square_size - hole_size) / 2
    x_hole = np.arange(offset, offset + hole_size + spacing, spacing)
    y_hole = np.arange(offset, offset + hole_size + spacing, spacing)
    hole_grid = np.transpose([np.tile(x_hole, len(y_hole)), np.repeat(y_hole, len(x_hole))])

    # Combine and remove duplicates
    vertices = np.vstack({tuple(row) for row in np.vstack([outer_grid, hole_grid])})

    return vertices

def stack_meshes(meshes):
    """
    Stacks multiple meshes into a single mesh.
    meshes: list of (vertices, faces). Faces may be (N,3), (3N,), or (N,) of 3-lists.
    Returns (combined_vertices, combined_faces) with dtypes float32 / int32.
    """

    if not meshes:
        return (np.empty((0, 3), np.float32), np.empty((0, 3), np.int32))

    def _normalize_faces(F_in):
        F = np.asarray(F_in)
        # Case 1: (N,3)
        if F.ndim == 2 and F.shape[1] == 3:
            return F.astype(np.int32, copy=False)
        # Case 2: flat 1D (3N,)
        if F.ndim == 1 and F.dtype != object:
            if F.size % 3 != 0:
                raise ValueError(f"Faces 1D array length {F.size} not divisible by 3.")
            return F.reshape(-1, 3).astype(np.int32, copy=False)
        # Case 3: object array (N,) of triplets → vstack
        if F.ndim == 1 and F.dtype == object:
            try:
                F2 = np.vstack([np.asarray(row, dtype=np.int64) for row in F])
            except Exception as e:
                raise ValueError("Faces appear to be an object array but could not be "
                                 "stacked into (N,3). Inspect the mesh providing these faces.") from e
            if F2.shape[1] != 3:
                raise ValueError(f"Faces object array stacked to shape {F2.shape}, expected (N,3).")
            return F2.astype(np.int32, copy=False)
        raise ValueError(f"Unsupported faces shape {F.shape} / dtype {F.dtype}")

    def _normalize_vertices(V_in, want_cols=None):
        V = np.asarray(V_in)
        if V.ndim != 2:
            raise ValueError(f"Vertices must be 2D, got shape {V.shape}.")
        if want_cols is not None and V.shape[1] != want_cols:
            raise ValueError(f"Inconsistent vertex dimensions: expected {want_cols}, got {V.shape[1]}.")
        return V.astype(np.float32, copy=False)

    # First normalize and collect; also compute totals
    norm = []
    vcols = None
    nV_total = 0
    nF_total = 0
    for V_in, F_in in meshes:
        V = _normalize_vertices(V_in, want_cols=vcols)
        vcols = V.shape[1] if vcols is None else vcols
        F = _normalize_faces(F_in)
        norm.append((V, F))
        nV_total += int(V.shape[0])
        nF_total += int(F.shape[0])

    # Allocate outputs once
    Vout = np.empty((nV_total, vcols), dtype=np.float32, order='C')
    Fout = np.empty((nF_total, 3),     dtype=np.int32,   order='C')

    # Fill with offsets
    voff = 0
    foff = 0
    for V, F in norm:
        nv, nf = V.shape[0], F.shape[0]
        Vout[voff:voff+nv] = V
        Fout[foff:foff+nf] = F + voff
        voff += nv
        foff += nf

    return Vout, Fout

def generate_terrain_mesh(x_range, y_range, dx):
    """
    Generate a terrain mesh using random heights.
    """
    x, y = np.meshgrid(np.arange(x_range[0], x_range[1], dx), np.arange(y_range[0], y_range[1], dx))
    z = np.random.randn(*x.shape)*10.
    return x.flatten(), y.flatten(), z.flatten()


def make(base_resolution, decimation_rates, tif_path, out_path, mesh_ext='.xmf',
         plarad=1737.4, lonlat0=(0, -90), rescale_fact=1.e-3):

    rds = rio.open_rasterio(tif_path)
    tiff_resolution = int(round(rds.rio.resolution()[0], 0))

    try:
        assert abs(abs(rds.rio.resolution()[0]) - abs(rds.rio.resolution()[1])) < 1.e-12
    except:
        logging.error(f"* Mesh pixes are not square. dx-dy={abs(abs(rds.rio.resolution()[0]) - abs(rds.rio.resolution()[1]))}.")
        exit()
        
    # if the required dn1 (base resolution) is larger than the native GTiff one, decimate the input
    if base_resolution == tiff_resolution:
        # xgrid = rds.coords['y'].values*-1.e-3
        # ygrid = rds.coords['x'].values*1.e-3
        # dem = rds.data[0].T[:, ::-1] * 1.e-3
        xgrid = rds.coords['x'].values * rescale_fact
        ygrid = rds.coords['y'].values * rescale_fact
        dem = rds.data[0][:, ::] * rescale_fact
    elif base_resolution > tiff_resolution and np.mod(base_resolution, tiff_resolution) == 0:
        decimation = int(base_resolution / tiff_resolution)
        xgrid = rds.coords['x'].values[::decimation] * rescale_fact
        ygrid = rds.coords['y'].values[::decimation] * rescale_fact
        dem = rds.data[0][::decimation, ::decimation] * rescale_fact
    else:
        logging.error(f"* Requested b{base_resolution} < tiff resolution ({tiff_resolution}) or"
                      f"not a multiple.")
        exit()

    logging.debug(f"- GTiff read at {base_resolution}mpp (from original {tiff_resolution}mpp).")

    mesh_versions = {}
    for decimation in decimation_rates:
        start = time.time()

        mesh_versions[decimation] = get_uniform_triangle_mesh(xgrid, ygrid, dem, decimation=decimation)
        logging.debug(
            f"- Mesh of shape {mesh_versions[decimation]['shape']} (decimation={decimation}) set up "
            f"after {round(time.time()-start, 5) * 1e3} milli-seconds")

        for coord_style in ["stereo", "cart"]:

            if coord_style == "cart":
                lon, lat = unproject_stereographic(mesh_versions[decimation]['V'][:, 0],
                                                   mesh_versions[decimation]['V'][:, 1], lonlat0[0], lonlat0[1],
                                                   R=plarad) # + mesh_versions[decimation]['V'][:, 2])

                x, y, z = sph2cart(plarad + mesh_versions[decimation]['V'][:, 2], lat, lon)
                V_cart = np.vstack([x, y, z]).T
                mesh = meshio.Mesh(V_cart, [('triangle', mesh_versions[decimation]['F'])])
                # fout = f"in/shackleton_{np.product(mesh_versions[decimation]['shape'])*2}.ply"
                fout = f"{out_path}b{base_resolution}_dn{decimation}{mesh_ext}"

            else:
                mesh = meshio.Mesh(mesh_versions[decimation]['V'], [('triangle', mesh_versions[decimation]['F'])])
                # fout = f"in/shackleton_{np.product(mesh_versions[decimation]['shape'])*2}_st.ply"
                fout = f"{out_path}b{base_resolution}_dn{decimation}_st{mesh_ext}"

            mesh.write(fout)
            print(f"- Delauney mesh computed and saved to {fout}.")
            logging.debug(f"- Delauney mesh computed and saved to {fout}.")

    logging.debug(f"(decimation,num_faces):\n{[(dec, len(mesh['F'])) for dec, mesh in mesh_versions.items()]}")

    return fout


if __name__ == '__main__':
    # 1D flat faces -> ok
    F_flat = np.arange(12)  # 4 triangles
    V = np.zeros((10, 3))
    stack_meshes([(V, F_flat)])  # should succeed

    # object array of triplets -> ok
    F_obj = np.array([[0, 1, 2], [2, 3, 4], [4, 5, 6]], dtype=object)
    stack_meshes([(V, F_obj)])  # should succeed

    # proper (N,3) -> ok
    F_ok = np.array([[0, 1, 2], [2, 3, 4], [4, 5, 6]], dtype=np.int32)
    stack_meshes([(V, F_ok)])  # should succeed
