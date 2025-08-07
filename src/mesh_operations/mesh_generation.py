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

    Args:
    - meshes: A list of tuples, where each tuple is (vertices, faces) for a mesh.

    Returns:
    - combined_vertices: The combined vertices array for all meshes.
    - combined_faces: The combined and correctly indexed faces array for all meshes.
    """
    all_vertices = []
    all_faces = []
    vertex_offset = 0

    for vertices, faces in meshes:
        # Append the current vertices
        all_vertices.append(vertices)

        # Adjust faces indices and append
        adjusted_faces = faces + vertex_offset
        all_faces.append(adjusted_faces)

        # Update the offset for the next mesh
        vertex_offset += vertices.shape[0]

    combined_vertices = np.vstack(all_vertices)
    combined_faces = np.vstack(all_faces)

    return combined_vertices, combined_faces

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
