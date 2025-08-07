import meshio
import numpy as np

from shadowspy.coord_tools import unproject_stereographic, sph2cart
from mesh_operations import load_mesh


def split_merged(total_mesh_path, len_inner_faces, inner_mesh_path, outer_mesh_path,
                 mesh_extension, planetary_radius, lonlat0=(0, -90)):

    mesh_tot = load_mesh(total_mesh_path)
    mesh_in = {'V': mesh_tot['V'], 'F': mesh_tot['F'][:len_inner_faces]}
    mesh_in_ = meshio.Mesh(mesh_in['V'], [('triangle', mesh_in['F'])])
    mesh_in_.write(f"{inner_mesh_path}_st{mesh_extension}")

    mesh_far = {'V': mesh_tot['V'], 'F': mesh_tot['F'][len_inner_faces:]}
    meshfar_ = meshio.Mesh(mesh_far['V'], [('triangle', mesh_far['F'])])
    meshfar_.write(f"{outer_mesh_path}_st{mesh_extension}")

    # we use mesh_tot as outer mesh since rendering does not like a hole in the middle,
    # we rather need to have a full mesh, but identical to inner_mesh where they overlap
    for mpath, mesh_ in {f"{inner_mesh_path}{mesh_extension}": mesh_in,
                         f"{outer_mesh_path}{mesh_extension}": mesh_tot}.items():
        lon, lat = unproject_stereographic(mesh_['V'][:, 0],
                                           mesh_['V'][:, 1], lon0=lonlat0[0], lat0=lonlat0[1],
                                           R=planetary_radius)  # + mesh_versions[decimation]['V'][:, 2])

        x, y, z = sph2cart(planetary_radius + mesh_['V'][:, 2], lat, lon)
        V_cart = np.vstack([x, y, z]).T
        mesh = meshio.Mesh(V_cart, [('triangle', mesh_['F'])])
        mesh.write(mpath)

    return inner_mesh_path, outer_mesh_path
