import meshio
import numpy as np
import logging

from mesh_operations.mesh_utils import import_mesh, remove_degenerate_faces
from shadowspy.shape import get_centroids


def get_uniform_triangle_mesh(xgrid, ygrid, data, decimation=1):

    # decimate as requested
    dem = data[::decimation, ::decimation]

    X_mesh, Y_mesh = np.meshgrid(xgrid[::decimation], ygrid[::decimation])
    points_mesh = np.array([X_mesh.flatten(), Y_mesh.flatten()]).T

    # generate triangle faces indexes
    X_idx_mesh, Y_idx_mesh = np.meshgrid(np.arange(len(xgrid[::decimation]) - 1),
                                         np.arange(len(ygrid[::decimation]) - 1))
    index_mesh = np.array([X_idx_mesh.flatten(), Y_idx_mesh.flatten()]).T

    # adapt dimensions to index_mesh XY order
    X_mesh_shape = (X_mesh.shape[1], X_mesh.shape[0])

    # upper triangles
    ik1 = np.ravel_multi_index([index_mesh[:, 0], index_mesh[:, 1]], dims=X_mesh_shape, order="F")
    ik2 = np.ravel_multi_index([index_mesh[:, 0] + 1, index_mesh[:, 1] + 1], dims=X_mesh_shape, order="F")
    ik3 = np.ravel_multi_index([index_mesh[:, 0], index_mesh[:, 1] + 1], dims=X_mesh_shape, order="F")
    Fu = np.vstack([ik1, ik2, ik3]).T
    # lower triangles
    ik1 = np.ravel_multi_index([index_mesh[:, 0], index_mesh[:, 1]], dims=X_mesh_shape, order="F")
    ik2 = np.ravel_multi_index([index_mesh[:, 0] + 1, index_mesh[:, 1]], dims=X_mesh_shape, order="F")
    ik3 = np.ravel_multi_index([index_mesh[:, 0] + 1, index_mesh[:, 1] + 1], dims=X_mesh_shape, order="F")
    Fl = np.vstack([ik1, ik2, ik3]).T

    F = np.vstack([Fu, Fl])

    V = points_mesh
    z = np.ndarray.flatten(dem)
    V = np.row_stack([V.T, z]).T

    # clean up mesh
    F = remove_degenerate_faces(V, F)

    P = get_centroids(V, F)

    # generate mesh of indexes to get relations among faces
    X_idx_mesh, Y_idx_mesh = np.meshgrid(np.arange(len(xgrid[::decimation])), np.arange(len(ygrid[::decimation])))
    index_mesh = np.array([X_idx_mesh.flatten(), Y_idx_mesh.flatten()]).T
    Vidx = index_mesh

    return {"F": F, "V": V, "P": P, "Vidx": Vidx, "shape": X_mesh[:-1, :-1].shape}


def relate_decimation_faces(vixy, shape, decimation=1):
    shape_lowres = tuple(
        int(np.floor(s / decimation)) for s in shape)  # only works if original decimation is 1 (it usually is, by def)

    # from highres vertices, get lowres vertices by decimation
    vixy = np.stack(vixy, axis=0)
    xy_lowres = np.floor(vixy[:, 0, :] / decimation).astype(int)

    # if an highres triangle exceeds the lowres mesh (non divisible decimation factor) just relate it to the last row/column
    xy_lowres = np.where(xy_lowres < shape_lowres, xy_lowres, xy_lowres - 1)

    lowres_square_idx = np.ravel_multi_index([xy_lowres[:, 0], xy_lowres[:, 1]], dims=shape_lowres, order="F")

    highres_tri_idx = np.arange(np.product(shape) * 2)  # num_triangles = nrows*ncols*2
    high_to_low_blocks = dict(zip(highres_tri_idx, lowres_square_idx))

    # check function of diagonal for a given block (given by yi-xi)
    lowres_blocks_idx = np.arange(np.product(shape_lowres))
    lowres_diags = np.diff(np.vstack(np.unravel_index(lowres_blocks_idx, shape_lowres, order='F')).T, axis=1)
    blocks_to_diags = dict(zip(lowres_blocks_idx, lowres_diags.T[0]))

    # associate to each small triangle a "diagonal rule" depending on the block to which it belongs
    diags_def = np.array([blocks_to_diags[k] for k in high_to_low_blocks.values()])
    # if all vertices have yi >= xi - decimation_factor*diag_fct, then hires-triangle is in upper lowres-triangle
    is_upper_triangle = np.sum(np.diff(vixy, axis=2)[:, :, 0] >= np.repeat(decimation * diags_def, 3).reshape(-1, 3),
                               axis=1) == 3

    # get dictionary of lowres triangle faces indexes to highres triangles faces
    high_to_low_tri = dict(zip(highres_tri_idx,
                               np.array(list(high_to_low_blocks.values())) + ~is_upper_triangle * np.product(
                                   shape_lowres)))

    # # TODO convenient but way too slow
    # high_to_low_tri_list = {lowres_tri:[k for k in high_to_low_tri if high_to_low_tri[k] == lowres_tri] for lowres_tri in range(np.product(mesh_versions[decimation]["shape"])*2)}
    # # # check that all highres faces have been assigned to a lowres face
    # assert len(mesh_versions[1]["F"]) == len(np.hstack([v for k, v in high_to_low_tri_list.items()]))

    return high_to_low_tri


def vertex_faces(vertices, triangles):
    # Create an empty list to store vertex faces
    vertex_faces = [[] for _ in range(len(vertices))]

    # Iterate through triangles and populate vertex_faces
    for i, triangle in enumerate(triangles):
        for vertex_index in triangle:
            vertex_faces[vertex_index].append(i)

    # Convert the lists of faces to NumPy arrays
    vertex_faces = [np.array(faces) for faces in vertex_faces]

    # Determine the maximum length of arrays in the list
    max_length = max(len(arr) for arr in vertex_faces)

    # Create an empty array filled with NaN values
    result = np.full((len(vertex_faces), max_length), np.nan)

    # Populate the result array with values from the list
    for i, arr in enumerate(vertex_faces):
        result[i, :len(arr)] = arr

    return result


def crop_mesh(polysgdf, meshes, mask, meshes_cropped):
    import geopandas as gpd
    import shapely

    if not isinstance(polysgdf, gpd.GeoDataFrame):
        logging.error("* mask should be a GeoDataFrame")
        exit()

    V_st, F_st, N_st, P_st = import_mesh(f"{meshes['stereo']}", get_normals=True, get_centroids=True)
    V, F, N, P = import_mesh(f"{meshes['cart']}", get_normals=True, get_centroids=True)

    pointsDF = gpd.GeoDataFrame(geometry=shapely.points(V_st[:, 0], V_st[:, 1]))
    pointsDF.set_crs(mask.crs, inplace=True)

    joinDF = pointsDF.sjoin(mask, how='left', op="within")
    # crop V
    Vcropped_st = V_st[joinDF.dropna(axis=0).index].astype(float)
    Vcropped = V[joinDF.dropna(axis=0).index].astype(float)

    if (len(Vcropped_st) == 0) or (len(Vcropped) == 0):
        logging.error("* Meshes and mask do not overlap. Weird. Stop")
        exit()

    # covert F idx to new Vcropped
    Vdict = {vold: idx for idx, vold in enumerate(joinDF.dropna(axis=0).index.to_list())}
    Fcropped = np.reshape([*map(Vdict.get, F.ravel())], (-1, 3))
    Fcropped = Fcropped[~np.isnan(Fcropped.astype(float)).any(axis=1)].astype(int)

    # save cropped mesh to file
    mesh = meshio.Mesh(Vcropped_st, [('triangle', Fcropped)])
    mesh.write(f"{meshes_cropped['stereo']}")
    mesh = meshio.Mesh(Vcropped, [('triangle', Fcropped)])
    mesh.write(f"{meshes_cropped['cart']}")

    return {'V': Vcropped, 'V_st': Vcropped_st, 'F': Fcropped}
