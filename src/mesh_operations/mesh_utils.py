import numpy as np
import meshio

from shadowspy.shape import get_centroids as get_cents, get_surface_normals


def filter_faces(vertices_mask, faces):
    """
    Filters out faces based on a boolean mask for vertices.

    Args:
    - vertices_mask (np.ndarray): Boolean array indicating active vertices (True for active).
    - faces (np.ndarray): Array of faces (Mx3) using vertex indices.

    Returns:
    - np.ndarray: Filtered array of faces, keeping only those where all vertices are active.
    """
    # Use the mask to check if all vertices in each face are active
    face_active_mask = vertices_mask[faces].all(axis=1)

    # Filter faces based on the active mask
    filtered_faces = faces[face_active_mask]

    return filtered_faces

def remove_degenerate_faces(V, F):
    v1, v2, v3 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    areas = np.linalg.norm(np.cross(v2 - v1, v3 - v1), axis=1) / 2
    valid_faces = areas > 1e-10  # Threshold for zero-area faces
    print(f"- Removed {len(F)-sum(valid_faces)} degenerate faces. ({len(F)}/{sum(valid_faces)})")
    return F[valid_faces]

def remove_faces_with_vertices(faces, vertices_to_remove):
    # Convert vertices_to_remove to a set for faster lookup
    vertices_to_remove_set = set(vertices_to_remove)

    # Create a mask to keep faces where not all vertices are in vertices_to_remove
    mask = np.array([not (face[0] in vertices_to_remove_set and
                          face[1] in vertices_to_remove_set and
                          face[2] in vertices_to_remove_set)
                     for face in faces])

    # Apply mask to faces
    filtered_faces = faces[mask]

    return filtered_faces

def load_mesh(file_path: str):
    """Load a VTK mesh from a file."""
    mesh = meshio.read(
        filename=file_path,  # string, os.PathLike, or a buffer/open file
    )

    V = mesh.points
    V = V[:, :3]
    F = mesh.cells[0].data


    return {'V': V, 'F': F}

def remove_inner_from_outer(outer_vertices, inner_bbox):
    """
    Remove vertices from the outer mesh that fall within the inner mesh's bounding box.
    """
    x_min, x_max, y_min, y_max = inner_bbox
    mask = ~((outer_vertices[:, 0] >= x_min) & (outer_vertices[:, 0] <= x_max) &
             (outer_vertices[:, 1] >= y_min) & (outer_vertices[:, 1] <= y_max))
    return outer_vertices[mask], mask


def import_mesh(mesh_path, get_normals=False, get_centroids=False):
    # use meshio to import obj shapefile
    mesh = meshio.read(
        filename=mesh_path,  # string, os.PathLike, or a buffer/open file
    )

    V = mesh.points
    # V = V.astype(np.float32)  # embree is anyway single precision # destroys normals
    V = V[:, :3]
    F = mesh.cells[0].data

    if (not get_normals) and (not get_centroids):
        return V, F

    if get_normals:
        P = get_cents(V, F)
        N = get_surface_normals(V, F)
        N[(N * P).sum(1) < 0] *= -1
        if get_centroids:
            return V, F, N, P
        else:
            return V, F, N
