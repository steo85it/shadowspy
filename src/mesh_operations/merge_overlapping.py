import meshio
from matplotlib import pyplot as plt
from scipy.spatial import Delaunay
import numpy as np

from mesh_operations.boundary_finding import triangulate_and_find_boundaries_vectorized
from mesh_operations.mesh_generation import generate_terrain_mesh, stack_meshes
from mesh_operations.mesh_utils import remove_inner_from_outer, filter_faces, load_mesh, remove_faces_with_vertices


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

    # Perform triangulation and find boundaries TODO check if ok
    faces_out, dummy, inner_vertices_outer = triangulate_and_find_boundaries_vectorized(faces, vertices_out)
    # faces_out, dummy, inner_vertices_outer = triangulate_and_find_boundaries_vectorized(faces, vertices_with_hole) # don't use, results in artifacts
    vertices_in = np.vstack((x_in, y_in, z_in)).T
    faces_in, outer_vertices_inner, dummy = triangulate_and_find_boundaries_vectorized(inner_mesh['F'], vertices_in)

    # Print results
    if debug:
        print("Outer Boundary Vertices:", outer_vertices_inner)
        print("Inner Boundary Vertices:", inner_vertices_outer)

    # generate transition faces
    transition_vertices = np.vstack([vertices_in[outer_vertices_inner], vertices_out[inner_vertices_outer]])
    delaunay = Delaunay(transition_vertices[:, :2])
    transition_faces = remove_faces_with_vertices(delaunay.simplices, range(len(outer_vertices_inner)))

    # Plot
    if debug:
        plt.triplot(vertices_out[:, 0], vertices_out[:, 1], faces_out)
        plt.triplot(vertices_in[:, 0], vertices_in[:, 1], faces_in)
        plt.triplot(transition_vertices[:, 0], transition_vertices[:, 1], transition_faces)
        plt.plot(vertices_in[outer_vertices_inner, 0], vertices_in[outer_vertices_inner, 1], 'ro', label='Outer Boundary')
        plt.plot(vertices_out[inner_vertices_outer, 0], vertices_out[inner_vertices_outer, 1], 'go', label='Inner Boundary')
        plt.legend()
        plt.show()

    # Stack them
    combined_vertices, combined_faces = stack_meshes([
        (vertices_in, faces_in),
        (transition_vertices, transition_faces),
        (vertices_out, faces_out)
    ])

    labels_dict = {
                    'inner': len(faces_in),
                    'transition': len(transition_faces),
                    'outer': len(faces_out),
                    'total': len(combined_faces)
                }
    print(labels_dict)

    # Write the combined mesh to a file
    # Create mesh objects using meshio and write to VTK files
    final_stacked_mesh = meshio.Mesh(points=combined_vertices,
                                     cells=[("triangle", combined_faces)])
    # Write to VTK files
    final_stacked_mesh.write(output_path)

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
