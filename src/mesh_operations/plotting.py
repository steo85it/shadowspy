import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import rioxarray

def plot_transition(vertices_in, faces_in, vertices_out, faces_out, transition_vertices, transition_faces):
    pass

def rasterize_grid(gridded_data, bounds, crs=None, to_geotiff=None):

    xmin, xmax, ymin, ymax = bounds

    # Create coordinate arrays based on the bounds and grid size
    x_coords = np.linspace(xmin, xmax, gridded_data.shape[1])
    y_coords = np.linspace(ymax, ymin, gridded_data.shape[0])  # Reverse y for correct orientation

    # Create xarray DataArray
    da = xr.DataArray(
        gridded_data,
        coords={"y": y_coords, "x": x_coords},
        dims=("y", "x"),
    )

    if crs != None:
        # Assign CRS to the DataArray
        da = da.rio.write_crs(crs)

    if to_geotiff != None:
        # Save as GeoTIFF
        da.rio.to_raster(to_geotiff, compress="LZW")

    return da


def rasterize_with_raytracing(variables, shape_model_st):

    # Create a dictionary of variables to iterate over (T[:,0] --> analyze/save surface only)
    # variables = {'T': Tsurf_prev, 'E': E0, 'Qrefl': Qrefl, 'QIR': QIR}
    V_st = shape_model_st.V

    xmin = np.min(V_st[:, 0])
    xmax = np.max(V_st[:, 0])
    ymin = np.min(V_st[:, 1])
    ymax = np.max(V_st[:, 1])
    extent = xmin, xmax, ymin, ymax

    # Loop through the dictionary
    layers = {}
    for idx, (var_name, var_value) in enumerate(variables.items()):

        def get_values_using_raytracing(field):
            import itertools as it
            print(field.ndim, 1, field.size, shape_model_st.num_faces)
            assert field.ndim == 1 and field.size == shape_model_st.num_faces
            dtype = field.dtype
            d = np.array([0, 0, 1], dtype=np.float64)
            m, n = len(xgrid), len(ygrid)
            grid = np.empty((m, n), dtype=dtype)
            grid[...] = np.nan
            for i, j in it.product(range(m), range(n)):
                x = np.array([xgrid[i], ygrid[j], -1], dtype=dtype)
                hit = shape_model_st.intersect1(x, d)
                if hit is not None:
                    grid[i, j] = field[hit[0]]
            return grid

        # set up plot grid
        N = 512
        xgrid = np.linspace(xmin, xmax, N)
        ygrid = np.linspace(ymin, ymax, N)
        grid = get_values_using_raytracing(var_value)

        da = rasterize_grid(grid.T, (xmin, xmax, ymax, ymin))
        da = da.rename(var_name)  # Assign variable name
        layers[var_name] = da

    # Combine all layers into a Dataset
    da = xr.Dataset(layers)

    return da


if __name__ == '__main__':
    # Example data (replace with your actual data)
    example_gridded_data = np.random.rand(512, 512)  # Replace with your (512, 512) grid
    xmin, xmax, ymin, ymax = -65, -25, -20, 20

    da = rasterize_grid(example_gridded_data, (xmin, xmax, ymin, ymax), crs='WGS84')
    da.plot(robust=True)
    plt.show()