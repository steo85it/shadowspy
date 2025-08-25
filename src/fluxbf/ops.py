from __future__ import annotations
import numpy as np
from .mesh import bf, TriMesh

def view_factor_csr(mesh: TriMesh, I: np.ndarray | None = None, J: np.ndarray | None = None):
    """
    Build the view-factor matrix using butterfly's C++ backend (CSR).
    Returns a bf.MatCsrReal (subclass of bf.Mat).
    """
    if I is not None:
        I = np.asarray(I, dtype=np.uintp)
    if J is not None:
        J = np.asarray(J, dtype=np.uintp)
    F = bf.MatCsrReal.new_view_factor_matrix_from_trimesh(mesh._bf, I, J)  # confirmed signature
    return F

def mv(F, X: np.ndarray) -> np.ndarray:
    """
    Multiply by F with one or multiple RHS:
      X: shape (n,) or (n,k) numpy array
      Returns: (m,) or (m,k) numpy array
    """
    X = np.asarray(X)
    Y = F @ X  # triggers Mat.__matmul__; returns VecReal or a dense Mat
    # 1D case → VecReal with .to_array()
    if hasattr(Y, "to_array"):
        return Y.to_array()
    # 2D case → MatDenseReal implements __array__; np.asarray handles it
    return np.asarray(Y)
