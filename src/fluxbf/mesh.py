from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

# Make the butterfly wrapper importable like butterfly/examples do
_here = Path(__file__).resolve().parent
_bf_wrapper = (_here / '../../wrappers/python').resolve()
if _bf_wrapper.exists():
    sys.path.insert(0, str(_bf_wrapper))

import butterfly as bf  # confirmed

def _write_obj(path: Path, V: np.ndarray, F: np.ndarray) -> None:
    with open(path, 'w') as f:
        for p in V:
            f.write(f"v {p[0]} {p[1]} {p[2]}\n")
        for tri in F:
            f.write(f"f {int(tri[0])+1} {int(tri[1])+1} {int(tri[2])+1}\n")  # OBJ is 1-based

class TriMesh:
    """Thin façade over bf.Trimesh."""
    def __init__(self, _bf_mesh):
        self._bf = _bf_mesh

    @classmethod
    def from_obj(cls, obj_path: str | Path) -> "TriMesh":
        m = bf.Trimesh.from_obj(str(Path(obj_path).expanduser().resolve()))
        return cls(m)

    @classmethod
    def from_numpy(cls, V: np.ndarray, F: np.ndarray) -> "TriMesh":
        # Use a temp OBJ to leverage the existing loader
        import tempfile
        V = np.asarray(V, dtype=np.float64)   # bf expects doubles
        F = np.asarray(F, dtype=np.uint32)    # faces as uint32 (0-based in memory; converted to 1-based in OBJ)
        with tempfile.TemporaryDirectory() as td:
            obj = Path(td) / "tmp.obj"
            _write_obj(obj, V, F)
            m = bf.Trimesh.from_obj(str(obj))
        return cls(m)

    def init_embree(self) -> None:
        self._bf.init_embree()  # confirmed

    @property
    def num_faces(self) -> int:
        return self._bf.num_faces  # confirmed property

    @property
    def num_verts(self) -> int:
        return self._bf.num_verts  # confirmed property
