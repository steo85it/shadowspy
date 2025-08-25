from __future__ import annotations
import numpy as np

def crater_profile_z(r, Rc, depth):
    Rs = (Rc*Rc + depth*depth) / (2.0*depth)
    return np.sqrt(np.maximum(Rs*Rs - r*r, 0.0)) - (Rs - depth)

def make_spherical_crater_mesh(Rc=1_000.0, depth=200.0, Rdom=3_000.0,
                               Nr=160, Nth=360):
    r = np.linspace(0, Rdom, Nr)
    th = np.linspace(0, 2*np.pi, Nth, endpoint=False)
    rr, tt = np.meshgrid(r, th, indexing='ij')
    x = rr*np.cos(tt); y = rr*np.sin(tt)
    z = np.where(rr <= Rc, crater_profile_z(rr, Rc, depth), 0.0)
    V = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    def idx(i,j): return i*Nth + (j % Nth)
    faces = []
    for i in range(Nr-1):
        for j in range(Nth):
            v00 = idx(i, j);     v01 = idx(i, j+1)
            v10 = idx(i+1, j);   v11 = idx(i+1, j+1)
            faces.append([v00, v10, v11])
            faces.append([v00, v11, v01])
    F = np.asarray(faces, dtype=np.uint32)
    return V, F
