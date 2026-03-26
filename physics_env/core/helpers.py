# helpers.py
import numpy as np

def limit_vector(vec, max_val):
    """
    Limite un vecteur de manière douce en préservant sa direction
    """
    norm = np.linalg.norm(vec)
    if norm <= max_val:
        return vec
    else:
        return vec * (max_val / norm)
    
def batch_cross(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Vectorized np.cross() for arrays of shape (N, 3)"""
    return np.column_stack([
        A[:,1]*B[:,2] - A[:,2]*B[:,1],
        A[:,2]*B[:,0] - A[:,0]*B[:,2],
        A[:,0]*B[:,1] - A[:,1]*B[:,0]
    ]) 