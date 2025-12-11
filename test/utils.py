import numpy as np

def linear_interpolation(emb1: np.ndarray, emb2: np.ndarray, alpha: float) -> np.ndarray:
    """ Linear interpolation between two embeddings. """
    return (1 - alpha) * emb1 + alpha * emb2

def spherical_linear_interpolation(emb1: np.ndarray, emb2: np.ndarray, alpha: float) -> np.ndarray:
    """ Spherical linear interpolation between two embeddings. """
    emb1_norm = emb1 / np.linalg.norm(emb1)
    emb2_norm = emb2 / np.linalg.norm(emb2)

    dot_product = np.dot(emb1_norm, emb2_norm)
    omega = np.arccos(np.clip(dot_product, -1.0, 1.0))
    sin_omega = np.sin(omega)

    if sin_omega < 1e-6:  # Fall back to linear interpolation
        return linear_interpolation(emb1, emb2, alpha)

    factor1 = np.sin((1 - alpha) * omega) / sin_omega
    factor2 = np.sin(alpha * omega) / sin_omega

    return factor1 * emb1 + factor2 * emb2