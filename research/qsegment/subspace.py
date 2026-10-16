import numpy as np

from qsegment import scenario as sc

K = 3
N = len(sc.FREE)
COSTS = np.array(sc.cost_table(K), dtype=float)


def pair_rotation(beta, a, b):
    """exp(-i beta (|a><b| + |b><a|)) on zones a and b, identity on the third zone. Same as XXPlusYY(2 beta)."""
    M = np.eye(3, dtype=complex)
    M[a, a] = M[b, b] = np.cos(beta)
    M[a, b] = M[b, a] = -1j * np.sin(beta)
    return M


def mixer_matrix(beta):
    """Ring mixer on one system: pairs (0,1), (1,2), (2,0) applied in that order."""
    return pair_rotation(beta, 2, 0) @ pair_rotation(beta, 1, 2) @ pair_rotation(beta, 0, 1)


def evolve(gammas, betas):
    """Exact state of the 3-zone QAOA inside the 3^7 valid partitions, starting from the equal superposition.
    The state is indexed like scenario.decode (free system k has digit k of the index)."""
    psi = np.full(K ** N, 1 / np.sqrt(K ** N), dtype=complex)
    for g, b in zip(gammas, betas):
        psi = np.exp(-1j * g * COSTS) * psi
        U = mixer_matrix(b)
        tensor = psi.reshape([K] * N)
        for axis in range(N):
            tensor = np.moveaxis(np.tensordot(U, tensor, axes=([1], [axis])), 0, axis)
        psi = tensor.reshape(-1)
    return psi


def probs(gammas, betas):
    return np.abs(evolve(gammas, betas)) ** 2
