import random
import cvxpy as cp
import numpy as np
from scipy.linalg import ldl
import torch

# ambuguous RNA letters: GAUCRYWSMKHBVDN
rna_letters = [letter for letter in '-.GAUCRYWSMKHBVDN']

rna2index = {letter: index for index, letter in enumerate(rna_letters)}
rna2index['START_TOKEN'] = len(rna2index)
rna2index['DELIMITER_TOKEN'] = len(rna2index)
# NOTE these two have to be last, since they can not be inpainted
rna2index['MASK_TOKEN'] = len(rna2index)
rna2index['PADDING_TOKEN'] = len(rna2index)

# NOTE: Predefined set of actual tokens used for non-static masking
nonstatic_mask_token_letters = [letter for letter in 'ACGU-']
nonstatic_mask_tokens = [rna2index[letter] for letter in nonstatic_mask_token_letters]


def lehmer_encode(i: int, n: int) -> torch.Tensor:
    """
    Encodes an integer i in the interval [0,n!-1] as a permutation of (0,1,2,...,n-1).

    Args:
        i (int): Lehmer index.
        n (int): Number of elements in the sequence to be permuted.

    Example:
        >>> lehmer_encode(10,7)
        tensor([0, 1, 2, 4, 6, 3, 5], dtype=torch.int64)

    Returns:
        torch.Tensor: Permutation.
    """

    pos = torch.empty((n - 1,), dtype=torch.int64)
    for j in range(2, n + 1):
        ii = i // j
        pos[n - j] = i - ii * j
        i = ii

    assert(i == 0)

    init = torch.arange(n, dtype=torch.int64)
    ret = torch.empty((n,), dtype=torch.int64)
    for j in range(n - 1):
        jj = 0
        for k in range(n):
            if init[k] == -1:
                continue
            if jj == pos[j]:
                ret[j] = k  # should be the same as init[k]
                init[k] = -1
                break
            jj += 1

    for k in range(n):
        if init[k] == -1:
            continue
        ret[n - 1] = k

    return ret


def perm_metric(perm1: torch.Tensor, perm2: torch.Tensor) -> int:
    """
    Discrete metric for permutation comparisons. Computes the number of transpositions.

    Args:
        perm1 (torch.Tensor): First permutation [M].
        perm2 (torch.Tensor): Second permutation [M].

    Raises:
        RuntimeError: Permutations are passed wrongly formatted.

    Returns:
        int: Number of transpositions.
    """

    invperm2 = torch.argsort(perm2)
    perm0 = perm1[invperm2]
    seen = torch.zeros_like(perm0, dtype=torch.bool)
    n = seen.shape[0]
    dist = 0

    for i in range(n):
        if seen[i]:
            continue
        seen[i] = True
        cur = i
        for j in range(n):
            cur = perm0[cur]
            if seen[cur]:
                if cur == i:
                    break
                raise RuntimeError("Probably not a permutation?")
            seen[cur] = True
            dist += 1

    assert torch.all(seen == True)
    return dist


def perm_gram_matrix(perms: torch.Tensor) -> torch.Tensor:
    """
    Computes a gram matrix of all-to-all permutation comparisons in terms of transpositions.

    Args:
        perms (torch.Tensor): Permutations [N, M].

    Returns:
        torch.Tensor: Symmetric gram matrix [N, N], which contains the number of transpositions for all permutation pairs.
    """

    n = perms.shape[0]
    ret = torch.zeros((n,n), dtype=torch.int64)
    for i in range(n-1):
        for j in range(i+1,n):
            tmp = perm_metric(perms[i], perms[j])
            ret[i,j] = tmp
            ret[j,i] = tmp

    return ret


def embed_finite_metric_space(d0: torch.Tensor, eps: float = 1e-4, dist_lower_bound: float = 0.25) -> torch.Tensor:
    """
    Computes the non-negative Euclidean embedding of the discrete permutation metric, given a gram matrix, by solving a linear program.

    Args:
        d0 (torch.Tensor): Symmetric gram matrix [N, N], which contains the number of transpositions for all permutation pairs.
        eps (float, optional): Constant for numerical stability. Defaults to 1e-4.
        dist_lower_bound (float, optional): Lower bound for the distortion, which is used to diminish the embedding dimensionality further. Defaults to 0.25.

    Returns:
        torch.Tensor: Euclidean embedding of the discrete permutation metric [N, L]. All vector components are non-negative.
    """

    n = d0.shape[0]
    X = cp.Variable((n-1, n-1), symmetric=True)
    z = cp.Variable((1,))
    constraints = [ X >> 0 ]
    constraints += [ z >= 0 ]
    min_distance = torch.inf
    d0 = d0.numpy()

    # define linear program
    for i in range(n-1):
        constraints += [ X[i, i] <= d0[i+1 ,0]**2 ]
        constraints += [ X[i, i] >= z * d0[i+1, 0]**2 ]
        min_distance = min(d0[i+1, 0], min_distance)
        for j in range(i+1,n-1):
            constraints += [ X[i, i] + X[j, j] - 2.0 * X[i, j] <= d0[i+1, j+1]**2 ]
            constraints += [ X[i, i] + X[j, j] - 2.0 * X[i, j] >= z * d0[i+1, j+1]**2 ]
            min_distance = min(d0[i+1, j+1], min_distance)

    prob = cp.Problem(cp.Maximize(z), constraints)
    prob.solve(eps=eps, use_indirect=False)

    # LDL^T decomposition is more stable than cholesky
    L, D, _ = ldl(X.value, lower=True)
    L = torch.tensor(L)
    D = torch.Tensor(D)
    diag = torch.sqrt(torch.maximum(torch.diag(D), torch.zeros(1)))
    L = L * diag[None, :]
    ret = torch.vstack((torch.zeros((1, n-1)), L)).float()

    # check result
    max_dist = 1.0
    for i in range(n-1):
        for j in range(i+1,n):
            max_dist = min(max_dist, torch.linalg.vector_norm(ret[i,:] - ret[j,:]) / d0[i,j])

    z_sqrt = torch.sqrt(torch.tensor(z.value))
    if torch.abs(z_sqrt - max_dist) > eps:
        print("WARNING: Embedding did not converge: {} vs {}".format(z_sqrt.item(), max_dist))

    # diminish dimensionality, if possible
    if max_dist > dist_lower_bound:
        max_entries = torch.amax(torch.abs(ret), dim=0)**2
        ind = torch.argsort(max_entries)
        max_entries_sorted = max_entries[ind]
        max_entries_cumsum = torch.cumsum(torch.flatten(max_entries_sorted), dim=0).view_as(max_entries_sorted)

        delta_sq = (max_dist - dist_lower_bound)**2 * min_distance**2
        ind2 = torch.where(max_entries_cumsum >= delta_sq)[0]
        ret = ret[:,ind[ind2]]

    # shift embedding into non-negative area of the vector space
    shift_vec = torch.amin(ret, dim=0)
    shift_vec *= -(shift_vec < 0).to(torch.int)
    ret += shift_vec

    return ret


def data_loader_worker_init(worker_id: int, rng_seed: int) -> None:
    """
    Initialization method for data loader workers, which fixes the random number generator seed.

    Args:
        worker_id (int): Worker ID.
        rng_seed (int): Random number generator seed.
    """

    np.random.seed(rng_seed)
    random.seed(rng_seed)
