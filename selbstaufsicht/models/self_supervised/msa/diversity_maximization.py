import torch
from typing import List, Tuple


def distance_matrix(msa: torch.Tensor, verbose: bool, process_id: int) -> torch.Tensor:
    """
    Creates symmetric distance matrix, containing the hamming distance between each pair of sequences.

    Args:
        msa (torch.Tensor): MSA [num_seq, len_seq].
        verbose (bool): Activates verbose output.
        process_id (int): Process id.

    Returns:
        torch.Tensor: Distance matrix [num_seq, num_seq].
    """
    
    num_seq, len_seq = msa.shape
    distance_mat = torch.zeros((num_seq, num_seq), dtype=torch.int16)
    
    for idx in range(num_seq):
        temp_1 = msa[idx:idx+1].expand(num_seq-idx, -1)
        temp_2 = msa[idx:]
        distance_mat[idx, idx:] = (temp_1 != temp_2).sum(dim=-1, dtype=torch.int16)
        
        if verbose:
            print(process_id, ":", idx, "/", num_seq)
    
    distance_mat += distance_mat.T - torch.diag(torch.diag(distance_mat))
    
    return distance_mat


def maximize_diversity_greedy(distance_mat: torch.Tensor, num_samples: int) -> torch.Tensor:
    """
    Implements a greedy approach to solve a maximum-diversity-problem (MDP), specified by a distance matrix.
    Initially, it selects the very first element as a reference. 
    Subsequently, the element with the highest average distance to the current set of selected sequences is selected from the set of unselected ones.

    Args:
        distance_mat (torch.Tensor): Distance matrix [num_seq, num_seq].
        num_samples (int): Number of sequences, whose diversity is to be maximized over all sequences.

    Returns:
        torch.Tensor: Solution of the MDP [num_seq].
    """
    
    distance_mat = distance_mat.float()
    solution = torch.zeros((distance_mat.shape[0]), dtype=torch.bool)
    solution[0] = 1
    
    for idx in range(1, num_samples):
        selected_idx = torch.argmax(torch.mean(distance_mat[solution, :], dim=0) * ~solution)
        solution[selected_idx] = 1
    
    assert solution.sum() == num_samples
    
    return solution


def maximize_diversity_msa_greedy(data: List[torch.Tensor], num_samples: int, verbose: bool, process_id: int) -> List[torch.Tensor]:
    """
    Creates a subset of sequences with maximized diversity for each MSA by solving the corresponding maximum-diversity-problem (MDP) using a greedy 
    approach.

    Args:
        data (List[torch.Tensor]): List of MSAs [num_seq, len_seq].
        num_samples (int): Number of sequences, whose diversity is to be maximized over all sequences.
        verbose (bool): Activates verbose output.
        process_id (int): Process id.

    Returns:
        List[torch.Tensor]: List of sequence indices per MSA [num_samples].
    """

    results = []
    
    for idx, msa in enumerate(data):
        if verbose:
            print(process_id, ": Starting MSA", idx+1, "/", len(data))
        distance_mat = distance_matrix(msa, verbose, process_id)
        solution = maximize_diversity_greedy(distance_mat, num_samples)
        seq_indices = torch.arange(solution.shape[0], dtype=torch.int64)
        results.append(seq_indices[solution])
        if verbose:
            print(process_id, ": MSA", idx+1, "/", len(data), "finished")
    
    return results