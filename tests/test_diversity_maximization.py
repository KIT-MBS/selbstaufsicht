import torch
import torch.testing as testing

from selbstaufsicht.models.self_supervised.msa import diversity_maximization as dm


def test_distance_matrix():
    msa = torch.tensor([[1, 2, 3, 4, 5], 
                        [5, 4, 3, 2, 1], 
                        [1, 2, 3, 2, 1]])
    
    distance_mat = dm.distance_matrix(msa, False, 0)
    distance_mat_ref = torch.tensor([[0, 4, 2], 
                                     [4, 0, 2], 
                                     [2, 2, 0]], dtype=torch.int16)
    
    testing.assert_close(distance_mat, distance_mat_ref, rtol=0, atol=0)


def test_maximize_diversity_greedy():
    distance_mat = torch.tensor([[0, 1, 2, 3], 
                                 [1, 0, 4, 5], 
                                 [2, 4, 0, 6],
                                 [3, 5, 6, 0]], dtype=torch.int16)
    
    num_samples = 3
    
    solution = dm.maximize_diversity_greedy(distance_mat, num_samples)
    print(solution)
    solution_ref = torch.tensor([1, 0, 1, 1], dtype=torch.bool)

    testing.assert_close(solution, solution_ref, rtol=0, atol=0)