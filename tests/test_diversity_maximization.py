import torch
import torch.testing as testing

from selbstaufsicht.models.self_supervised.msa import diversity_maximization as dm


def test_distance_matrix():
    msa = torch.tensor([[1, 2, 3, 4, 5], 
                        [5, 4, 3, 2, 1], 
                        [1, 2, 3, 2, 1]])
    
    distance_mat = dm.distance_matrix(msa)
    distance_mat_ref = torch.tensor([[0, 4, 2], 
                                     [4, 0, 2], 
                                     [2, 2, 0]], dtype=torch.int16)
    
    testing.assert_close(distance_mat, distance_mat_ref, rtol=0, atol=0)


def test_objective_function():
    solution = torch.tensor([1, 0, 1, 1], dtype=torch.bool)
    distance_mat = torch.tensor([[0, 1, 2, 3], 
                                 [1, 0, 4, 5], 
                                 [2, 4, 0, 6],
                                 [3, 5, 6, 0]], dtype=torch.int16)
    
    objective_val = dm.objective_function(solution, distance_mat)
    objective_val_ref = 11
    
    assert objective_val == objective_val_ref
    
    solution = torch.tensor([[1, 1, 1, 0],
                             [1, 1, 0, 1],
                             [1, 0, 1, 1],
                             [0, 1, 1, 1]], dtype=torch.bool)
    
    objective_values = dm.objective_function(solution, distance_mat)
    objective_values_ref = torch.tensor([7, 9, 11, 15], dtype=torch.int32)
    
    testing.assert_close(objective_values, objective_values_ref, rtol=0, atol=0)


def test_determine_sdv():
    solution = torch.tensor([1, 0, 1, 1], dtype=torch.bool)
    distance_mat = torch.tensor([[0, 1, 2, 3], 
                                 [1, 0, 4, 5], 
                                 [2, 4, 0, 6],
                                 [3, 5, 6, 0]], dtype=torch.int16)
    num_sdv = 2
    
    sdv = dm.determine_sdv(solution, distance_mat, num_sdv)
    sdv_ref = torch.tensor([3, 2], dtype=torch.int64)
    
    testing.assert_close(sdv, sdv_ref, rtol=0, atol=0)


def test_set_intersection():
    set_1 = torch.tensor([0, 3, 6, 2, 4, 1, 5], dtype=torch.int32)
    set_2 = torch.tensor([1, 2, 3, 5, 8, 13], dtype=torch.int32)
    
    set_intersection = dm.set_intersection(set_1, set_2)
    set_intersection_ref = torch.tensor([1, 2, 3, 5], dtype=torch.int32)
    
    testing.assert_close(set_intersection, set_intersection_ref, rtol=0, atol=0)


def test_set_difference():
    set_1 = torch.tensor([0, 3, 6, 2, 4, 1, 5], dtype=torch.int32)
    set_2 = torch.tensor([1, 2, 3, 5, 8, 13], dtype=torch.int32)
    
    set_difference = dm.set_difference(set_1, set_2)
    set_difference_ref = torch.tensor([0, 6, 4], dtype=torch.int32)
    
    testing.assert_close(set_difference, set_difference_ref, rtol=0, atol=0)


def test_recombine_solutions():
    solution_1 = torch.tensor([1, 1, 1, 0], dtype=torch.bool)
    solution_2 = torch.tensor([0, 1, 1, 1], dtype=torch.bool)
    distance_mat = torch.tensor([[0, 1, 2, 3], 
                                 [1, 0, 4, 5], 
                                 [2, 4, 0, 6],
                                 [3, 5, 6, 0]], dtype=torch.int16)
    num_samples = 3
    num_sdv = 2
    
    recombined_solution, objective_val = dm.recombine_solutions(solution_1, solution_2, distance_mat, num_samples, num_sdv)
    recombined_solution_ref = torch.tensor([1, 0, 1, 1], dtype=torch.bool)
    objective_val_ref = 11
    
    testing.assert_close(recombined_solution, recombined_solution_ref, rtol=0, atol=0)
    assert objective_val == objective_val_ref


def test_tabu_search():
    initial_solution = torch.tensor([1, 1, 1, 0], dtype=torch.bool)
    initial_objective_val = 7
    distance_mat = torch.tensor([[0, 1, 2, 3], 
                                 [1, 0, 4, 5], 
                                 [2, 4, 0, 6],
                                 [3, 5, 6, 0]], dtype=torch.int16)
    candidate_list_size = 1  # min(sqrt(num_samples), sqrt(num_seq-num_samples))
    improvement_cutoff = 18  # 6*m
    min_tabu_extension = 15
    max_tabu_extension = 25
    
    solution, objective_val = dm.tabu_search(initial_solution, initial_objective_val, distance_mat, candidate_list_size, 
                                             improvement_cutoff, min_tabu_extension, max_tabu_extension)
    solution_ref = torch.tensor([0, 1, 1, 1], dtype=torch.bool)
    objective_val_ref = 15
    
    testing.assert_close(solution, solution_ref, rtol=0, atol=0)
    assert objective_val == objective_val_ref


def test_initialize_population():
    # TODO: Find distance matrix with local minima, s.t. tabu search finds different solutions.
    
    distance_mat = torch.tensor([[0, 1, 2, 3], 
                                 [1, 0, 4, 5], 
                                 [2, 4, 0, 6],
                                 [3, 5, 6, 0]], dtype=torch.int16)
    num_samples = 3
    candidate_list_size = 1  # min(sqrt(num_samples), sqrt(num_seq-num_samples))
    improvement_cutoff = 18  # 6*m
    min_tabu_extension = 15
    max_tabu_extension = 25
    population_size = 1
    max_reinit = 4
    
    population, objective_values = dm.initialize_population(distance_mat, num_samples, candidate_list_size, improvement_cutoff, min_tabu_extension, max_tabu_extension, population_size, max_reinit)
    population_ref = torch.tensor([[0, 1, 1, 1]], dtype=torch.bool)
    objective_values_ref = torch.tensor([15], dtype=torch.int32)
    
    testing.assert_close(population, population_ref, rtol=0, atol=0)
    assert objective_values == objective_values_ref
    

def test_rebuild_population():
    # TODO: Find distance matrix with local minima, s.t. tabu search finds different solutions.
    # NOTE: Without such a distance matrix, a test is not viable
    
    pass


def test_update_population():
    population = torch.tensor([[1, 1, 0, 1],
                               [1, 0, 1, 1]], dtype=torch.bool)
    objective_values = torch.tensor([7, 11], dtype=torch.int32)
    new_solution = torch.tensor([1, 1, 1, 0], dtype=torch.bool)
    new_objective_val = 7
    
    updated_population, updated_objective_values, population_changed = dm.update_population(population.clone(), objective_values.clone(), new_solution, new_objective_val)
    
    testing.assert_close(updated_population, population, rtol=0, atol=0)
    testing.assert_close(updated_objective_values, objective_values, rtol=0, atol=0)
    assert not population_changed
    
    new_solution = torch.tensor([0, 1, 1, 1], dtype=torch.bool)
    new_objective_val = 15
    
    updated_population, updated_objective_values, population_changed = dm.update_population(population.clone(), objective_values.clone(), new_solution, new_objective_val)
    population_ref = torch.tensor([[0, 1, 1, 1],
                                   [1, 0, 1, 1]], dtype=torch.bool)
    objective_values_ref = torch.tensor([15, 11], dtype=torch.int32)
    
    testing.assert_close(updated_population, population_ref, rtol=0, atol=0)
    testing.assert_close(updated_objective_values, objective_values_ref, rtol=0, atol=0)
    assert population_changed


def test_maximize_diversity_mats():
    # TODO: Find suitable MDP (not too complex, but also non-trivial)
    
    pass


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