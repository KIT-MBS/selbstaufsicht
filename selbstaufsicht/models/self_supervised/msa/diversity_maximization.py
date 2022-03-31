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


def objective_function(solution: torch.Tensor, distance_mat: torch.Tensor) -> torch.Tensor:
    """
    Computes objective function (i.e., the sum of all pair-wise different places from the all-to-all comparison of selected sequences) 
    for one or multiple given solution(s). Supports batch dimension(s).

    Args:
        solution (torch.Tensor): Mask that determines which sequences are selected [*, num_seq].
        distance_mat (torch.Tensor): Distance matrix [num_seq, num_seq].

    Returns:
        torch.Tensor: Objective value(s) [*]
    """
    
    mask_1 = solution.view(*solution.shape, 1).expand(*solution.shape, solution.shape[-1])
    mask_2 = solution.view(*(solution.shape[:-1]), 1, solution.shape[-1]).expand(*solution.shape, solution.shape[-1])
    mask = torch.logical_and(mask_1, mask_2)
    
    return (0.5 *(distance_mat[(None,)*(solution.ndim-1)].expand(*solution.shape, solution.shape[-1]) * mask).sum(dim=(-1, -2), dtype=torch.int32)).to(torch.int32)


def determine_sdv(solution: torch.Tensor, distance_mat: torch.Tensor, num_sdv: int) -> torch.Tensor:
    """
    Determines strongly determined variables, i.e. variables that contribute most to diversity maximization,
    given a feasible solution.

    Args:
        solution (torch.Tensor): Feasible solution [num_seq].
        distance_mat (torch.Tensor): Distance matrix [num_seq, num_seq].
        num_sdv (int): Number of strongly determined variables.

    Returns:
        torch.Tensor: Strongly determined variables [num_sdv].
    """
    
    distance_mat_temp = distance_mat.clone()
    distance_mat_temp[~solution, :] = 0
    sdv = distance_mat_temp[:, solution].sum(dim=1)
    return torch.topk(sdv, num_sdv)[1]


def set_intersection(set_1: torch.Tensor, set_2: torch.Tensor) -> torch.Tensor:
    """
    Computes set intersection between two 1d tensors of unique integers.

    Args:
        set_1 (torch.Tensor): First set.
        set_2 (torch.Tensor): Second set.

    Returns:
        torch.Tensor: Set intersection.
    """
    
    combined = torch.cat((set_1, set_2))
    uniques, counts = combined.unique(return_counts=True)
    
    return uniques[counts > 1].to(set_1.dtype)


def set_difference(set_1: torch.Tensor, set_2: torch.Tensor) -> torch.Tensor:
    """
    Computes set difference between two 1d tensors of unique integers.

    Args:
        set_1 (torch.Tensor): First set.
        set_2 (torch.Tensor): Second set.

    Returns:
        torch.Tensor: Set difference.
    """
    
    return set_1[~set_1.unsqueeze(1).eq(set_2).any(1)]
    

def recombine_solutions(solution_1: torch.Tensor, solution_2: torch.Tensor, distance_mat: torch.Tensor, num_samples: int, 
                        num_sdv: int) -> Tuple[torch.Tensor, int]:
    """
    Implements the recombination operator, which creates a new offspring solution out of two given solutions.

    Args:
        solution_1 (torch.Tensor): First feasible solution [num_seq].
        solution_2 (torch.Tensor): Second feasible solution [num_seq].
        distance_mat (torch.Tensor): Distance matrix [num_seq, num_seq].
        num_samples (int): Number of sequences, whose diversity is to be maximized over all sequences.
        num_sdv (int): Number of strongly determined variables.

    Returns:
        Tuple[torch.Tensor, int]: New offspring solution [num_seq]; Objective value of the new offspring solution.
    """
    
    sdv_1, sdv_2 = determine_sdv(solution_1, distance_mat, num_sdv), determine_sdv(solution_2, distance_mat, num_sdv)
    
    consistent_sdv = set_intersection(sdv_1, sdv_2)
    
    all_variables = torch.arange(solution_1.shape[0], dtype=consistent_sdv.dtype)
    selectable_variables = set_difference(all_variables, consistent_sdv)
    selectable_variables = selectable_variables[torch.randperm(selectable_variables.shape[0])]
    num_selected = num_samples - consistent_sdv.numel()
    selected_variables = torch.cat((selectable_variables[:num_selected], consistent_sdv))
    selected_variables, _ = torch.sort(selected_variables)
    
    recombined_solution = torch.zeros_like(solution_1)
    recombined_solution[selected_variables] = 1
    
    objective_val = objective_function(recombined_solution, distance_mat).item()
    
    return recombined_solution, objective_val


def tabu_search(initial_solution: torch.Tensor, initial_objective_val: int, distance_mat: torch.Tensor, candidate_list_size: int, 
                improvement_cutoff: int, min_tabu_extension: int, max_tabu_extension: int) -> Tuple[torch.Tensor, int]:
    """
    Performs tabu search to optimize the given solution locally.

    Args:
        initial_solution (torch.Tensor): Initial solution [num_seq].
        initial_objective_val (int): Objective value of the initial solution.
        distance_mat (torch.Tensor): Distance matrix [num_seq, num_seq].
        candidate_list_size (int): Number of best candidates that are selected in each iteration of Tabu Search.
        improvement_cutoff (int): Number of iterations of Tabu Search without improvement, after which the procedure is stopped.
        min_tabu_extension (int): Minimum number of iterations of Tabu Search, by which a variable is restricted from change.
        max_tabu_extension (int): Maximum number of iterations of Tabu Search, by which a variable is restricted from change.

    Returns:
        Tuple[torch.Tensor, int]: Optimized solution [num_seq]; Objective value of the optimized solution.
    """
    
    U = initial_solution.clone()
    num_U = U.sum().item()
    Z = ~U
    num_Z = initial_solution.shape[0] - num_U
    
    assert candidate_list_size <= num_U
    assert candidate_list_size <= num_Z
    
    def determine_swap(D: torch.Tensor) -> Tuple[int, int, int]:
        # NOTE: If 0 in D[U] and D[Z], stability cannot be guaranteed. 
        # However, not only is this unlikely in theory, as num_samples+1 rows would have to be equal, it is even impossible for given 
        # MSAs, since all their sequences are different from each other.
        _, D_sorted_idx = D.sort(descending=True)
        ZCL = D_sorted_idx[:candidate_list_size]
        UCL = D_sorted_idx[num_Z:num_Z+candidate_list_size]
        
        swap_candidates = torch.cartesian_prod(UCL, ZCL).T
        
        d = torch.empty((swap_candidates.shape[1],), dtype=torch.int32)
        d = D[swap_candidates[0]] + D[swap_candidates[1]] - distance_mat[swap_candidates[0], swap_candidates[1]]
        
        max_d, max_d_idx = torch.max(d, dim=0)
        max_d = max_d.item()
        UCL_swap_idx, ZCL_swap_idx = swap_candidates[0][max_d_idx].item(), swap_candidates[1][max_d_idx].item()
        
        return max_d, UCL_swap_idx, ZCL_swap_idx
    
    D = torch.empty(initial_solution.shape, dtype=torch.int32)
    D_nt = torch.empty_like(D)
    D_t = torch.empty_like(D)
    D[U] = -distance_mat[U, :][:, U].sum(dim=1, dtype=torch.int32)
    D[Z] = distance_mat[Z, :][:, U].sum(dim=1, dtype=torch.int32)
    
    T = torch.zeros(initial_solution.shape, dtype=torch.int32)
    num_iter = 0
    num_non_imp_iter = 0
    solution = initial_solution.clone()
    objective_val = initial_objective_val
    
    while num_non_imp_iter < improvement_cutoff:
        D_nt[:] = D
        D_t[:] = D
        pad_val = -10000
        D_nt[T > num_iter] = pad_val
        D_t[T <= num_iter] = pad_val
        
        max_d_nt, UCL_swap_idx_nt, ZCL_swap_idx_nt = determine_swap(D_nt)
        max_d_t, UCL_swap_idx_t, ZCL_swap_idx_t = determine_swap(D_t)
        
        if max_d_t > max_d_nt and max_d_t > 0:
            max_d, UCL_swap_idx, ZCL_swap_idx = max_d_t, UCL_swap_idx_t, ZCL_swap_idx_t
        else:
            max_d, UCL_swap_idx, ZCL_swap_idx = max_d_nt, UCL_swap_idx_nt, ZCL_swap_idx_nt
        
        initial_solution[UCL_swap_idx] = 0
        initial_solution[ZCL_swap_idx] = 1
        initial_objective_val += max_d_nt
        
        D[[UCL_swap_idx, ZCL_swap_idx]] = -D[[UCL_swap_idx, ZCL_swap_idx]] + distance_mat[UCL_swap_idx, ZCL_swap_idx]
        
        U[UCL_swap_idx] = 0
        Z[ZCL_swap_idx] = 0
        
        U_idx_mask = torch.zeros_like(U)
        Z_idx_mask = torch.zeros_like(Z)
        U_idx_mask[UCL_swap_idx] = 1
        Z_idx_mask[ZCL_swap_idx] = 1
        D[U] += distance_mat[U_idx_mask, U] - distance_mat[Z_idx_mask, U]
        D[Z] += distance_mat[Z_idx_mask, Z] - distance_mat[U_idx_mask, Z]
        
        U[ZCL_swap_idx] = 1
        Z[UCL_swap_idx] = 1
        
        tabu_extension = torch.randint(size=(2,), low=min_tabu_extension, high=max_tabu_extension+1, dtype=torch.int32)
        T[[UCL_swap_idx, ZCL_swap_idx]] = num_iter + tabu_extension
        
        if initial_objective_val > objective_val:
            solution[:] = initial_solution
            objective_val = initial_objective_val
            num_non_imp_iter = 0
        else:
            num_non_imp_iter += 1
        
        num_iter += 1
    
    return solution, objective_val

    
def initialize_population(distance_mat: torch.Tensor, num_samples: int, candidate_list_size: int, improvement_cutoff: int, 
                          min_tabu_extension: int, max_tabu_extension: int, population_size: int, max_reinit: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Initializes population of feasible solutions.

    Args:
        distance_mat (torch.Tensor): Distance matrix [num_seq, num_seq].
        num_samples (int): Number of sequences, whose diversity is to be maximized over all sequences.
        candidate_list_size (int): Number of best candidates that are selected in each iteration of Tabu Search.
        improvement_cutoff (int): Number of iterations of Tabu Search without improvement, after which the procedure is stopped.
        min_tabu_extension (int): Minimum number of iterations of Tabu Search, by which a variable is restricted from change.
        max_tabu_extension (int): Maximum number of iterations of Tabu Search, by which a variable is restricted from change.
        population_size (int): Number of feasible solutions that are held in the population.
        max_reinit (int): Number of random reinitializations, after which the procedure ends with the current population size.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Population of feasible solutions [pop_size, num_seq]; Objective values [pop_size].
    """
    
    population = torch.zeros((population_size, distance_mat.shape[0]), dtype=torch.bool)
    population[:, :num_samples] = 1
    population = population[:, torch.randperm(distance_mat.shape[0])]
    objective_values = objective_function(population, distance_mat)
    
    std_mean = torch.std_mean(objective_values.float(), unbiased=True)
    print("Random init avg:", std_mean[1].item(), "+-", std_mean[0].item())
        
    num_succ = 0
    for idx in range(population_size):
        for idx_2 in range(max_reinit):
            
            initial_solution = population[idx].clone()
            initial_objective_val = objective_values[idx].item()
            solution, objective_val = tabu_search(initial_solution, initial_objective_val, distance_mat, candidate_list_size, 
                                                  improvement_cutoff, min_tabu_extension, max_tabu_extension)
            
            if not any(torch.all(population == solution, dim=1)):
                population[num_succ] = solution
                objective_values[num_succ] = objective_val
                num_succ += 1
                break
    
    return population[:num_succ], objective_values[:num_succ]


def rebuild_population(population: torch.Tensor, objective_values: torch.Tensor, distance_mat: torch.Tensor, num_samples: int, 
                       candidate_list_size: int, improvement_cutoff: int, min_tabu_extension: int, max_tabu_extension: int, 
                       perturbation_fraction: float, max_reinit: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Rebuilds population of feasible solutions after several iterations of global and local optimization without improvement.

    Args:
        population (torch.Tensor): Population of feasible solutions [pop_size, num_seq].
        objective_values (torch.Tensor): Objective values [pop_size].
        distance_mat (torch.Tensor): Distance matrix [num_seq, num_seq].
        num_samples (int): Number of sequences, whose diversity is to be maximized over all sequences.
        candidate_list_size (int): Number of best candidates that are selected in each iteration of Tabu Search.
        improvement_cutoff (int): Number of iterations of Tabu Search without improvement, after which the procedure is stopped.
        min_tabu_extension (int): Minimum number of iterations of Tabu Search, by which a variable is restricted from change.
        max_tabu_extension (int): Maximum number of iterations of Tabu Search, by which a variable is restricted from change.
        perturbation_fraction (float): Fraction of selected variables that are perturbed in the process of population rebuilding.
        max_reinit (int): Number of random reinitializations, after which the procedure ends with the current population size.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Population of feasible solutions [pop_size, num_seq]; Objective values [pop_size].
    """
    
    new_population = torch.zeros_like(population)
    new_objective_values = torch.zeros_like(objective_values)
    
    best_solution_idx = torch.argmax(objective_values)
    remaining_solution_indices = [idx for idx in range(population.shape[0]) if idx != best_solution_idx]
    
    new_population[-1] = population[best_solution_idx]
    new_objective_values[-1] = objective_values[best_solution_idx]
    
    num_succ = 0
    for idx in range(population.shape[0]-1):
        old_solution_idx = remaining_solution_indices[idx]
        old_solution = population[old_solution_idx]
        
        assert old_solution.sum() == num_samples
        
        for idx_2 in range(max_reinit):       
            new_solution = old_solution.clone()
            perturbation_mask_1 = torch.bernoulli(torch.full(new_solution.shape, perturbation_fraction)).bool()
            perturbation_mask_1 *= new_solution
            num_perturbations = perturbation_mask_1.sum().item()

            perturbation_mask_2 = torch.arange(new_solution.shape[0], dtype=torch.int64)
            perturbation_mask_2 = perturbation_mask_2[~new_solution]
            perturbation_mask_2 = perturbation_mask_2[torch.randperm(perturbation_mask_2.shape[0])]
            perturbation_mask_2 = perturbation_mask_2[:num_perturbations]

            new_solution[perturbation_mask_1] = 0
            new_solution[perturbation_mask_2] = 1
            
            assert new_solution.sum() == num_samples
            
            new_objective_val = objective_function(new_solution, distance_mat).item()
            new_solution, new_objective_val = tabu_search(new_solution, new_objective_val, distance_mat, candidate_list_size, 
                                                          improvement_cutoff, min_tabu_extension, max_tabu_extension)
            
            if not any(torch.all(new_population == new_solution, dim=1)):
                new_population[num_succ] = new_solution
                new_objective_values[num_succ] = new_objective_val
                num_succ += 1
                break
    
    new_population = torch.cat((new_population[:num_succ], new_population[-1:]))
    new_objective_values = torch.cat((new_objective_values[:num_succ], new_objective_values[-1:]))
    return new_population, new_objective_values


def update_population(population: torch.Tensor, objective_values: torch.Tensor, new_solution: torch.Tensor, 
                      new_objective_val: int) -> Tuple[torch.Tensor, torch.Tensor, bool]:
    """
    Updates the population of feasible solutions by a new solution. It replaces the currently worst solution, if it is better 
    and not a duplicate of an already existent solution.

    Args:
        population (torch.Tensor): Population of feasible solutions [pop_size, num_seq].
        objective_values (torch.Tensor): Objective values [pop_size].
        new_solution (torch.Tensor): New solution [num_seq].
        new_objective_val (int): Objective value of the new solution.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, bool]: Updated population of feasible solutions [pop_size, num_seq]; 
        Updated objective values [pop_size]; Whether population has changed.
    """
    
    uniqueness_criterion = not any(torch.all(population == new_solution, dim=1))
    
    lowest_objective_val, worst_solution_idx = torch.min(objective_values, dim=0)
    performance_criterion = new_objective_val > lowest_objective_val
    
    update_criterion = uniqueness_criterion and performance_criterion
    
    if update_criterion:
        population[worst_solution_idx] = new_solution
        objective_values[worst_solution_idx] = new_objective_val
    
    return population, objective_values, update_criterion


def maximize_diversity_mats(distance_mat: torch.Tensor, num_samples: int, num_iter: int, candidate_list_size: int, improvement_cutoff: int,
                            num_sdv: int, max_reinit: int, process_id: int, min_tabu_extension: int = 15, max_tabu_extension: int = 25, 
                            population_size: int = 10, population_rebuilding_threshold: int = 30, perturbation_fraction: float = 0.3, 
                            verbose: bool = False) -> torch.Tensor:
    """
    Implements the MA/TS algorithm (cf. https://doi.org/10.1016/j.engappai.2013.09.005) to solve a maximum-diversity-problem (MDP), 
    specified by a distance matrix. It is a memetic algorithm, i.e. it extends an evolutionary algorithm (EA) by Tabu-Search-based 
    local optimization. 

    Args:
        distance_mat (torch.Tensor): Distance matrix [num_seq, num_seq].
        num_samples (int): Number of sequences, whose diversity is to be maximized over all sequences.
        num_iter (int): Number of outer iterations performed by the EA.
        candidate_list_size (int): Number of best candidates that are selected in each iteration of Tabu Search.
        improvement_cutoff (int): Number of iterations of Tabu Search without improvement, after which the procedure is stopped.
        num_sdv (int): Number of strongly determined variables.
        max_reinit (int): Number of random reinitializations, after which initialization and rebuilding procedures ends with the current population size.
        process_id (int): Process id.
        min_tabu_extension (int, optional): Minimum number of iterations of Tabu Search, by which a variable is restricted from 
        change. Defaults to 15.
        max_tabu_extension (int, optional): Maximum number of iterations of Tabu Search, by which a variable is restricted from 
        change. Defaults to 25.
        population_size (int, optional): Number of feasible solutions that are held in the population. Defaults to 10.
        population_rebuilding_threshold (int, optional): Number of inner EA iterations without improvement, after which population is 
        rebuilt. Defaults to 30.
        perturbation_fraction (float, optional): Fraction of selected variables that are perturbed in the process of 
        population rebuilding. Defaults to 0.3.
        verbose (bool, optional): Activates verbose output. Defaults to False.

    Returns:
        torch.Tensor: Solution of the MDP [num_seq].
    """
    
    
    population, objective_values = initialize_population(distance_mat, num_samples, candidate_list_size, improvement_cutoff, 
                                                         min_tabu_extension, max_tabu_extension, population_size, max_reinit)
    objective_val_record, solution_record_idx = torch.max(objective_values, dim=0)
    solution_record = population[solution_record_idx]
    
    if verbose:
        print(process_id, ": Initial diversity:", objective_val_record.item())
    
    for idx in range(1, num_iter+1):
        update_non_succ = 0
        while update_non_succ <= population_rebuilding_threshold:
            different_solutions = False
            while not different_solutions:
                solution_indices = torch.randint(size=(2,), high=population.shape[0], dtype=torch.int64)
                different_solutions = solution_indices[0] != solution_indices[1]
            solution_1, solution_2 = population[solution_indices[0]], population[solution_indices[1]]
            new_solution, new_objective_val = recombine_solutions(solution_1, solution_2, distance_mat, num_samples, num_sdv)
            new_solution, new_objective_val = tabu_search(new_solution, new_objective_val, distance_mat, candidate_list_size, 
                                                          improvement_cutoff, min_tabu_extension, max_tabu_extension)
            if new_objective_val > objective_val_record:
                solution_record[:] = new_solution
                objective_val_record = new_objective_val
            population, objective_values, population_changed = update_population(population, objective_values, new_solution, 
                                                                                 new_objective_val)
            if not population_changed:
                update_non_succ += 1
            else:
                update_non_succ = 0
        population, objective_values = rebuild_population(population, objective_values, distance_mat, num_samples, candidate_list_size, 
                                                          improvement_cutoff, min_tabu_extension, max_tabu_extension, 
                                                          perturbation_fraction, max_reinit)
        if verbose:
            print(process_id, ": Diversity after iteration", idx, "/", num_iter, ":", objective_val_record)
    
    return solution_record


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


def maximize_diversity_msa_mats(data: List[torch.Tensor], num_samples: int, num_iter: int, candidate_list_size: int, improvement_cutoff: int, num_sdv: int, 
                                min_tabu_extension: int, max_tabu_extension: int, population_size: int, population_rebuilding_threshold: int, 
                                perturbation_fraction: float, max_reinit: int, verbose: bool, process_id: int) -> List[torch.Tensor]:
    """
    Creates a subset of sequences with maximized diversity for each MSA by solving the corresponding maximum-diversity-problem (MDP) using the MA/TS 
    algorithm.

    Args:
        data (List[torch.Tensor]): List of MSAs [num_seq, len_seq].
        num_samples (int): Number of sequences, whose diversity is to be maximized over all sequences.
        num_iter (int): Number of outer iterations performed by the EA.
        candidate_list_size (int): Number of best candidates that are selected in each iteration of Tabu Search.
        improvement_cutoff (int): Number of iterations of Tabu Search without improvement, after which the procedure is stopped.
        num_sdv (int): Number of strongly determined variables.
        min_tabu_extension (int): Minimum number of iterations of Tabu Search, by which a variable is restricted from 
        change.
        max_tabu_extension (int): Maximum number of iterations of Tabu Search, by which a variable is restricted from 
        change.
        population_size (int): Number of feasible solutions that are held in the population.
        population_rebuilding_threshold (int): Number of inner EA iterations without improvement, after which population is 
        rebuilt.
        perturbation_fraction (float): Fraction of selected variables that are perturbed in the process of 
        population rebuilding.
        max_reinit (int): Number of random reinitializations, after which initialization and rebuilding procedures ends with the current population size.
        verbose (bool): Activates verbose output.
        process_id (int): Process id.

    Returns:
        List[torch.Tensor]: List of sequence indices per MSA [num_samples].
    """

    results = []
    
    for idx, msa in enumerate(data):
        if verbose:
            print(process_id, ": Starting MSA", idx+1, "/", len(data))
        cls_ = candidate_list_size if candidate_list_size > 0 else int(min(num_samples**0.5, (msa.shape[0]-num_samples)**0.5))
        distance_mat = distance_matrix(msa, verbose, process_id)
        solution = maximize_diversity(distance_mat, num_samples, num_iter, cls_, improvement_cutoff, num_sdv, max_reinit, process_id,
                                      min_tabu_extension, max_tabu_extension, population_size, population_rebuilding_threshold, perturbation_fraction,
                                      verbose)
        seq_indices = torch.arange(solution.shape[0], dtype=torch.int64)
        results.append(seq_indices[solution])
        if verbose:
            print(process_id, ": MSA", idx+1, "/", len(data), "finished")
        
    return results


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