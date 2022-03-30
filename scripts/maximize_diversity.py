import argparse
from itertools import chain, repeat
import itertools
import multiprocessing as mp
import os
import torch
from typing import Any, List

from selbstaufsicht import datasets
from selbstaufsicht.models.self_supervised.msa.transforms import MSATokenize
from selbstaufsicht.models.self_supervised.msa.diversity_maximization import maximize_diversity_msa
from selbstaufsicht.utils import rna2index


def distribute_data(data: List[Any], num_jobs: int) -> List[List[Any]]:
    len_data_per_job = len(data) // num_jobs
    len_rest_data = len(data) % num_jobs
    
    distributed_data = [data[idx*(len_data_per_job+1):(idx+1)*(len_data_per_job+1)] for idx in range(len_rest_data)]
    offset = len_rest_data*(len_data_per_job+1)
    distributed_data += [data[offset+idx*len_data_per_job:offset+(idx+1)*len_data_per_job] for idx in range(num_jobs-len_rest_data)]

    return distributed_data


def main():
    parser = argparse.ArgumentParser(description='Selbstaufsicht MSA Diversity Maximization Script')
    # Diversity maximization
    parser.add_argument('--num-samples', default=50, type=int, help="Number of samples whose diversity is to be maximized.")
    parser.add_argument('--num-iter', default=5, type=int, help="Number of outer iterations performed by the evolutionary algorithm.")
    parser.add_argument('--cls', default=-1, type=int, help="Number of best candidates that are selected in each iteration of Tabu Search. If non-positive, sqrt-rule is used.")
    parser.add_argument('--imp-cutoff', default=300, type=int, help="Number of iterations of Tabu Search without improvement, after which the procedure is stopped.")
    parser.add_argument('--num-sdv', default=35, type=int, help="Number of strongly determined variables.")
    parser.add_argument('--min-te', default=15, type=int, help="Minimum number of iterations of Tabu Search, by which a variable is restricted from change.")
    parser.add_argument('--max-te', default=25, type=int, help="Maximum number of iterations of Tabu Search, by which a variable is restricted from change.")
    parser.add_argument('--pop-size', default=10, type=int, help="Number of feasible solutions that are held in the population.")
    parser.add_argument('--pop-rt', default=30, type=int, help="Number of inner iterations of the evolutionary algorithm without improvement, after which population is rebuilt.")
    parser.add_argument('--perturb-frac', default=0.3, type=float, help="Fraction of selected variables that are perturbed in the process of population rebuilding.")
    parser.add_argument('--max-reinit', default=50, type=int, help="Number of random reinitializations, after which initialization and rebuilding procedures ends with the current population size.")
    # Data parallelism
    parser.add_argument('--num-jobs', default=1, type=int, help="Number of jobs.")
    # Logging
    parser.add_argument('--log-path', default='../selbstaufsicht/datasets/coconet_diversity_maximization.pt', type=str, help='Logging path.')
    parser.add_argument('--verbose', action='store_true', help="Activates verbose output.")
    # Profiling
    parser.add_argument('--profile', action='store_true', help="Activates profiling.")
    parser.add_argument('--profile-num-seq', default=15000, type=int, help="Number of random sequences in profiling.")

    args = parser.parse_args()
    
    if args.profile:
        data = [[torch.randint(size=(args.profile_num_seq, 400), high=20)] for idx in range(args.num_jobs)]
    else:
        root = os.environ['DATA_PATH']
        data = datasets.CoCoNetDataset(root, 'train', transform=MSATokenize(rna2index), discard_train_size_based=True)
        data = [data[idx][0]['msa'] for idx in range(len(data))]
        data = distribute_data(data, args.num_jobs)  # [num_jobs, num_msa_per_job, num_seq, len_seq]
    
    mp_pool = mp.Pool(processes=args.num_jobs)
    pool_args = zip(data, repeat(args.num_samples), repeat(args.num_iter), repeat(args.cls), repeat(args.imp_cutoff), 
                    repeat(args.num_sdv), repeat(args.min_te), repeat(args.max_te), repeat(args.pop_size), 
                    repeat(args.pop_rt), repeat(args.perturb_frac), repeat(args.max_reinit), repeat(args.verbose),
                    repeat(args.profile))
    results = mp_pool.starmap(maximize_diversity_msa, pool_args)  # [num_jobs, num_msa_per_job, num_samples]
    results = list(chain.from_iterable(results))  # [num_jobs*num_msa_per_job, num_samples]
    results = torch.stack(results)  # [num_jobs*num_msa_per_job, num_samples]
    torch.save(results, args.log_path)
    
    
if __name__ == '__main__':
    main()