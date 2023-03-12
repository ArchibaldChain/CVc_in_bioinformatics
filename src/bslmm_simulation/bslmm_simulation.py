import os
import numpy as np
from pheno_simulation import phenotype_generator as generator, train_test_split
import gemma_cross_validation as gcv
from multiprocessing import Process

bimbam_path = './bimbam_data/bimbam_10000_full_false_major_minor.txt'
error_save_path = './bimbam_data/CVc_error_simulation_bslmm.csv'
num_processes = os.cpu_count()
print("num_processes: ", num_processes)
processes = []


def multi_process_helper(simulation_times=1000):
    # create a list to indicate how many simulation for each processor
    tasks_per_processor = simulation_times // num_processes
    tasks = [tasks_per_processor] * num_processes
    tasks[0] += simulation_times % num_processes
    assert sum(tasks) == simulation_times

    for i, m in enumerate(tasks):
        print(f'Process {i} has {m} tasks.')
        p = Process(target=bslmm_simulation, args=[m])
        processes.append(p)

    for p in processes:
        p.start()


def bslmm_simulation(m=1000,
                     bimbam_dir=bimbam_path,
                     nfolds=10,
                     error_data_dir=error_save_path):
    # create the error file
    if not os.path.exists(error_data_dir):
        with open(error_data_dir, 'w') as f:
            head = [
                'CV_error', 'CV_error_H_cv', 'te_error', 'te_error_h_te', 'w'
            ]
            f.write(','.join(head) + '\n')

    for i in range(m):

        print(f'# Simulation {i} time #')
        # generate the phenotype
        bimbam_data, pheno = generator(bimbam_dir)
        (geno_tr, pheno_tr), (geno_te,
                              pheno_te) = train_test_split(bimbam_data, pheno)

        # simulate the error
        data = gcv.one_time_simulation(geno_tr, geno_te, pheno_tr, pheno_te,
                                       nfolds)

        try:
            with open(error_data_dir, 'a') as f:
                dataline = [str(x) for x in data]
                f.write(','.join(dataline) + '\n')
        except:
            print(data)


if __name__ == '__main__':
    # multi_process_helper(100)
    bslmm_simulation(2)