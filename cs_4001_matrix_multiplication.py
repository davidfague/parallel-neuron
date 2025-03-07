from mpi4py import MPI
import numpy as np
import time
import sys
import pandas as pd
import ast

def matrix_multiply(A, B):
    C = np.zeros((A.shape[0], B.shape[1]))
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            for k in range(A.shape[1]):
                C[i][j] += A[i][k] * B[k][j]
    return C

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    parallel_options = ast.literal_eval(sys.argv[-1])
    location = sys.argv[-2]
    Ns = [10, 100, 500, 1000]
    
    # Only master will collect results for output.
    times = []
    parallels = []
    locations = []
    mpi_sizes = []
    
    for parallel in parallel_options:
        for N in Ns:
            start_time = time.time()
            if parallel:
                if rank == 0:
                    # Master process: generate matrices and send parts to workers
                    A = np.random.rand(N, N)
                    B = np.random.rand(N, N)
                    chunk_size = A.shape[0] // size
                    A_chunks = [A[i:i+chunk_size] for i in range(0, A.shape[0], chunk_size)]
                    
                    # Send chunks to workers
                    for i in range(1, size):
                        comm.send(A_chunks[i-1], dest=i, tag=1)
                        comm.send(B, dest=i, tag=2)
                    
                    # Compute master's chunk
                    C_partial = matrix_multiply(A_chunks[0], B)
                    
                    # Gather results from workers
                    for i in range(1, size):
                        C_partial += comm.recv(source=i, tag=3)
                else:
                    # Worker processes: receive data, compute, and send result back
                    A_chunk = comm.recv(source=0, tag=1)
                    B = comm.recv(source=0, tag=2)
                    C_partial = matrix_multiply(A_chunk, B)
                    comm.send(C_partial, dest=0, tag=3)
            else:
                # Serial computation if parallel is False
                A = np.random.rand(N, N)
                B = np.random.rand(N, N)
                C = matrix_multiply(A, B)
            end_time = time.time()
            
            # Only record times and print output on master
            if rank == 0:
                times.append(end_time - start_time)
                parallels.append(parallel)
                locations.append(location)
                mpi_sizes.append(size)
                print(f"Execution time: {end_time - start_time} seconds for N = {N}")
    
    # Only master writes the CSV file
    if rank == 0:
        df = pd.DataFrame({
            'NxN': Ns * len(parallel_options),
            'Time': times,
            'Parallelization': parallels,
            'Location': locations,
            'mpi_size': mpi_sizes
        })
        df.to_csv(f"Results_{location}.csv", index=False)
        print(f"Results written to Results_{location}.csv")