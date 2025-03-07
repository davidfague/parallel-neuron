# requires mpi4py: pip install mpi4py
# intended use: python cs_4001_matrix_multiplication "{location}" "[True, False]"
# saves a csv of results

from mpi4py import MPI
import numpy as np
import time
import sys
import pandas as pd
import ast

# Function to perform matrix multiplication
def matrix_multiply(A, B):
    C = np.zeros((A.shape[0], B.shape[1]))
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            for k in range(A.shape[1]):
                C[i][j] += A[i][k] * B[k][j]
    return C

if __name__ == "__main__":
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    parallel_options = ast.literal_eval(sys.argv[2])
    location = sys.argv[1]
    Ns = [10,100,500,1000]
    times = []
    parallels = []
    locations = []
    mpi_sizes = []
    for parallel in parallel_options:
        for N in Ns:
            start_time = time.time()
            if parallel:
                # Master process
                if rank == 0:
                    # Generate matrices A and B
                    A = np.random.rand(N, N)
                    B = np.random.rand(N, N)
        
                    # Split matrices for distribution
                    chunk_size = A.shape[0] // size
                    A_chunks = [A[i:i+chunk_size] for i in range(0, A.shape[0], chunk_size)]
        
                    # Send parts of A and B to worker processes
                    for i in range(1, size):
                        comm.send(A_chunks[i-1], dest=i, tag=1)
                        comm.send(B, dest=i, tag=2)
        
                    # Calculate its own part of multiplication
                    C_partial = matrix_multiply(A_chunks[0], B)
        
                    # Collect results from worker processes
                    for i in range(1, size):
                        C_partial += comm.recv(source=i, tag=3)
        
                    # Print the resulting matrix
                    # print("Resulting matrix C:")
                    # print(C_partial)
                # Worker processes
                else:
                    # Receive matrix chunks from master
                    A_chunk = comm.recv(source=0, tag=1)
                    B = comm.recv(source=0, tag=2)
        
                    # Perform multiplication
                    C_partial = matrix_multiply(A_chunk, B)
        
                    # Send back the result to master
                    comm.send(C_partial, dest=0, tag=3)
            else:
                # Generate matrices A and B
                A = np.random.rand(N, N)
                B = np.random.rand(N, N)
                C = matrix_multiply(A, B)
            end_time = time.time()
            times.append(end_time - start_time)
            parallels.append(parallel)
            locations.append(location)
            mpi_sizes.append(size)
            print(f"Execution time: {end_time - start_time} seconds for N = {N}")
  
    # dataframe of results
    df = pd.DataFrame({'NxN': Ns*len(parallel_options), 'Time': times, 'Parallelization':parallels, 'Location':locations, 'mpi_size':mpi_sizes})    
    df.to_csv(f"Results_{location}.csv", index=False)
