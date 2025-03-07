from neuron import h
from neuron.units import ms, mV
import matplotlib.pyplot as plt
from ring import Ring
import time
import pandas as pd

start_time = time.time()

# Create a distributed network (each MPI process handles its own set of cells)
ring = Ring(N=100)

# Initialize the ParallelContext for MPI communication
pc = h.ParallelContext()
pc.set_maxstep(10 * ms)  # maximum allowed time step for psolve

# Record the simulation time
t = h.Vector().record(h._ref_t)

# Initialize the simulation
h.finitialize(-65 * mV)
# Run the simulation in parallel using pc.psolve, which synchronizes processes
pc.psolve(100 * ms)

# send all spike time data to node 0
# Each process gathers spike times for its cells
local_data = {cell._gid: list(cell.spike_times) for cell in ring.cells}
# Use MPI all-to-all communication to send each process's local data to all processes
all_data = pc.py_alltoall([local_data] + [None] * (pc.nhost() - 1))

# Each process computes its local cell count
local_cell_count = len(ring.cells)

# Gather the cell counts from all processes.
# Create a list of length pc.nhost(), where each process provides its local count.
cell_counts = pc.py_alltoall([local_cell_count] + [None] * (pc.nhost() - 1))

if pc.id() == 0: # Only rank 0 combines the data and performs plotting
    # combine the data from the various processes
    data = {}
    for process_data in all_data:
        data.update(process_data)
    # plot it
    plt.figure()
    for i, spike_times in data.items():
        plt.vlines(spike_times, i + 0.5, i + 1.5)
    plt.show()
    plt.savefig(f'neuron_ring_results/{len(ring.cells)}_cells.png')

    total_cells = sum(cell_counts)
    average_cells = total_cells / pc.nhost()
    print("Cell counts per worker:", cell_counts)
    print("Average number of cells per worker:", average_cells)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time}")

    location = 'cloudlab'

    # print(f"pc.nhost(): {pc.nhost()}")
    # print(f"len(ring.cells): {len(ring.cells)}")
    # print(f"location: {location}")

    df = pd.DataFrame({
        'Time': [elapsed_time],
        'avg_num_cells': [average_cells],
        'Location': [location],
        'parallel workers': [pc.nhost()]
    })
    df.to_csv(f"neuron_ring_results/{location}_{pc.nhost()}workers.csv", index=False)

pc.barrier()
pc.done()
h.quit()