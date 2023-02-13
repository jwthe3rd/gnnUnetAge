import torch
import numpy as np
import pandas as pd
import argparse

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('-direct', type=str, default='./', help='direct')
    args, _ = parser.parse_known_args()

    return args

def read_edge_index_from_of(filename: str, working_dir: str) -> torch.tensor:
    """
    The purpose of this function is to read the edge file created by the edgeMat solver in OpenFOAM,
    then output a tensor which is the edge list. This is essential for the graph neural network

    Args:

        filename:
            Name of the file output from the edgeMat solver
        working_dir:
            Directory of the OpenFOAM case

    Returns:

        This returns the edge list in the form of a torch tensor

    Raises:

        N/A

    """
# Uses pandasto read the edgeMa output file, skipping the header
    edge_dataFrame = pd.read_csv(f'{working_dir}/{filename}',header=None,sep='\t', skiprows=34)

# Defines the order of the edge matrix, namely it will go (home cell, adjacent cell) (e.g (0,35))
    home_cell = edge_dataFrame[0]
    adj_cell = edge_dataFrame[1]

# Conversion of disparate lists into one matrix, followed by some type changes
    edge_index = [home_cell, adj_cell]

    edge_index = np.asarray(edge_index)

    edge_index = torch.tensor(edge_index, dtype=torch.long)

    return edge_index


if __name__ == "__main__":

    args = get_args()
    direct = args.direct

    ei = read_edge_index_from_of('edges', direct)
    print(ei)
    torch.save(ei, f'{direct}/e_{direct}.pt')