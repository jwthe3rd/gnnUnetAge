import numpy as np
import pandas as pd
from collections import defaultdict

DATA_FILE = 'test_results.txt'


def data_curation(data_lines, data_type):

    data = []

    for line in data_lines:

        t = line.split()

        print(t[0])

        if t[0] == data_type:
            data.append(t[2])

    return data




if __name__ == "__main__":

    DATA_TYPES = ["Case", "Re", "BaffleSize", 
                  "Double", "Loss", "Acc"]
    
    DATA_DICT = defaultdict(int)

    with open(DATA_FILE, 'r') as f:

        file_lines = f.readlines()

        for dataType in DATA_TYPES:

            data_pts = data_curation(data_lines=file_lines, data_type=dataType)

            DATA_DICT[dataType] = data_pts

    print(DATA_DICT)

    df = pd.DataFrame(data=DATA_DICT)
    df.to_csv('test_results.csv')
