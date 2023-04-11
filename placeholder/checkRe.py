import os
import numpy as np

dirs = []

for d in os.scandir('./'):
    if d.is_dir():

        dirs.append(d.path)

for folder in dirs:

    with open(f'{folder}/constant/transportProperties', 'r') as f:

        lines = f.readlines()

    f.close()

    for line in lines:

        value = line.split()

        if value != []:

            if value[0] == 'nu':

                print(folder)
            
                try:
                    nu = float(value[1][:len(value[1])-1])
                    print(f'{0.2/nu:.3e}')
                except ValueError:
                    raise Exception(f'nu is: {float(value[1][:len(value)-1])}')

