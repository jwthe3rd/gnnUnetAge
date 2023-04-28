import numpy as np

data_file = 'var_log3'


with open(data_file, 'r') as f:

    lines = f.readlines()

f.close()

#lines = lines[2:]

cases = []
Re = []
BaffleSize = []
Double = []
Loss = []
Acc = []
errors = []
exp = []

for line in lines:

    line_split = line.split()

    if line_split[0] == "Case":

        cases.append(line_split[2])
    
    elif line_split[0] == "Test":

        cases.append(line_split[2][:len(line_split[2])-1])

    elif line_split[0] == "Re":

        Re.append(line_split[2])

    elif line_split[0] == "BaffleSize":

        BaffleSize.append(line_split[2])

    elif line_split[0] == "Double":

        Double.append(line_split[2])

    elif line_split[0] == "Loss":

        Loss.append(line_split[2])

    elif line_split[0] == "Acc":

        Acc.append(line_split[2])
    
    elif line_split[0] == "Error":

        errors.append(line_split[2])

    elif line_split[0] == "Expected":

        exp.append(line_split[3])


    else:
        print("Wahhhhh")

with open("raw_test_results_var.csv", 'w') as r:

    #r.writelines("Case, Re, BaffleSize, Double, Loss, Acc\n")
    r.writelines("Case,Error,Expected Error\n")
    print(exp)
    for i, val in enumerate(cases):

        #r.writelines(f'{val}, {Re[i]}, {BaffleSize[i]}, {Double[i]}, {Loss[i]}, {Acc[i]}\n')
        r.writelines(f'{val},{errors[i]},{exp[i]}\n')

r.close()
