import statistics as stats
import numpy as np
import os
def find_max_iter(dir):

    numbers = []

    for d in os.scandir(dir):
        if d.is_dir():
            try:
                numbers.append(int(d.name))
            except:
                continue

    return f'{max(numbers)}/'

def load_feature_scalar(case, dir,file, norm=False):
    feature_Values = []

    with open(case+dir+file) as f:
        lines = f.readlines()

    feature_data_raw = lines[21:]

    num_of_cells = int(feature_data_raw[0])

    feature_data_raw = feature_data_raw[2:num_of_cells+2]

    feature_Values = np.asarray(feature_data_raw, dtype=float)

    if norm:

        norm_vals = np.asarray([0, 2, 5, 10, 15, 20, 25, 30, 40, 50])
        cats = norm_vals*22.5

        for i, val in enumerate(feature_Values):
            tst = val / 22.5
            categorized = False
            for j, cat in enumerate(norm_vals):
                if tst > cat:
                    feature_Values[i] = cats[j]
                    categorized = True
            if not categorized:
                feature_Values[i] = cats[-1]

    return feature_Values

if __name__=="__main__":

    tests = ['0_8_2-00740_9/','0_54_445_6/','dbl_0_4_485_2/']

    for test_file in tests:

        max_dir = find_max_iter(test_file)
        age_data = load_feature_scalar(test_file, dir=max_dir, file='age')
        norm_age_data = load_feature_scalar(test_file, dir=max_dir, file='age', norm=True)
        pred_age_data = load_feature_scalar(test_file, dir=max_dir, file='age_norm_pred')

        print(f'{max(norm_age_data)}')
        print(f'{max(age_data) / 22.5} | {max(age_data)}')

        print(f'Var Age: {np.sqrt(stats.variance(age_data))} | Var Norm Age: {np.sqrt(stats.variance(norm_age_data))} | Var Pred Age: {np.sqrt(stats.variance(pred_age_data)) / 90}')
        print(f'Error : {100*(np.sqrt(stats.variance(age_data)) - np.sqrt(stats.variance(norm_age_data))) / np.sqrt(stats.variance(age_data)):.2f}')