import argparse
import numpy as np

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('-num_iterations', type=int, default=1, help='num_iterations')
    parser.add_argument('-x_value', type=float, default=1.0, help='x_value')
    parser.add_argument('-start', type=int, default=0, help='x_value')

    args, _ = parser.parse_known_args()

    return args

def factorial(n):

    if n == 0:
        return 1
    else:
        return n*factorial(n-1)

def function(x, n):

    return (x**n) / factorial(n) 

def ideal(x):

    return np.exp(x)   

def loop_and_add(ideal, function, n, start, x):

    value = 0

    for i in range(start, n):

        value += function(x, i)

    print(f'Actual Value = {ideal(x)} ; Value with n @ {n} steps = {value}')

    return 0


def main():

    args = get_args()

    loop_and_add(ideal, function, n=args.num_iterations, start=args.start, x=args.x_value)

if __name__ == "__main__":

    main()
