#!/usr/bin/env python3
#
# Author: Teun Mathijssen (https://github.com/teuncm)
# Description: View all direct bitwise neighbors of arbitrary float values.
# Example: python float_neighbors.py 7.2 -p 16 -n 25 -e ">" | less

import argparse
import numpy as np

def get_hex_repr(float_arr):
    """Convert NumPy float array to hexadecimal representation."""
    return f'0x{float_arr.tobytes().hex()}'

def print_table(args, before_arr, float_arr, after_arr):
    """Print neighbor table."""
    # Adjust hex table size.
    hex_table_size = 10
    if args.p == 128:
        hex_table_size = 34
    if args.p == 64:
        hex_table_size = 18
    
    # Print table header.
    print(f'Float neighbors for "{args.number[0]}"')
    print(f'({args.p}-bit precision, {args.n} neighbors, "{args.e}" endianness)')
    print()
    print(f'{"Offset":>8} | {"Hex value":<{hex_table_size}} | {"Numerical value"}')

    # Print downwards neighbors.
    for i in reversed(range(args.n)):
        before_hex = get_hex_repr(before_arr[i:i+1])
        print(f'{-(i+1):>8} | {before_hex:<{hex_table_size}} | ', end='')
        print(before_arr[i])

    # Print float itself.
    float_hex = get_hex_repr(float_arr)
    print(f'{0:>8} | {float_hex:<{hex_table_size}} | ', end='')
    print(float_arr[0])

    # Print upwards neighbors.
    for i in range(args.n):
        after_hex = get_hex_repr(after_arr[i:i+1])
        print(f'{i+1:>8} | {after_hex:<{hex_table_size}} | ', end='')
        print(after_arr[i])

def main(args):
    # Create custom dtype. C will use this under the hood.
    my_dtype = np.dtype(f'{args.e}f{args.p//8}')
    
    # Always stay within NumPy array interface to maintain endianness 
    # and precision. These get lost in Python float handling.
    float_arr = np.array(args.number, my_dtype)

    # Walk downwards.
    before_arr = np.full(args.n, float_arr[0], my_dtype)
    NINF = np.array([np.NINF], my_dtype)[0]
    for i in range(args.n):
        if i != 0:
            before_arr[i] = before_arr[i-1]
        before_arr[i] = np.nextafter(before_arr[i], NINF)

    # Walk upwards.
    after_arr = np.full(args.n, float_arr[0], my_dtype)
    PINF = np.array([np.PINF], my_dtype)[0]
    for i in range(args.n):
        if i != 0:
            after_arr[i] = after_arr[i-1]
        after_arr[i] = np.nextafter(after_arr[i], PINF)

    print_table(args, before_arr, float_arr, after_arr)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Display a float and its direct possible neighbors.')
    parser.add_argument('number', type=str, nargs=1, 
        help='Any of the following: {"x", "1ex", "inf"}. Specify \
            negatives with a leading space.')
    parser.add_argument('-p', metavar='precision', type=int, default=64, 
        help='Float precision. Any of {16, 32, 64, 128}. Default: 64(-bit).')
    parser.add_argument('-n', metavar='neighbors', type=int, default=5, 
        help='Number of neighbors. Default: 5.')
    parser.add_argument('-e', metavar='endianness', type=str, default='>', 
        help='Endianness. Any of {"<", ">", "="}. Default: ">" (big-endian).')

    main(parser.parse_args())