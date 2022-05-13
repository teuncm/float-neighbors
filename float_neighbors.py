#!/usr/bin/env python3
#
# Author: Teun Mathijssen (https://github.com/teuncm)
# Description: Explore direct neighbors and limits of IEEE floating-point values.

import argparse
import numpy as np

def from_hex(hex, my_dtype):
    """Convert hexadecimal representation to NumPy float array."""
    binary = bytes.fromhex(hex[2:])
    return np.frombuffer(binary, dtype=my_dtype)

def to_hex(float_arr):
    """Convert NumPy float array to hexadecimal representation."""
    return f'0x{float_arr.tobytes().hex()}'

def print_table(args, before_arr, float_arr, after_arr, finfo):
    """Print neighbor table."""
    # Adjust hex table size.
    hex_table_size = 10
    if args.p == 128:
        hex_table_size = 34
    if args.p == 64:
        hex_table_size = 18

    # Print table header.
    print(f'Analyzing float neighbors for input "{args.number}"')
    print()
    print(f'Data: {args.p}-bit precision, {args.n} neighbors, "{args.e}" endianness')
    print(f'Bits: 1 (sign), {finfo.nexp} (exponent), {finfo.nmant} (fraction)')
    print()
    print(f'{"Offset":>8} | {"Hex value":<{hex_table_size}} | {"Numerical value"}')

    # Print downwards neighbors.
    for i in reversed(range(args.n)):
        before_hex = to_hex(before_arr[i:i+1])
        print(f'{-(i+1):>8} | {before_hex:<{hex_table_size}} | ', end='')
        print(before_arr[i])

    # Print float itself.
    float_hex = to_hex(float_arr)
    print(f'{"0":>8} | {float_hex:<{hex_table_size}} | ', end='')
    print(float_arr[0])

    # Print upwards neighbors.
    for i in range(args.n):
        after_hex = to_hex(after_arr[i:i+1])
        print(f'{i+1:>+8} | {after_hex:<{hex_table_size}} | ', end='')
        print(after_arr[i])

def main(args):
    # Create custom dtype. C will use this under the hood.
    my_dtype = np.dtype(f'{args.e}f{args.p//8}')

    # Retrieve machine info for this custom dtype.
    finfo = np.finfo(my_dtype)

    # Always stay within NumPy array interface to maintain endianness
    # and precision. These get lost in Python float handling.
    if args.number[:2] == '0x':
        float_arr = from_hex(args.number, my_dtype)
    else:
        float_arr = np.array([args.number], my_dtype)

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

    print_table(args, before_arr, float_arr, after_arr, finfo)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Display a float and its direct possible neighbors.')
    parser.add_argument('number', type=str,
        help='Any of the following: {"<x>", "1e<y>", "0x<z>", "inf"}. Specify \
            negatives with a leading space.')
    parser.add_argument('-p', metavar='precision', type=int, default=64,
        choices=[16, 32, 64, 128],
        help='Float precision. Any of {16, 32, 64, 128}. Default: 64(-bit).')
    parser.add_argument('-n', metavar='neighbors', type=int, default=5,
        help='Number of neighbors. Default: 5.')
    parser.add_argument('-e', metavar='endianness', type=str, default='>',
        choices=['<', '>', '='],
        help='Endianness. Any of {"<", ">", "="}. Default: ">" (big-endian).')

    main(parser.parse_args())