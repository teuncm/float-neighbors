#!/usr/bin/env python3
#
# Author: Teun Mathijssen
# URL: https://github.com/teuncm/float-neighbors
# Description: Explore direct neighbors and limits of IEEE floating-point values.

import argparse
import numpy as np

def from_hex(hex, my_dtype):
    """Convert hexadecimal representation to NumPy float array."""
    binary = bytes.fromhex(hex[2:])
    float_arr = np.array(np.frombuffer(binary, dtype=my_dtype))
    float_arr.setflags(write=True)

    return float_arr

def to_hex(float_arr):
    """Convert NumPy float array to hexadecimal representation."""
    return f'0x{float_arr.tobytes().hex()}'

def get_hex_col_width(args):
    """Get hex column width for printing."""
    hex_col_width = 10

    if args.p == 64:
        hex_col_width = 18
    if args.p == 128:
        hex_col_width = 34

    return hex_col_width

def print_table_header(args, finfo):
    """Print neighbor table header."""
    hex_col_width = get_hex_col_width(args)

    # Print table header.
    print(f'Analyzing float neighbors for input "{args.number}"')
    print()
    print(f'Data: {args.p}-bit precision, {args.n} neighbors, "{args.e}" endianness')
    print(f'Bits: 1 (sign), {finfo.nexp} (exponent), {finfo.nmant} (fraction)')
    print()
    print(f'{"Offset":>9} | {"Hex value":<{hex_col_width}} | {"Numerical value"}')

def print_table(args, float_arr, iter_start, NINF):
    """Print neighbor table."""
    hex_col_width = get_hex_col_width(args)

    # Walk downwards.
    for i in range(iter_start, -args.n - 1, -1):
        after_hex = to_hex(float_arr)
        print(f'{i:>+9} | {after_hex:<{hex_col_width}} | ', end='')
        print(float_arr[0])

        if np.equal(float_arr[0], NINF):
            # If we hit -inf, we stop here.
            break

        float_arr[0] = np.nextafter(float_arr[0], NINF)

def main(args):
    # Overflow errors can occur.
    np.seterr(over='ignore')

    # Create custom dtype. C will use this under the hood.
    my_dtype = np.dtype(f'{args.e}f{args.p//8}')

    # Retrieve machine info for this custom dtype.
    finfo = np.finfo(my_dtype)

    # Convert to NumPy array to maintain endianness and precision.
    if args.number[:2] == '0x':
        # Hex conversion is not directly supported in NumPy.
        float_arr = from_hex(args.number, my_dtype)
    else:
        float_arr = np.array([args.number], my_dtype)

    # Cache infinity values.
    PINF = np.array([np.PINF], my_dtype)[0]
    NINF = np.array([np.NINF], my_dtype)[0]

    iter_start = args.n

    # Walk upwards.
    for i in range(args.n):
        if np.equal(float_arr[0], PINF):
            # If we hit +inf, we will start from here.
            iter_start = i
            break

        float_arr[0] = np.nextafter(float_arr[0], PINF)

    print_table_header(args, finfo)
    print_table(args, float_arr, iter_start, NINF)

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
