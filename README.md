## Float neighbor exploration
Explore direct neighbors and limits of IEEE floating-point values. This program can be used as educational tool to delve deep into float rounding errors, epsilons and explore the properties and limits of IEEE floats.

### Configurable options
- Precision: 16-bit, 32-bit, 64-bit, 128-bit*
- Number of neighbors
- Endianness: little-endian, big-endian, machine-default

*: [machine-dependent](https://numpy.org/doc/stable/user/basics.types.html#extended-precision) (usually padded or unavailable)

### Example usage
`python float_neighbors.py 0x0000 -p 16 -e "<" -n 2`
```
   Offset | Hex value  | Numerical value
       +2 | 0x0200     | 1e-07
       +1 | 0x0100     | 6e-08
       +0 | 0x0000     | 0.0
       -1 | 0x0180     | -6e-08
       -2 | 0x0280     | -1e-07
```

`python float_neighbors.py " -inf" -p 32 -n 2`
```
   Offset | Hex value  | Numerical value
       +2 | 0xff7ffffe | -3.4028233e+38
       +1 | 0xff7fffff | -3.4028235e+38
       +0 | 0xff800000 | -inf
```

`python float_neighbors.py 2 -p 16 -n 2`
```
   Offset | Hex value  | Numerical value
       +2 | 0x4002     | 2.004
       +1 | 0x4001     | 2.002
       +0 | 0x4000     | 2.0
       -1 | 0x3fff     | 1.999
       -2 | 0x3ffe     | 1.998
```

### Technical details and design challenges
Python floats are internally 64 bits in size and use machine-default endianness. By only using NumPy arrays for data, we circumvent any internal Python conversions. Furthermore, we are now able to leverage custom NumPy datatypes.

A custom NumPy float datatype could for example be `np.dtype('>f4')` where `>` indicates big-endianness, `f` indicates float storage and `4` indicates that 2 bytes are used for storage.

Obtaining the next closest numerical float value is not trivial. Fortunately, `np.nextafter/2` is implemented in the NumPy specification, which returns the direct next possible float neighbor (towards `np.NINF` and `np.PINF`) and handles edge cases when passing through any float category:

`inf > normal > subnormal > 0.0 > -subnormal > -normal > -inf`

The infinity values used for retrieving neighbors have to be treated with special care: they are put in an array with the custom dtype and subsequently retrieved to maintain precision and byte order.

Runtime complexity of the program is `O(n)` as the previous float is always used to calculate the next float.

Memory complexity of the program is `O(1)` as the program initially walks up and subsequently walks down to calculate and display all the floats one by one in descending order.

Converting a float into its hex representation is not trivial. Luckily, NumPy provides a method to convert an entire NumPy array to a bytes object: `tobytes/0`. We use this method on a NumPy array containing a single float and then use Python's `hex/0` to convert it into hex format.

Converting a hex representation into a float is done in two steps. We can simply use `bytes.fromhex/1` to obtain the bytes representation and then rebuild the (1-element) float array using `np.frombuffer/2` with our custom dtype.

Printing a 128-bit float directly within an f-string is not possible: internally Python will convert the number to a 64-bit float. Hence the print statements for the numerical values are on separate lines: this will circumvent conversion and properly print the float to its full precision.

#### Summary
This was an educational project to combine the flexibility and elegance of Python with the low-level power of C provided by the NumPy interface. The main challenge was to maintain precision and endianness of the stored data while performing intermediate operations.
