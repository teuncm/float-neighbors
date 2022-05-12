## Float neighbor exploration
View all direct bitwise neighbors of arbitrary float values. This program can be used as educational tool to delve deep into float rounding errors and explore the properties and limits of IEEE floats.

### Configurable options
- Precision: 16-bit, 32-bit, 64-bit, 128-bit*
- Number of neighbors
- Endianness: little-endian, big-endian, machine-default

*: machine-dependent

### Example usage
```bash
./float_neighbors.py 7.2 -p 16 -n 25 -e ">" | less
```
```bash
./float_neighbors.py " -inf" -p 32 -e "<"
```
```bash
./float_neighbors.py 0
```

### Example output
```bash
./float_neighbors.py 2 -p 16 -n 2
```

```
  Offset | Hex value  | Numerical value
      -2 | 0x3ffe     | 1.998
      -1 | 0x3fff     | 1.999
       0 | 0x4000     | 2.0
       1 | 0x4001     | 2.002
       2 | 0x4002     | 2.004
```

### Technical details and challenges
Python floats are internally 64 bits long and use machine-default endianness. By only using NumPy arrays for data, we circumvent any internal Python conversions. Furthermore, we are now able to leverage custom NumPy datatypes.

A custom NumPy float datatype could for example be `np.dtype('>f4")` where `>` indicates big-endianness, `f` indicates float storage and `4` indicates that 2 bytes are used for storage.

Obtaining the bitwise next float is not trivial. Luckily, `np.nextafter/2` is implemented in the NumPy specification, which returns the direct next possible float neighbor (towards `np.NINF` and `np.PINF`) and handles all edge cases.

The infinity values used for retrieving neighbors have to be treated with special care: they are put in an array with the custom dtype and subsequently retrieved to maintain precision and byte order.

Runtime complexity of the program is `O(n)` as the previous float is always used to calculate the next float.

Converting a float into its hex representation is not trivial. Luckily, NumPy provides a method to convert an entire NumPy array to a bytes object: `tobytes/0`. We use this method on a NumPy array containing a single float and then use Python's `hex/0` to convert it into hex format.

Printing a 128-bit float directly within an f-string is not possible: internally Python will convert the number to a 64-bit float. Hence the print statements for the numerical values are on separate lines: this will circumvent conversion and properly print the float to its full precision.

#### Summary
This was an educational project to combine the flexibility and elegance of Python with the low-level power of C provided by the NumPy interface. The main challenge was to maintain precision and endianness of the stored data while performing intermediate operations.