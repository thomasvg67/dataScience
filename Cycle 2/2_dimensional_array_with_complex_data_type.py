
import numpy as np

arr2d = np.array([
	[1 + 2j, 3 + 4j, 5 + 6j],
	[7 + 8j, 9 + 10j, 11 + 12j]
],dtype=complex)

print("2D Complex Array:")
print(arr2d)

rows, cols= arr2d.shape
print("\nNumber of rows:",rows)
print("Number of columns:",cols)

print("\nDimension of the Array:",arr2d.ndim)

reshaped_array = arr2d.reshape((3,2))
print("\nReshaped Array (3x2):")
print(reshaped_array)
