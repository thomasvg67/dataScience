# dataScience
## Cycle 1
### 1. Program to Print All Non-Prime Numbers in an Interval

```python
a = int(input("Enter starting range: "))
b = int(input("Enter ending range: "))

def is_prime(num):
    if num < 2:
        return False
    for i in range(2, (num // 2) + 1):
        if num % i == 0:
            return False
    return True

for num in range(a, b + 1):
    if not is_prime(num):
        print(num)
```
### 2. Program to Print the First N Fibonacci Numbers

```python
n = int(input("Enter the number of Fibonacci series terms: "))
a = 0
b = 1
for i in range(n):
    print(a)
    c = a + b
    a = b
    b = c
```
### 3. Program to find the roots of a quadratic equation(rounded to 2 decimal places)

```python
import math

print("Enter the values of a, b, c in (ax^2 + bx + c) : ")
a = int(input("Enter the value of a : "))
b = int(input("Enter the value of b : "))
c = int(input("Enter the value of c : "))

d = (b**2 - 4*a*c)

if d > 0:
	root1 = (-(b) + math.sqrt(d)) / (2*a)
	root2 = (-(b) - math.sqrt(d)) / (2*a)
	print(f"Roots are real and different\nRoot 1 : {root1:.2f}\nRoot 2 : {root2:.2f}")
elif d < 0:
	real = b/2*a
	imag = math.sqrt(-1 * d) / (2 * a)
	print(f"Roots are complex and different\nRoot 1 : {real:.2f} + {imag:.2f}i\nRoot 2 : {real:.2f} - {imag:.2f}i")
else:
	root = -b / (2 * a)
	print(f"Roots are real and same\nRoot : {root:.2f}")
```

### 4. program to check weather a given number is perfect or not (sum of factors = num).

```python

num = int(input("Enter a number : "))

def is_perfect(num):
    sum = 0
    for i in range(1, num):
        if num // i == num / i:
            sum += i
    return sum == num

if is_perfect(num):
    print(f"Number {num} is perfect")
else:
    print(f"Number {num} is not perfect")

```

### 5. program to display armstrong number up-to 1000.

```python

for num in range(1, 1001):
    sum = 0
    temp = num

    while temp > 0:
        remainder = temp % 10
        sum += remainder ** len(str(num))
        temp //= 10
    if sum == num:
        print(num)
```

### 6. program to perform bubble sort on a given set of elements.

```python

n = int(input("Enter the number of terms : "))
a = []
for i in range(0, n):
    a.append(int(input(f"Enter number {i+1} : ")))
print("List before sorting : ", a)

for i in range(0, n-1):
    for j in range(0, n-i-1):
        if a[j] > a[j+1]:
            temp = a[j+1]
            a[j+1] = a[j]
            a[j] = temp

print("Bubble sorted list is : ", a)
```

### 7. program to accept a 10 digit mobile number and find the digits which are absent in it.

```python

num = int(input("Enter a mobile number : "))
numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

if len(str(num)) == 10:
    while num > 0:
        digit = num % 10
        if digit in numbers:
            numbers.remove(digit)
        num //= 10
    print(numbers)
else:
    print("Mobile number should contain 10 numbers.")
```
## Cycle 2

### 1. Create a three dimensional array specifying float data type and print it.

```python

import numpy as np

arr = np.array([
	[
    	[1.1, 1.2, 1.3, 1.4, 1.5],
    	[2.1, 2.2, 2.3, 2.4, 2.5],
    	[3.1, 3.2, 3.3, 3.4, 3.5],
    	[4.1, 4.2, 4.3, 4.4, 4.5]
	],
	[
    	[5.1, 5.2, 5.3, 5.4, 5.5],
    	[6.1, 6.2, 6.3, 6.4, 6.5],
    	[7.1, 7.2, 7.3, 7.4, 7.5],
    	[8.1, 8.2, 8.3, 8.4, 8.5]
	]
], dtype=float)

print("3D Array")
print(arr)

```
### 2. Create a 2 dimensional array (2X3) with elements belonging to complex data type and print it. Also display
a. the no: of rows and columns <br>
b. dimension of an array  <br>
c. reshape the same array to 3X2  <br>

```python

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

```

### 3. Familiarize with the functions to create
a. an uninitialized array <br>
b. array with all elements as 1, <br>
c. all elements as 0 <br>

```python

import numpy as np

uninitialized_array = np.empty(shape=(2,3))
print("Uninitialized Array:")
print(uninitialized_array)

ones_array = np.ones((2,3))
print("\nArray with all elements as 1:")
print(ones_array)

zeros_array = np.zeros((2,3))
print("\nArray with all elements as 0:")
print(zeros_array)

```

### 4. Create an one dimensional array using arange function containing 10 elements.
Display
a. First 4 elements <br>
b. Last 6 elements <br>
c. Elements from index 2 to 7 <br>

```python

import numpy as np
one_dimensional_array = np.arange(10)
first_4_elements = one_dimensional_array[:4]
last_6_elements = one_dimensional_array[-6:]
elements_2_to_7 = one_dimensional_array[2:8]
print("Original Array:", one_dimensional_array)
print("a. First 4 elements:", first_4_elements)
print("b. Last 6 elements:", last_6_elements)
print("c. Elements from index 2 to 7:", elements_2_to_7)

```

### 5. Create an 1D array with arrange containing first 15 even numbers as elements
a. Elements from index 2 to 8 with step 2(also demonstrate the same using slice function) <br>
b. Last 3 elements of the array using negative index <br>
c. Alternate elements of the array <br>
d. Display the last 3 alternate elements <br>

```python

import numpy as np
even_numbers = np.arange(2, 31, 2)
slice_result = even_numbers[2:9:2]
last_3_elements = even_numbers[-3:]
alternate_elements = even_numbers[::2]
last_3_alternate_elements = alternate_elements[-3:]
print("Original array:", even_numbers)
print("Elements from index 2 to 8 with step 2:", slice_result)
print("Last 3 elements of the array using negative index:",
last_3_elements)
print("Alternate elements of the array:", alternate_elements)
print("Last 3 alternate elements:", last_3_alternate_elements)

```

### 6. Create a 2 Dimensional array with 4 rows and 4 columns.
a. Display all elements excluding the first row <br>
b. Display all elements excluding the last column <br>
c. Display the elements of 1 st and 2 nd column in 2 nd and 3 rd row <br>
d. Display the elements of 2 nd and 3 rd column <br>
e. Display 2 nd and 3 rd element of 1 st row <br>
f. Display the elements from indices 4 to 10 in descending order(use –values) <br>

```python

import numpy as np
two_dimensional_array = np.array([[1, 2, 3, 4],
[5, 6, 7, 8],
[9, 10, 11, 12],
[13, 14, 15, 16]])
excluding_first_row = two_dimensional_array[1:]
excluding_last_column = two_dimensional_array[:, :-1]
column_1_2_in_row_2_3 = two_dimensional_array[1:3, 0:2]
column_2_3 = two_dimensional_array[:, 1:3]
elements_2_3_in_first_row = two_dimensional_array[0, 1:3]
descending_order = two_dimensional_array.ravel()[::-1][4:11]
print("Original 2D array:\n", two_dimensional_array)
print("Elements excluding the first row:\n", excluding_first_row)
print("Elements excluding the last column:\n", excluding_last_column)
print("Elements of the 1st and 2nd column in the 2nd and 3rd row:\n",
column_1_2_in_row_2_3)
print("Elements of the 2nd and 3rd column:\n", column_2_3)
print("2nd and 3rd element of the 1st row:\n",
elements_2_3_in_first_row)
print("Elements from indices 4 to 10 in descending order:\n",
descending_order)

```

### 7. Create two 2D arrays using array object and
a. Add the 2 matrices and print it <br>
b. Subtract 2 matrices <br>
c. Multiply the individual elements of matrix <br>
d. Divide the elements of the matrices <br>
e. Perform matrix multiplication <br>
f. Display transpose of the matrix <br>
g. Sum of diagonal elements of a matrix <br>

```python

import numpy as np
matrix1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
matrix2 = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]])
matrix_sum = matrix1 + matrix2
matrix_diff = matrix1 - matrix2
matrix_product = matrix1 * matrix2
matrix_divide = matrix1 / matrix2
matrix_multiply = np.dot(matrix1, matrix2)
matrix1_transpose = np.transpose(matrix1)
diagonal_sum = np.trace(matrix1)
print("Matrix 1:\n", matrix1)
print("Matrix 2:\n", matrix2)
print("Matrix Sum:\n", matrix_sum)
print("Matrix Difference:\n", matrix_diff)
print("Matrix Element-wise Product:\n", matrix_product)
print("Matrix Element-wise Division:\n", matrix_divide)
print("Matrix Multiplication:\n", matrix_multiply)
print("Transpose of Matrix 1:\n", matrix1_transpose)
print("Sum of Diagonal Elements of Matrix 1:", diagonal_sum)

```

### 8. Demonstrate the use of insert() function in 1D and 2D array

```python

import numpy as np
array_1d = np.array([1, 2, 3, 4, 5])
print("1D Array before insertion:")
print(array_1d)
array_1d = np.insert(array_1d, 2, 6)
print("1D Array after insertion:")
print(array_1d)
array_2d = np.array([[1, 2, 3],
[4, 5, 6]])
print("Original 2D Array:")
print(array_2d)
array_2d = np.insert(array_2d, 1, [7, 8, 9], axis=0)
print("2D Array after insertions:")
print(array_2d)

```

### 9. Demonstrate the use of diag() function in 1D and 2D array.(use both square matrix and matrix with different dimensions)

```python
import numpy as np;
arr_id = np.array([1, 2, 3, 4, 5])
diagonal_matrix = np.diag(arr_id)
print("1D Array:")
print(arr_id)
print("\nDiagonal Matrix:")
print(diagonal_matrix)
arr_2d_square = np.array([[1, 2, 3],
[4, 5, 6],
[7, 8, 9]])
diagonal_elements = np.diag(arr_2d_square)
print("\n2D Square Matrix:")
print(arr_2d_square)
print("\nDiagonal Elements:")
print(diagonal_elements)
arr_2d_non_square = np.array([[1, 2, 3],
[4, 5, 6]])

diagonal_elements_non_square = np.diag(arr_2d_non_square)
print("\n2D Non-Square Matrix:")
print(arr_2d_non_square)
print("\nDiagonal Elements (Non-Square):")
print(diagonal_elements_non_square)

```

### 10.Create a square matrix with random integer values(use randint()) and use appropriate functions to find:
i. Inverse <br>
ii. rank of matrix <br>
iii. Determinant <br>
iv. transform matrix into 1D array <br>
v. eigen values and vectors <br>

```python

import numpy as np;
matrix_size = 3
matrix = np.random.randint(10,20, size=(matrix_size, matrix_size))
print("Original Matrix:")
print(matrix)
if np.linalg.matrix_rank(matrix) == matrix_size:
	inverse_matrix = np.linalg.inv(matrix)
	print("\nInverse Matrix:")
	print(inverse_matrix)
else:
	print("\nThe matrix is not invertible (its rank is less than thesize).")
rank = np.linalg.matrix_rank(matrix)
print("\nRank of the Matrix:", rank)
determinant = np.linalg.det(matrix)
print("\nDeterminant of the Matrix:", determinant)
matrix_1d = matrix.flatten()
print("\nMatrix as 1D Array:")
print(matrix_1d)
eigenvalues, eigenvectors = np.linalg.eig(matrix)
print("\nEigenvalues:")
print(eigenvalues)
print("\nEigenvectors:")
print(eigenvectors)

```

### 11. Create a matrix X with suitable rows and columns

i. Display the cube of each element of the matrix using different methods(use multiply(), *, power(),**) <br>
ii. Display identity matrix of the given square matrix. <br>
iii. Display each element of the matrix to different powers. <br>

```python

import numpy as np
X = np.array([[1, 2,3 ],
[4, 5, 6],
[7, 8, 9]])

X_cube_multiply = np.multiply(X, np.multiply(X, X))
X_cube_operator = X * X * X
X_cube_power = np.power(X, 3)
X_cube_double_star = X ** 3
identity_matrix = np.identity(X.shape[0])
X_power_2 = np.power(X, 2)
X_power_3 = np.power(X, 3)
X_power_4 = np.power(X, 4)
print("Original Matrix X:")
print(X)
print("\nCubed Matrix (Method 1 - multiply()):")
print(X_cube_multiply)
print("\nCubed Matrix (Method 2 - * operator):")
print(X_cube_operator)
print("\nCubed Matrix (Method 3 - power()):")
print(X_cube_power)
print("\nCubed Matrix (Method 4 - ** operator):")
print(X_cube_double_star)
print("\nIdentity Matrix:")
print(identity_matrix)
print("\nMatrix to Different Powers:")
print("X^2:")
print(X_power_2)
print("\nX^3:")
print(X_power_3)
print("\nX^4:")
print(X_power_4)

```

### 12.Create a matrix Y with same dimension as X and perform the operation X 2 +2Y

```python

import numpy as np;
X = np.array([[1, 2],
[3, 4]])
Y = np.random.rand(*X.shape)
result = X * 2 + 2 * Y
print(result)

```
