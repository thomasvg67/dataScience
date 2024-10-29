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
b. array with all elements as 1 <br>
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

### 13.Define matrices A with dimension 5x6 and B with dimension 3x3. Extract a sub matrix of dimension 3x3 from A and multiply it with B. Replace the extracted sub matrix in A with the matrix obtained after multiplication

```python

import numpy as np
A = np.array([[1,2,3,4,5,6],
[7,8,9,10,11,12],
[13,14,15,16,17,18],
[19,20,21,22,23,24],
[25,26,27,28,29,30]])

print("Matrix A is : ")
print(A)
B = np.array([[1,2,3,],[4,5,6],[7,8,9]])
print("Matrix B is : ")
print(B)
sub_matrix = A[:3, :3]
print("The sub matrix is ")
print(sub_matrix)
result = np.dot(sub_matrix,B)
print("Matrix after multiplication with the sub matrix of A an d matrix B")
print(result)
A[:3, :3] = result
print("Matrix A after the operation:")
print(A)

```

### 14.Given 3 Matrices A, B and C. Write a program to perform matrix multiplication of the 3 matrices.

```python

import numpy as np
A = np.array([[1, 2, 3],
[4, 5, 6],
[7, 8, 9]])
B = np.array([[9, 8, 7],
[6, 5, 4],
[3, 2, 1]])
C = np.array([[10, 5, 9],
[20, 15, 19],
[30, 2, 29]])
result = np.dot(np.dot(A, B), C)
print("Matrix A:")
print(A)
print("Matrix B:")
print(B)
print("Matrix C:")
print(C)
print("Result of (A * B) * C:")
print(result)

```

### 15.Write a program to check whether a given matrix is symmetric or Skew Symmetric.

```python

print("SJC23MCA-2058 : THOMAS V.G")
print("Batch : MCA 2023-25")
import numpy as np
def is_symmetric(matrix):
	return (matrix == matrix.T).all()
def is_skew_symmetric(matrix):
	return (matrix == -matrix.T).all()
matrix = np.array([[0, 1, -2],
[-1, 0, 3],
[2, -3, 0]])
if is_symmetric(matrix):
	print("The matrix is symmetric.")
elif is_skew_symmetric(matrix):
	print("The matrix is skew-symmetric.")
else:
	print("The matrix is neither symmetric nor skew-symmetric.")

```

### 16.Given a matrix-vector equation AX=b. Write a program to find out the value of X using solve(), given A and b as below

```python

import numpy as np
A = np.array([[2, 1,-2],[3,0,1],[1,1,-1]])
b = np.array([-3,5,-2])
X = np.linalg.solve(A, b)
print("Matrix A:")
print(A)
print("Vector b:")
print(b)
print("Solution for X:")
print(X)

```

### 17.Write a program to perform the SVD of a given matrix A. Also reconstruct the given matrix from the 3 matrices obtained after performing SVD.

```python

import numpy as np
A = np.array([[5, 27, 32], [14, 53, 62], [67, 88, 19]])
U, S, Vt = np.linalg.svd(A)
A_hat = U @ np.diag(S) @ Vt
print('Original Matrix A :' )
print(A)
print('\nSingular Values : ')
print(S)
print('\nReconstructed Matrix A_hat : ')
print(A_hat)

```
## Cycle 3

### 1. Sarah bought a new car in 2001 for $24,000. The dollar value of her car changed each year as shown in the table below.
Value of Sarah's Car<br>
Year Value<br>
2001 $24,000<br>
2002 $22,500<br>
2003 $19,700<br>
2004 $17,500<br>
2005 $14,500<br>
2006 $10,000<br>
2007 $ 5,800<br>
Represent the following information using a line graph with following style properties<br>
● X- axis - Year<br>
Y –axis - Car Value<br>
● title –Value Depreciation (left Aligned)<br>
● Line Style dash dot and Line-color should be red<br>
● point using * symbol with green color and size 20<br>

Subplot() provides multiple plots in one figure.

```python

import matplotlib.pyplot as plt
years = [2001, 2002, 2003, 2004, 2005, 2006, 2007]
car_values = [24000, 22500, 19700, 17500, 14500, 10000, 5800]
plt.figure(figsize=(10, 6))
plt.subplot(111) # Only one subplot in this example
plt.plot(years, car_values, linestyle='-.', color='red', marker='*', markersize=20, markerfacecolor='green')
plt.title("THOMAS V.G \nMCA 2023-25", loc="right")
plt.title("Value Depreciation", loc="left")
plt.xlabel("Year")
plt.ylabel("Car Value")
plt.show()

```
### 2.Following table gives the daily sales of the following items in a shop <br>
<table>
        <thead>
            <tr>
                <th>Day</th>
                <th>Mon</th>
                <th>Tues</th>
                <th>Wed</th>
                <th>Thurs</th>
                <th>Fri</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>Drinks</td>
                <td>300</td>
                <td>450</td>
                <td>150</td>
                <td>400</td>
                <td>650</td>
            </tr>
            <tr>
                <td>Food</td>
                <td>400</td>
                <td>500</td>
                <td>350</td>
                <td>300</td>
                <td>500</td>
            </tr>
        </tbody>
    </table>

Use subplot function to draw the line graphs with grids(color as blue and line style
dotted) for the above information as 2 separate graphs in two rows <br>
a) Properties for the Graph 1:<br>
● X label- Days of week<br>
● Y label-Sale of Drinks<br>
● Title-Sales Data1 (right aligned)<br>
● Line –dotted with cyan color<br>
● Points- hexagon shape with color magenta and outline black<br>
b) Properties for the Graph 2:<br>
● X label- Days of Week<br>
● Y label-Sale of Food<br>
● Title-Sales Data2 ( center aligned)<br>
● Line –dashed with yellow color<br>
● Points- diamond shape with color green and outline red<br>

```python
import matplotlib.pyplot as plt
days = ['Mon', 'Tues', 'Wed', 'Thurs', 'Fri']
drinks_sales = [300, 450, 150, 400, 650]
food_sales = [400, 500, 350, 300, 500]
fig, axs = plt.subplots(2, 1, figsize=(8, 8))
axs[0].plot(days, drinks_sales, linestyle='--', color='cyan', marker='H', markersize=8, markerfacecolor='magenta', markeredgecolor='black')
axs[0].set_xlabel('Days of Week')
axs[0].set_ylabel('Sale of Drinks')
axs[0].set_title('Sales Data1', loc='right')
axs[0].set_title('THOMAS V.G \nMCA 2023-25', loc='left')
axs[0].grid(True, color='blue', linestyle='dotted')
axs[1].plot(days, food_sales, linestyle='-', color='yellow', marker='D', markersize=8, markerfacecolor='green', markeredgecolor='red')
axs[1].set_xlabel('Days of Week')
axs[1].set_ylabel('Sale of Food')
axs[1].set_title('Sales Data2', loc='center')
axs[1].grid(True, color='blue', linestyle='dotted')
plt.tight_layout()
plt.show()
```

### 3. Create scatter plot for the below data:(use Scatter function)
<table>
        <thead>
            <tr>
                <th>Product</th>
                <th>Jan</th>
                <th>Feb</th>
                <th>Mar</th>
                <th>Apr</th>
                <th>May</th>
                <th>Jun</th>
                <th>Jul</th>
                <th>Aug</th>
                <th>Sep</th>
		<th>Oct</th>
                <th>Nov</th>
                <th>Dec</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>Affordable Segment</td>
                <td>173</td>
                <td>153</td>
                <td>195</td>
                <td>147</td>
                <td>120</td>
                <td>144</td>
                <td>148</td>
                <td>109</td>
                <td>174</td>
                <td>130</td>
                <td>172</td>
                <td>131</td> 
            </tr>
            <tr>
                <td>Luxury Segment</td>
                <td>189</td>
                <td>189</td>
                <td>105</td>
                <td>112</td>
                <td>173</td>
                <td>109</td>
                <td>151</td>
                <td>197</td>
                <td>174</td>
                <td>145</td>
                <td>177</td>
                <td>161</td>
            </tr>
            <tr>
                <td>Super Luxury Segment</td>
                <td>185</td>
                <td>185</td>
                <td>126</td>
                <td>134</td>
                <td>196</td>
                <td>153</td>
                <td>112</td>
                <td>133</td> 
                <td>200</td>
                <td>145</td>
                <td>167</td>
                <td>110</td>
            </tr>
        </tbody>
    </table>

Create scatter plot for each Segment with following properties within one graph<br>
● X Label- Months of Year with font size 18<br>
● Y-Label- Sales of Segments<br>
● Title –Sales Data<br>
● Color for Affordable segment- pink<br>
● Color for Luxury Segment- Yellow<br>
● Color for Super luxury segment-blue<br>

```python
import matplotlib.pyplot as plt
import numpy as np
month =np.array(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
AS = np.array([173,153,195,147,120,144,148,109,174,130,172,131])
LS = np.array([189,189,105,112,173,109,151,197,174,145,177,161])
SLS = np.array([185,185,126,134,196,153,112,133,200,145,167,110])
plt.xlabel('Months of Year', fontsize=18)
plt.ylabel('Sales of Segments')
plt.title('Sales Data')
plt.title('THOM V.G\nMCA 2023-2025', loc='right')
plt.scatter(month,AS, label='Affordable Segment', color='pink')
plt.scatter(month,LS, label='Luxury Segment', color='yellow')
plt.scatter(month,SLS, label='Super Luxury Segment', color='blue')
plt.show()
```

### 4. Display the above data using multiline plot( 3 different lines in same graph)
● Display the description of the graph in upper right corner(use legend())<br>
● Use different colors and line styles for 3 different lines<br>
```python
import matplotlib.pyplot as plt
import numpy as np
month =np.array(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
AS = np.array([173,153,195,147,120,144,148,109,174,130,172,131])
LS = np.array([189,189,105,112,173,109,151,197,174,145,177,161])
SLS = np.array([185,185,126,134,196,153,112,133,200,145,167,110])
plt.plot(month,AS, label='Affordable', color='pink',linestyle='--')
plt.plot(month,LS, label='Luxury', color='yellow',linestyle='-.')
plt.plot(month,SLS, label='Super Luxury', color='blue',linestyle=':')
plt.xlabel('Months of Year', fontsize=18)
plt.ylabel('Sales of Segments')
plt.title('Sales Data')
plt.title('THOMAS V.G\nMCA 2023-2025', loc='right')
plt.show()
```

### 5. 100 students were asked what their primary mode of transport for getting to school was.
The results of this survey are recorded in the table below. Construct a bar graph representing this information.<br>

<table>
        <thead>
            <tr>
                <th>Mode of Transport</th>
                <th>Frequency</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>Walking</td>
                <td>29</td>
            </tr>
            <tr>
                <td>Cycling</td>
                <td>15</td>
            </tr>
            <tr>
                <td>Car</td>
                <td>35</td>
            </tr>
            <tr>
                <td>Bus</td>
                <td>18</td>
            </tr>
            <tr>
                <td>Train</td>
                <td>3</td>
            </tr>
        </tbody>
    </table>

Create a bar graph with<br>
● X axis -mode of Transport and Y axis ‘frequency’<br>
● Provide appropriate labels and title<br>
● Width .1, color green<br>

```python
import matplotlib.pyplot as plt
import numpy as np
mode_transport = np.array(['Walking','Cycling','Car','Bus','Train'])
feq = np.array([29,15,35,18,3])
plt.xlabel('Mode of Transport')
plt.ylabel('Frequency')
plt.title('THOMAS V.G\nMCA 2023-2025', loc='right')
plt.bar(mode_transport,feq, width=0.1, color='green')
plt.show()
```
### 6. We are provided with the height of 30 cherry trees.The height of the trees (in inches): 61,63, 64, 66, 68, 69, 71, 71.5, 72, 72.5, 73, 73.5, 74, 74.5, 76, 76.2, 76.5, 77, 77.5, 78, 78.5,79, 79.2, 80, 81, 82, 83, 84, 85,87.Create a histogram with a bin size of 5

```python
import matplotlib.pyplot as plt
import numpy as np
x = np.random.normal([61, 63, 64, 66, 68, 69, 71, 71.5, 72, 72.5, 73, 73.5, 74, 74.5, 76, 76.2, 76.5, 77, 77.5, 78, 78.5, 79, 79.2, 80, 81, 82, 83, 84, 85, 87])
plt.hist(x, bins=range(40,110,5), )
plt.title('Cherry tree heights',loc='left')
plt.title('THOMAS V.G\nMCA 2023-2025', loc='right')
plt.xlabel('Height (in inches)')
plt.ylabel('Frequency')
plt.show()
```

### 7. Using the pandas function read_csv(), read the given ‘iris’ data set.

i. Display Shape of the data set.<br>
ii. First 5 and last five rows of data set(head and tail).<br>
iii. Size of dataset.<br>
iv. No. of samples available for each variety.<br>
v. Description of the data set( use describe ).<br>

```python
import pandas as pd
df = pd.read_csv('iris.csv')
print("Shape of the dataset is : ",df.shape)
print("First 5 and last five rows of data set\n",df)
print("Size of dataset : ",df.size)
print("No. of samples available for each variety\n",df.count())
print("Description of the data set\n",df.describe())
```

### 8. Use the pairplot() function in seaborn to display pairwise relationships between attributes.Try different kind of plots {‘scatter’, ‘kde’, ‘hist’, ‘reg’} and different kind of markers.

```python
import pandas as pd
import seaborn
import matplotlib.pyplot as plt
df = pd.read_csv('iris.csv')
seaborn.pairplot(df, kind='scatter')
seaborn.pairplot(df, kind='kde')
seaborn.pairplot(df, kind='hist')
seaborn.pairplot(df, kind='reg')
plt.show()
```

### 9. Using the iris data set,get familiarize with functions:

1) displot()<br>
2) histplot()<br>
3) relplot()<br>

Note: import pandas and seaborn packages
```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
iris=pd.read_csv("iris.csv")
sns.displot(iris['sepal.length'],kde=True,rug=True)
plt.title("Distribution of Sepal length")
plt.show()
sns.histplot(iris['petal.width'],kde=True,bins=20)
plt.title("Histogram of Petal width")
plt.show()
sns.relplot(x="sepal.length",y="sepal.width",data=iris,hue="variety",style="variety")
plt.title("Sepal Length v/s Sepal Width")
plt.show()
```

## Cycle 4
### 1. Using the iris data set, implement the KNN algorithm. Take different values for the Test and training data set .Also use different values for k. Also find the accuracy level.

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load the dataset
dataset = pd.read_csv("iris.csv")

# Split features and target variable
X = dataset.iloc[:, :-1].values  # All columns except the last
y = dataset.iloc[:, -1].values     # Last column as target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Create and train the KNN classifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)

# Make predictions
y_pred = classifier.predict(X_test)

# Print classification report and confusion matrix
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

```

### 2. Download another data set suitable for the KNN and implement the KNN algorithm. Take different values for the Test and training data set .Also use different values for k.

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load the dataset
dataset = pd.read_csv("Blood Transfusion.csv")

# Split features and target variable
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values  # Assuming the target is in the last column

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Create and train the classifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)

# Make predictions
y_pred = classifier.predict(X_test)

# Print classification report and confusion matrix
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

```

### 3. Using iris data set, implement naive bayes classification for different naive Bayes classification algorithms.( (i) gaussian (ii) bernoulli etc)

● Find out the accuracy level w.r.t to each algorithm
● Display the no:of mislabeled classification from test data set
● List out the class labels of the mismatching records

I. Gaussian

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score

# Load the dataset
dataset = pd.read_csv('iris.csv')

# Split features and target variable
X = dataset.iloc[:, :4].values
y = dataset['variety'].values

# Display the first few rows
print(dataset.head())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train the classifier
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Make predictions
y_pred = classifier.predict(X_test)

# Display predictions
print("Predictions:", y_pred)

# Compute and print the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Calculate and print the accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Create a DataFrame to compare real and predicted values
results_df = pd.DataFrame({'Real Values': y_test, 'Predicted Values': y_pred})
print(results_df)

```

II. Bernoulli

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import confusion_matrix, accuracy_score

# Load the dataset
dataset = pd.read_csv('iris.csv')

# Split features and target variable
X = dataset.iloc[:, :4].values
y = dataset['variety'].values

# Display the first few rows
print(dataset.head())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train the classifier
classifier = BernoulliNB()
classifier.fit(X_train, y_train)

# Make predictions
y_pred = classifier.predict(X_test)

# Display predictions
print("Predictions:", y_pred)

# Compute and print the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Calculate and print the accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Create a DataFrame to compare real and predicted values
results_df = pd.DataFrame({'Real Values': y_test, 'Predicted Values': y_pred})
print(results_df)

```

### 4. Use car details CSV file and implement decision tree algorithm

● Find out the accuracy level.
● Display the no: of mislabelled classification from test data set
● List out the class labels of the mismatching records

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv('car.csv')

# Assign column names
col_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
data.columns = col_names

# Factorize categorical columns
for column in col_names:
    data[column], _ = pd.factorize(data[column])

# Display the first few rows
print(data.head())

# Separate features and target variable
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train the Decision Tree classifier
tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)

# Predict on the test set
y_pred = tree.predict(X_test)

# Calculate and print the number of misclassified samples
count_misclassified = (y_test != y_pred).sum()
print('Misclassified samples count:', count_misclassified)

# Calculate and print the accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

```

### 5. Implement Simple and multiple linear regression for the data sets ‘student_score.csv’ and ‘company_data .csv’ respectively

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plt

# Load the dataset
student = pd.read_csv('student_scores.csv')

# Display basic information
print(student.head())
print(student.describe())
print(student.info())

# Plotting the data
plt.scatter(student.iloc[:, 0], student.iloc[:, 1])
plt.xlabel("No. of hours")
plt.ylabel("Score")
plt.title("Student Scores")
plt.show()

# Prepare the data for training
X = student.iloc[:, :-1].values
y = student.iloc[:, 1].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the regressor
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Display the coefficients
print('Intercept:', regressor.intercept_)
print('Coefficient:', regressor.coef_)

# Predict on the test set
y_pred = regressor.predict(X_test)

# Calculate and display error metrics
print("Mean Absolute Error:", metrics.mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", metrics.mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error:", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# Compare actual vs predicted values
for actual, predicted in zip(y_test, y_pred):
    if actual != predicted:
        print(f"Actual value: {actual}, Predicted value: {predicted}")

# Count the number of mislabeled points
mislabeled_count = (y_test != y_pred).sum()
print("Number of mislabeled points from test data set:", mislabeled_count)

```

### 6. Create a neural network for the given ‘houseprice.csv’ to predict the weather price of the house is above or below median value or not

```python
import tensorflow as tf
import keras
import pandas
import sklearn
import matplotlib
import pandas as pd
df = pd.read_csv('housepricedata.csv')
print(df.head())
dataset = df.values
X = dataset[:,0:10]
Y = dataset[:,10]
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
X_scale = min_max_scaler.fit_transform(X)
print(X_scale)
from sklearn.model_selection import train_test_split
X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X_scale,Y, test_size=0.3)
X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test,Y_val_and_test, test_size=0.5)
print(X_train.shape, X_val.shape, X_test.shape, Y_train.shape, Y_val.shape,Y_test.shape)
from keras.models import Sequential
from keras.layers import Dense
model = Sequential([Dense(32, activation='relu', input_shape=(10,)), Dense(32,activation='relu'),Dense(1, activation='sigmoid'),])
model.compile(optimizer='sgd', loss='binary_crossentropy',metrics=['accuracy'])
hist = model.fit(X_train, Y_train, batch_size=32, epochs=100,validation_data=(X_val, Y_val))
model.evaluate(X_test, Y_test)[1]
import matplotlib.pyplot as plt
plt.plot(hist.history['loss'])

plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()
```

## Cycle 5

### 1. Write a program to implement a simple web crawler using Python. Extract and display the content of the page(p tag)

```python
import requests
from bs4 import BeautifulSoup

def get_data(url):
    response = requests.get(url)
    return response.content

# Fetch and parse HTML data
url = "https://www.toppr.com/guides/essays/globalization-essay/"
html_data = get_data(url)
soup = BeautifulSoup(html_data, 'html.parser')

# Find and print all paragraph tags
paragraphs = soup.find_all('p')
print("<P> tag count:", len(paragraphs))

for paragraph in paragraphs:
    print(paragraph.get_text())

```
### 2. Write a program to implement a simple web crawler using Python. Display all hyperlinks in the page

```python
import requests
from bs4 import BeautifulSoup

def get_data(url):
    response = requests.get(url)
    return response.content

# Fetch and parse HTML data
url = "http://sjcetpalai.ac.in"
html_data = get_data(url)
soup = BeautifulSoup(html_data, 'html.parser')

# Find and print all links
links = soup.find_all("a")
print("Total number of links:", len(links))

for link in links:
    href = link.get("href")
    if href:  # Check if href is not None or empty
        print("Link:", href, "Text:", link.get_text())

```

### 3. Program for Natural Language Processing which performs n-grams(without using library)

```python
def gen_ngrams(text, n):
    words = text.split()
    return [words[i:i + n] for i in range(len(words) - n + 1)]

# Example usage
ngrams = gen_ngrams('Using the iris data set, implement the KNN algorithm', 3)
print(ngrams)

```

### 4. Program for Natural Language Processing which performs n-grams(using nltk library)

```python
from nltk import ngrams

# Define the sentence and n for n-grams
sentence = 'I reside in India'
n = 3

# Generate trigrams
for grams in ngrams(sentence.split(), n):
    print(grams)

```

### 5. For given text,
● perform word
● sentence tokenization
● Remove the stop words from the given text
● create n-grams

```python
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import ngrams

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Sample text
text1 = 'The data given satisfies the requirement for model generation. This is used in Data Science Lab'

# Sentence tokenization
print('Sentence tokenization:')
print(sent_tokenize(text1))

# Word tokenization
words = word_tokenize(text1)
print("\nWord tokenization:")
print(words)

# Removing stop words
filtered_words = [word for word in words if word.lower() not in stopwords.words('english')]
print("\nRemoving stop words:")
print(filtered_words)

# Generating bigrams
print("\nBigrams:")
for bigram in ngrams(filtered_words, 2):
    print(bigram)

```

### 6. Given dataset contains 200 records and five columns, two of which describe the customer’s annual income and spending score. The latter is a value from 0 to 100. The higher the number, the more this customer has spent with the company
in the past:
Using k means clustering creates 6 clusters of customers based on their spending
pattern.
● Visualize the same in a scatter plot with each cluster in a different color scheme.
● Display the cluster labels of each point.(print cluster indexes)
● Display the cluster centers.
● Use different values of K and visualize the same using scatter plot

```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load the dataset
customer = pd.read_csv('Mall_Customers.csv')

# Extract relevant data
points = customer.iloc[:, 3:5].values
x = points[:, 0]
y = points[:, 1]

# Initial scatter plot
plt.scatter(x, y, s=50, alpha=0.7)
plt.xlabel('Annual income (k$)')
plt.ylabel('Spending Score')
plt.title('Customer Data')
plt.show()

# Function to perform KMeans clustering and plot results
def plot_kmeans(n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    predicted_indexes = kmeans.fit_predict(points)
    
    plt.scatter(x, y, c=predicted_indexes, s=50, alpha=0.7, cmap='viridis')
    plt.xlabel('Annual income (k$)')
    plt.ylabel('Spending Score')
    plt.title(f'KMeans Clustering with {n_clusters} Clusters')
    plt.show()

# Plot with different number of clusters
plot_kmeans(6)
plot_kmeans(7)
```
