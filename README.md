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

print("SJC23MCA-2058 : THOMAS V.G")
print("Batch : MCA 2023-25")

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
