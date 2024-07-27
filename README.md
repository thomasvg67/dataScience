# dataScience
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
