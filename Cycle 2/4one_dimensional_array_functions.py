import numpy as np

one_dimensional_array = np.arange(10)
first_4_elements = one_dimensional_array[:4]
last_6_elements = one_dimensional_array[-6:]
elements_2_to_7 = one_dimensional_array[2:8]

print("Original Array : ",one_dimensional_array)
print("First 4 elements : ",first_4_elements)
print("Last 6 elements : ",last_6_elements)
print("Elements from index 2 to 7 :",elements_2_to_7)