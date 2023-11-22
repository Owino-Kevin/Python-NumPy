# NumPy(Numerical Python) is a Python Library used for working with arrays.
import numpy as np  # This imports numpy and renames it as np for flexibility.

arr1 = np.array(42)  # This is a 0-Dimensional array
arr = np.array([1, 2, 3, 5, 7, 8])  # This creates a 1-Dimensional array
arr2 = np.array([[1, 2, 3], [4, 5, 6]])  # This is a 2-Dimensional array
print(arr1)
print(arr2)

arr5 = np.array([1, 2, 3, 4], ndmin=5)
print(arr5)

print(arr)
print(np.__version__)  # This shows the numpy version installed.
print(arr[1:5])  # Slicing arr. Meaning, we take values from second(included) to fifth(excluded) positions.
print(arr[4:])  # Slice elements from index 4 to the end
print(arr[:4])  # Slice elements from the begining to index 4
print(arr[-3:-1])  # Slice from index 3 form the end to index 1 from the end.
print(arr.dtype)  # To show the data type in the array
x = arr.copy()  # To copy the original array
print(x)
print(arr2.shape)  # To show the number of elements in each dimension.
# arr3 = np.concatenate((arr, arr2))   This joins the two arrays

# In this Section, we look at Random numbers(algorithm generated/Pseudo random numbers)
from numpy import random

x = random.randint(100)  # Generates a random integer from 0 to 100

print(x)
y = random.rand()  # Generates a random float
print(y)
x = random.randint(100, size=(10))  # Generates a random 1-Dimensional array
print(x)
x = random.randint(100, size=(3, 5))  # Generates a 2-D array, each row containing 5 random integers from 0 to 100.
print(x)
x = random.choice([3, 5, 7, 9], p=[0.1, 0.3, 0.6, 0.0], size=(100))  # Generate a 1-D array containing 100 values,
# where each value has to be 3,5,7,9 with the respective probabilities of occurrences.
print(x)

random.shuffle(arr)  # Shuffles the elements
print(arr)
print(random.permutation(arr))  # Generate a random permutation of elements in the array

# We look at SEABORN module: Seaborn uses matplotlib to plot graphs. Used to visualize distributions.
import seaborn as sns

sns.distplot([0, 1, 2, 3, 4, 5], hist=False)  # Plots distribution curve for the array using matplotlib and seaborn

# plt.show()  # To show the distribution, remove the #

# The NORMAL/GAUSSIAN distribution.
from numpy import random

x = random.normal(loc=1, scale=2, size=(2, 3))  # Generates random normal distribution of size 2 by 3
print(x)
sns.distplot(random.normal(size=1000), hist=False)
# plt.show()  # To show the distribution, remove the #

# The BINOMIAL Distribution
# A discrete distribution that describes the outcomes of binary scenarios eg a toss of a coin
x = random.binomial(n=10, p=0.5, size=12)  # Generate 10 data points given 10 trials for coin toss
print(x)
sns.distplot(random.binomial(n=10, p=0.5, size=1000), hist=True, kde=False)  # Plotting a histogram without curve
# plt.show()  # To show the distribution, remove the #
from numpy import random
import matplotlib.pyplot as plt
import seaborn as sns

sns.distplot(random.normal(loc=50, scale=5, size=1000), hist=False, label='normal')  # Compares the two distributions
sns.distplot(random.binomial(n=100, p=0.5, size=1000), hist=False, label='binomial')
# plt.show()  # To show the distributions, remove the #

# The POISSON Distribution
# Estimates how many times an event can happen in a specified time
x = random.poisson(lam=2, size=10)
print(x)
sns.distplot(random.poisson(lam=2, size=1000), kde=False)
# plt.show()  # To show the distribution, remove the #
sns.distplot(random.normal(loc=50, scale=7, size=1000), hist=False, label='normal')  # Compares the two distributions
sns.distplot(random.poisson(lam=50, size=1000), hist=False, label='poisson')  # Compares the two distributions
# plt.show()  # To show the distributions, remove the #

# The NumPy Universal Functions.
import numpy as np

x = [1, 2, 3, 4]
y = [4, 5, 6, 7]
z = np.add(x, y)  # Sums the two lists
print(z)

# Creating own function in Python
import numpy as np


def myadd(x, y):
    return x + y


myadd = np.frompyfunc(myadd, 2, 1)  # Functions that sums the two lists
print(myadd([1, 2, 3, 4], [5, 6, 7, 8]))

newarr = np.subtract(x, y)  # Subtraction of the two arrays x and y
print(newarr)

newarr1 = np.multiply(x, newarr)  # Multiplies the two arrays x and newarr
print(newarr1)

x = np.lcm.reduce(arr)  # Finds the LCM of the array
print(x)

y = np.gcd.reduce(arr)  # Prints the GCD of the array
print(y)

z = np.sin(arr)  # To find the sin of the array. Cos/h and Tan/h can also be done in a similar way.
print(z)

x = np.arcsin(arr)  # To find the angle of each value in the array
print(x)
