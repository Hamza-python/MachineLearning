# These are the libraries will be used for this lab.
# import inline as inline
import matplotlib
import torch
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

# %matplotlib inline
# From where data is lie.
# https://labs.cognitiveclass.ai/tools/jupyterlab/lab/tree/labs/DL0110EN/1.1_1Dtensors_v2.ipynb?lti=true
# Check PyTorch version:
print(torch.__version__)


# This is the function for plotting diagrams. You will use this function to plot the vectors in Coordinate system.

# Plot vectors, please keep the parameters in the same length
# @param: Vectors = [{"vector": vector variable, "name": name of vector, "color": color of the vector on diagram}]

def plotVec(vectors):
    ax = plt.axes()

    # For loop to draw the vectors
    for vec in vectors:
        ax.arrow(0, 0, *vec["vector"], head_width=0.05, color=vec["color"], head_length=0.1)
        plt.text(*(vec["vector"] + 0.1), vec["name"])

    plt.ylim(-2, 2)
    plt.xlim(-2, 2)


# Types and Shape
# You can find the type of the following list of integers [0, 1, 2, 3, 4] by applying the constructor torch.tensor():

# Convert a integer list with length 5 to a tensor
ints_to_tensor = torch.tensor([0, 1, 2, 3, 4])
print("The dtype of tensor object after converting it to tensor: ", ints_to_tensor.dtype)
print("The type of tensor object after converting it to tensor: ", ints_to_tensor.type())
# As a result, the integer list has been converted to a long tensor.

# The Python type is still torch.Tensor:

type(ints_to_tensor)
# You can find the type of this float list [0.0, 1.0, 2.0, 3.0, 4.0] by applying the method torch.tensor():

# Convert a float list with length 5 to a tensor.
floats_to_tensor = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
print("The dtype of tensor object after converting it to tensor: ", floats_to_tensor.dtype)
print("The type of tensor object after converting it to tensor: ", floats_to_tensor.type())

# The float list is converted to a float tensor.

list_floats = [0.0, 1.0, 2.0, 3.0, 4.0]

floats_int_tensor = torch.tensor(list_floats, dtype=torch.int64)
print("The dtype of tensor object is: ", floats_int_tensor.dtype)
print("The type of tensor object is: ", floats_int_tensor.type())
# Note: The elements in the list that will be converted to tensor must have the same type.

# From the previous examples, you see that torch.tensor() converts the list to the tensor type,
# which is similar to the original list type. However,
# what if you want to convert the list to a certain tensor type?
# torch contains the methods required to do this conversion.
# The following code converts an integer list to float tensor:

# Convert a integer list with length 5 to float tensor
new_float_tensor = torch.FloatTensor([0, 1, 2, 3, 4])
new_float_tensor.type()
print("The type of the new_float_tensor:", new_float_tensor.type())
new_float_tensor = torch.FloatTensor([0, 1, 2, 3, 4])

# You can also convert an existing tensor object (tensor_obj) to another tensor type.
# Convert the integer tensor to a float tensor:

# Another method to convert the integer list to float tensor
old_int_tensor = torch.tensor([0, 1, 2, 3, 4])
new_float_tensor = old_int_tensor.type(torch.FloatTensor)
print("The type of the new_float_tensor:", new_float_tensor.type())
# The tensor_obj.size() helps you to find out the size of the tensor_obj.
# The tensor_obj.ndimension() shows the dimension of the tensor object.

# Introduce the tensor_obj.size() & tensor_ndimension.size() methods
print("The size of the new_float_tensor: ", new_float_tensor.size())
print("The dimension of the new_float_tensor: ", new_float_tensor.ndimension())

# The tensor_obj.view(row, column) is used for reshaping a tensor object.
#
# What if you have a tensor object with torch.Size([5]) as a new_float_tensor as shown in the previous example?
# After you execute new_float_tensor.view(5, 1), the size of new_float_tensor will be torch.Size([5, 1]).
# This means that the tensor object new_float_tensor has been reshaped from a one-dimensional tensor object with
# 5 elements to a two-dimensional tensor object with 5 rows and 1 column.

# Introduce the tensor_obj.view(row, column) method
twoD_float_tensor = new_float_tensor.view(5, 1)
print("Original Size: ", new_float_tensor)
print("Size after view method", twoD_float_tensor)
# Note that the original size is 5. The tensor after reshaping becomes a 5X1 tensor analog to a column vector.

# Note: The number of elements in a tensor must remain constant after applying view.

# What if you have a tensor with dynamic size but you want to reshape it? You can use -1 to do just that.

# Introduce the use of -1 in tensor_obj.view(row, column) method
twoD_float_tensor = new_float_tensor.view(-1, 1)
print("Original Size: ", new_float_tensor)
print("Size after view method", twoD_float_tensor)

# You get the same result as the previous example. The -1 can represent any size.
# However, be careful because you can set only one argument as -1.

# You can also convert a numpy array to a tensor, for example:

# Convert a numpy array to a tensor
numpy_array = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
new_tensor = torch.from_numpy(numpy_array)

print("The dtype of new tensor: ", new_tensor.dtype)
print("The type of new tensor: ", new_tensor.type())
# Converting a tensor to a numpy is also supported in PyTorch. The syntax is shown below:

# Convert a tensor to a numpy array
back_to_numpy = new_tensor.numpy()
print("The numpy array from tensor: ", back_to_numpy)
print("The dtype of numpy array: ", back_to_numpy.dtype)

# back_to_numpy and new_tensor still point to numpy_array.
# As a result if we change numpy_array both back_to_numpy and new_tensor will change.
# For example if we set all the elements in numpy_array to zeros, back_to_numpy and  new_tensor will follow suit.

# Set all elements in numpy array to zero
numpy_array[:] = 0
print("The new tensor points to numpy_array : ", new_tensor)
print("and back to numpy array points to the tensor: ", back_to_numpy)
# Pandas Series can also be converted by using the numpy array that is stored in pandas_series.values.
# Note that pandas_series can be any pandas_series object.

# Convert a panda series to a tensor
pandas_series = pd.Series([0.1, 2, 0.3, 10.1])
new_tensor = torch.from_numpy(pandas_series.values)
print("The new tensor from numpy array: ", new_tensor)
print("The dtype of new tensor: ", new_tensor.dtype)
print("The type of new tensor: ", new_tensor.type())

# consider the following tensor

this_tensor = torch.tensor([0, 1, 2, 3])
# The method item() returns the value of this tensor as a standard Python number. This only works for one element.

this_tensor = torch.tensor([0, 1, 2, 3])
print("the first item is given by", this_tensor[0].item(), "the first tensor value is given by ", this_tensor[0])
print("the second item is given by", this_tensor[1].item(), "the second tensor value is given by ", this_tensor[1])
print("the third  item is given by", this_tensor[2].item(), "the third tensor value is given by ", this_tensor[2])
# we can use the method  tolist() to return a list

torch_to_list = this_tensor.tolist()
print('tensor:', this_tensor, "\nlist:", torch_to_list)

# Indexing and Slicing
# In Python, the index starts with 0. Therefore, the last index will always be 1 less than the length of the
# tensor object. You can access the value on a certain index by using the square bracket, for example:

# A tensor for showing how the index work on tensors
index_tensor = torch.tensor([0, 1, 2, 3, 4])
print("The value on index 0:", index_tensor[0])
print("The value on index 1:", index_tensor[1])
print("The value on index 2:", index_tensor[2])
print("The value on index 3:", index_tensor[3])
print("The value on index 4:", index_tensor[4])
# Note that the index_tensor[5] will create an error.

# The index is shown in the following figure:

# Python Index
# Now, you'll see how to change the values on certain indexes.

# Suppose you have a tensor as shown here:

# A tensor for showing how to change value according to the index
tensor_sample = torch.tensor([20, 1, 2, 3, 4])

# Assign the value on index 0 as 100:

# Change the value on the index 0 to 100
print("Initial value on index 0:", tensor_sample[0])
tensor_sample[0] = 100
print("Modified tensor:", tensor_sample)
# As you can see, the value on index 0 changes. Change the value on index 4 to 0:

# Change the value on the index 4 to 0
print("Initial value on index 4:", tensor_sample[4])
tensor_sample[4] = 0
print("Modified tensor:", tensor_sample)
# The value on index 4 turns to 0.

# If you are familiar with Python, y/ou know that there is a feature called slicing on a list.
# Tensors support the same feature.

# Get the subset of tensor_sample. The subset should contain the values in tensor_sample from index 1 to index 3.

# Slice tensor_sample
subset_tensor_sample = tensor_sample[1:4]
print("Original tensor sample: ", tensor_sample)
print("The subset of tensor sample:", subset_tensor_sample)
# As a result, the subset_tensor_sample returned only the values on index 1, index 2, and index 3.
# Then, it stored them in a subset_tensor_sample.

# Note: The number on the left side of the colon represents the index of the first value.
# The number on the right side of the colon is always 1 larger than the index of the last value.
# For example, tensor_sample[1:4] means you get values from the index 1 to index 3 (4-1).

# As for assigning values to the certain index, you can also assign the value to the slices:

# Change the value of tensor_sample from index 3 to index 4:

# Change the values on index 3 and index 4
print("Initial value on index 3 and index 4:", tensor_sample[3:5])
tensor_sample[3:5] = torch.tensor([300.0, 400.0])
print("Modified tensor:", tensor_sample)
# The values on both index 3 and index 4 were changed. The values on other indexes remain the same.

# You can also use a variable to contain the selected indexes and pass that variable to a tensor slice
# operation as a parameter, for example:

# Using variable to contain the selected index, and pass it to slice operation
selected_indexes = [3, 4]
subset_tensor_sample = tensor_sample[selected_indexes]
print("The initial tensor_sample", tensor_sample)
print("The subset of tensor_sample with the values on index 3 and 4: ", subset_tensor_sample)
# You can also assign one value to the selected indexes by using the variable.
# For example, assign 100,000 to all the selected_indexes:

# Using variable to assign the value to the selected indexes
print("The initial tensor_sample", tensor_sample)
selected_indexes = [1, 3]
tensor_sample[selected_indexes] = 100000
print("Modified tensor with one value: ", tensor_sample)
# The values on index 1 and index 3 were changed to 100,000. Others remain the same.

# Note: You can use only one value for the assignment.

# Mean and Standard Deviation
# You'll review the mean and standard deviation methods first. They are two basic statistical methods.

# Create a tensor with values [1.0, -1, 1, -1]:

# Sample tensor for mathematics calculation methods on tensor
math_tensor = torch.tensor([1.0, -1.0, 1, -1])
print("Tensor example: ", math_tensor)
# Here is the mean method:

# Calculate the mean for math_tensor
mean = math_tensor.mean()
print("The mean of math_tensor: ", mean)
# The standard deviation can also be calculated by using tensor_obj.std():

# Calculate the standard deviation for math_tensor
standard_deviation = math_tensor.std()
print("The standard deviation of math_tensor: ", standard_deviation)
# Max and Min
# Now, you'll review another two useful methods: tensor_obj.max() and tensor_obj.min().
# These two methods are used for finding the maximum value and the minimum value in the tensor.

# Create a max_min_tensor:

# Sample for introducing max and min methods
max_min_tensor = torch.tensor([1, 1, 3, 5, 5])
print("Tensor example: ", max_min_tensor)
# Note: There are two minimum numbers as 1 and two maximum numbers as 5 in the tensor.
# Can you guess how PyTorch is going to deal with the duplicates?

# Apply tensor_obj.max() on max_min_tensor:

# Method for finding the maximum value in the tensor
max_val = max_min_tensor.max()
print("Maximum number in the tensor: ", max_val)
# The answer is tensor(5). Therefore, the method tensor_obj.max() is grabbing the maximum value but not the elements
# that contain the maximum value in the tensor.

# max_min_tensor.max()
# Use tensor_obj.min() on max_min_tensor:

# Method for finding the minimum value in the tensor
min_val = max_min_tensor.min()
print("Minimum number in the tensor: ", min_val)
# The answer is tensor(1). Therefore, the method tensor_obj.min() is grabbing the minimum value but not the elements
# that contain the minimum value in the tensor.

# Sin
# Sin is the trigonometric function of an angle. Again, you will not be introduce vto any mathematics functions.
# You'll focus on Python.

# Create a tensor with 0, π/2 and π. Then, apply the sin function on the tensor.
# Notice here that the sin() is not a method of tensor object but is a function of torch:

# Method for calculating the sin result of each element in the tensor
pi_tensor = torch.tensor([0, np.pi / 2, np.pi])
sin = torch.sin(pi_tensor)
print("The sin result of pi_tensor: ", sin)
# The resultant tensor sin contains the result of the sin function applied to each element in the pi_tensor.
# This is different from the previous methods. For tensor_obj.mean(), tensor_obj.std(), tensor_obj.max(),
# and tensor_obj.min(), the result is a tensor with only one number because these are aggregate methods.
# However, the torch.sin() is not. Therefore, the resultant tensors have the same length as the input tensor.

# Create Tensor by torch.linspace()
# A useful function for plotting mathematical functions is torch.linspace(). torch.linspace() returns evenly spaced
# numbers over a specified interval. You specify the starting point of the sequence and the ending point of
# the sequence. The parameter steps indicates the number of samples to generate. Now, you'll work with steps = 5.

# First try on using linspace to create tensor
len_5_tensor = torch.linspace(-2, 2, steps=5)
print("First Try on linspace", len_5_tensor)
# Assign steps with 9:

# Second try on using linspace to create tensor
len_9_tensor = torch.linspace(-2, 2, steps=9)
print("Second Try on linspace", len_9_tensor)
# Use both torch.linspace() and torch.sin() to construct a tensor that contains the 100 sin result in range
# from 0 (0 degree) to 2π (360 degree):

# Construct the tensor within 0 to 360 degree
pi_tensor = torch.linspace(0, 2 * np.pi, 100)
sin_result = torch.sin(pi_tensor)
# Plot the result to get a clearer picture. You must cast the tensor to a numpy array before plotting it.

# Plot sin_result
plt.plot(pi_tensor.numpy(), sin_result.numpy())
# If you know the trigonometric function, you will notice this is the diagram of the sin result in the range
# 0 to 360 degrees.

# Tensor Operations
# In the following section, you'll work with operations that you can apply to a tensor.

# Tensor Addition
# You can perform addition between two tensors.

# Create a tensor u with 1 dimension and 2 elements. Then, create another tensor v with the same number of dimensions
# and the same number of elements:

# Create two sample tensors
u = torch.tensor([1, 0])
v = torch.tensor([0, 1])
# Add u and v together:

# Add u and v
w = u + v
print("The result tensor: ", w)
# The result is tensor([1, 1]). The behavior is [1 + 0, 0 + 1].

# Plot the result to to get a clearer picture.

# Plot u, v, w
plotVec([
    {"vector": u.numpy(), "name": 'u', "color": 'r'},
    {"vector": v.numpy(), "name": 'v', "color": 'b'},
    {"vector": w.numpy(), "name": 'w', "color": 'g'}
])
# Try
# Implement the tensor subtraction with u and v as u-v.

# Try by yourself to get a result of u-v
u = torch.tensor([1, 0])
v = torch.tensor([0, 1])
# Double-click here for the solution.

# Tensors must be of the same data type to perform addition as well as other operations.
# If you uncomment the following code and try to run it you will get an error as the two tensors are of two different
# data types. NOTE This lab was created on a older PyTorch version so in the current version
# we are using this is possible and will produce a float64 tensor.

# torch.tensor([1,2,3],dtype=torch.int64)+torch.tensor([1,2,3],dtype=torch.float64)
# You can add a scalar to the tensor. Use u as the sample tensor:

# tensor + scalar
u = torch.tensor([1, 2, 3, -1])
v = u + 1
print("Addition Result: ", v)

# Tensor Multiplication
# Now, you'll review the multiplication between a tensor and a scalar.

# Create a tensor with value [1, 2] and then multiply it by 2:

# tensor * scalar
u = torch.tensor([1, 2])
v = 2 * u
print("The result of 2 * u: ", v)
# The result is tensor([2, 4]), so the code 2 * u multiplies each element in the tensor by 2.
# This is how you get the product between a vector or matrix and a scalar in linear algebra.

# You can use multiplication between two tensors.

# Create two tensors u and v and then multiply them together:

# tensor * tensor
u = torch.tensor([1, 2])
v = torch.tensor([3, 2])
w = u * v
print("The result of u * v", w)
# The result is simply tensor([3, 4]). This result is achieved by multiplying every element in u with the
# corresponding element in the same position v, which is similar to [1 * 3, 2 * 2].

# Dot Product
# The dot product is a special operation for a vector that you can use in Torch.

# Here is the dot product of the two tensors u and v:

# Calculate dot product of u, v
u = torch.tensor([1, 2])
v = torch.tensor([3, 2])
print("Dot Product of u, v:", torch.dot(u, v))
# The result is tensor(7). The function is 1 x 3 + 2 x 2 = 7.

# Practice
# Convert the list [-1, 1] and [1, 1] to tensors u and v. Then, plot the tensor u and v as a vector by using the
# function plotVec and find the dot product:

# Practice: calculate the dot product of u and v, and plot out two vectors
# Type your code here
