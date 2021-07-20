# **Calculating network matrix from a square matrix using pyCuda**

We are Given a matrix M of size 10 cross 10 consisting of 10 objects, each with 10 attributes. The
objects are arranged as rows of the 10 cross 10 matrix M, so that each object will have 10 attributes.
The arrangement is shown in the figure. From the matrix we create another matrix N of size 10 cross 10 with binary entries. To fill up the (i,j)th entry of N, we first calculate the pearson
coefficient between the i
th and j
th object of A. If the coefficient is less than 0.5 we place 0 at the
(i,j)th entry of N. Otherwise we place 1. To calculate the coefficient we scan the i
th and j
th
row of
M and treat the rows as two arrays.
We apply parallel computing while calculating the pearson coefficient between i
th and j
th
rows.
We write a kernel function and launch the kernel over a grid of single block of size 10 cross 10. That
is the process of filling each entry of N is done parallely on 10 10 parallel threads of the gpu. 
The code can be found in partI.py file.

![Figure I](https://github.com/upam00/CPU-vs-GPU/blob/main/ImageI.png?raw=true)

# Comparison of the GPU and CPU Implementation of network matrix calculator function

In partI, a function for calculating network matrix has been implemented on a
GPU utilising parallel threading features of a GPU. In this update another function for
calculating network matrix has been created that can be implemented on a CPU. Then
implementation time for both GPU and CPU functions are compared and also it was checked
that they both provide that same output for the same input. The CPU function is named as
“update_cpu” and the GPU function is named as “update_gpu”.
For any input matrix of dimension N N, “update_cpu” and “update_gpu” are run 10 times and
the average run time is noted. Also the size of N is changed from 5 to 30 and the data is
collected.
It is found that run time of “update_cpu” increases exponentially as N increased. On the other
hand “update_gpu” has almost a linear increase rate. Also it is found that GPU implementation
is way faster than CPU implementation. In Fact the advantage time is also exponential. The
following graphs describe the results.

**Conclusion: GPU implementation has advantage over CPU implementation.**

![Figure II](https://github.com/upam00/CPU-vs-GPU/blob/main/ImageII.png?raw=true)

