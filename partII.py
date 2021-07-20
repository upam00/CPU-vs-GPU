import pycuda.autoinit
import pycuda.driver as drv
from pycuda import gpuarray
from pycuda.compiler import SourceModule
import numpy as np
np.set_printoptions(suppress=True)
from numpy.random import seed
from numpy.random import rand
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tabulate import tabulate
from time import time
import math
#---------------------------GPU Implementation starts---------------#
ker = SourceModule("""
#define _X ( threadIdx.x + blockIdx.x * blockDim.x )
#define _Y ( threadIdx.y + blockIdx.y * blockDim.y )
#define _WIDTH ( blockDim.x * gridDim.x )
#define _HEIGHT ( blockDim.y * gridDim.y )
#define _XM(x) ( (x + _WIDTH) % _WIDTH )
#define _YM(y) ( (y + _HEIGHT) % _HEIGHT )
#define _INDEX(x,y) ( _XM(x) + _YM(y) * _WIDTH )
__device__ float nbrs(int x, int y, float * in, int n)
{
 float sum_X = 0, sum_Y = 0, sum_XY = 0;
 float squareSum_X = 0, squareSum_Y = 0;
 //int n=10;
 for (int i = 0; i < n; i++)
 {
 // sum of elements of array X.
 sum_X = sum_X + in[_INDEX(i, x)];
 // sum of elements of array Y. sum_Y = sum_Y + in[_INDEX(i,y)];
 // sum of X[i] * Y[i].
 sum_XY = sum_XY + in[_INDEX(i, x)] * in[_INDEX(i,y)];
 // sum of square of array elements.
 squareSum_X = squareSum_X + in[_INDEX(i,x)]*in[_INDEX(i,x)];
 squareSum_Y = squareSum_Y + in[_INDEX(i,y)]*in[_INDEX(i,y)];
 }
 // use formula for calculating correlation coefficient.
 float corr = ((n * sum_XY - sum_X * sum_Y) / sqrt((n * squareSum_X -
sum_X * sum_X) * (n * squareSum_Y - sum_Y * sum_Y)));
 return corr;
}
__global__ void conway_ker(int N, float* lattice_val,float * lattice_out,
float * lattice )
{
 // x, y are the appropriate values for the cell covered by this thread
 int x = _X, y = _Y;
 float n= nbrs(x,y,lattice, N);
 if(abs(n)<0.5)
 lattice_out[_INDEX(x,y)]=0;
 else
 lattice_out[_INDEX(x,y)]=1;
 lattice_val[_INDEX(x,y)]=abs(n);
}
"""
)conway_ker = ker.get_function("conway_ker")
def update_gpu(valLattice_gpu, newLattice_gpu, lattice_gpu, N):
 conway_ker(np.int32(N), valLattice_gpu, newLattice_gpu, lattice_gpu,
grid=(1,1,1), block=( N,N,1))
 #print('Calculated on GPU:\n')
 #table = tabulate(lattice_gpu.get())
 #print(table)
 #table=tabulate(newLattice_gpu.get())
 #print(table)
 #table=tabulate(valLattice_gpu.get())
 #print(table)
#---------------------GPU Implementation Ends----------------------#
#---------------------CPU Implementation starts---------------------#
def calculate_coeff(i, j, lattice, n):
 import math
 sum_X = 0
 sum_Y = 0
 sum_XY = 0
 squareSum_X = 0
 squareSum_Y = 0
 k = 0
 while k < n :
 # sum of elements of array X.
 sum_X = sum_X + lattice[i][k]
 # sum of elements of array Y.
 sum_Y = sum_Y + lattice[j][k]
 # sum of X[i] * Y[i].
 sum_XY = sum_XY + lattice[i][k] * lattice[j][k]
 # sum of square of array elements.
 squareSum_X = squareSum_X + lattice[i][k] * lattice[i][k] squareSum_Y = squareSum_Y + lattice[j][k] * lattice[j][k]
 k = k + 1
 # use formula for calculating correlation
 # coefficient.
 corr = (n * sum_XY - sum_X * sum_Y)/(math.sqrt((n * squareSum_X -
sum_X * sum_X)* (n * squareSum_Y - sum_Y * sum_Y)))
 return corr
def update_cpu(valLattice, newLattice, lattice, N):
 for i in range(N):
 for j in range(N):
 m=calculate_coeff(i, j, lattice, N)
 valLattice[i][j]=abs(m)
 if(abs(m)<0.5):
 newLattice[i][j]=0
 else:
 newLattice[i][j]=1
 #print('Calculated on CPU:\n')
 #table = tabulate(lattice)
 #print(table)
 #table=tabulate(newLattice)
 #print(table)
 #table=tabulate(valLattice)
 #print(table)
#---------------------CPU Implementation Ends---------------------#
#---------------------Driver function Starts----------------------#
def run(N):
 seed(N)
 values = rand(N)
 lattice=np.float32(np.random.choice(values, N*N).reshape(N, N))
 newLattice=np.empty_like(lattice)
 valLattice=np.empty_like(lattice)
 lattice_gpu = gpuarray.to_gpu(lattice)
 newLattice_gpu = gpuarray.empty_like(lattice_gpu) valLattice_gpu = gpuarray.empty_like(lattice_gpu)
 table = tabulate(lattice)
 print(table)
 #t1=time()
 update_cpu(valLattice, newLattice, lattice, N)
 #t2=time()
 table=tabulate(newLattice)
 print(table)
 #print ('total time to compute on CPU: %f' % (t2 - t1))
 #t1=time()
 update_gpu(valLattice_gpu,newLattice_gpu, lattice_gpu,N)
 #t2=time()
 table=tabulate(newLattice_gpu.get())
 print(table)
 #print ('total time to compute on GPU: %f \n' % (t2 - t1))
 #values=np.empty((0,2),float)
 #for i in range(10):
 #t1=time()
 #update_cpu(valLattice, newLattice, lattice, N)
 #t2=time()
 #t3=time()
 #update_gpu(valLattice_gpu,newLattice_gpu, lattice_gpu,N)
 #t4=time()
 #values=np.append(values,[[t2-t1, t4-t3]], axis=0)
 #print(np.mean(values, axis = 0))
#---------------------Driver function Ends----------------------#
#---------------Main() Function Starts---------------------------------#
if __name__ == '__main__':
 for i in range(5, 31):
 run(i)
#---------------Main() Function Ends---------------------------------#
