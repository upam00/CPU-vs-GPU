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
ker = SourceModule("""
#define _X ( threadIdx.x + blockIdx.x * blockDim.x )
#define _Y ( threadIdx.y + blockIdx.y * blockDim.y )
#define _WIDTH ( blockDim.x * gridDim.x )
#define _HEIGHT ( blockDim.y * gridDim.y )
#define _XM(x) ( (x + _WIDTH) % _WIDTH )
#define _YM(y) ( (y + _HEIGHT) % _HEIGHT )
#define _INDEX(x,y) ( _XM(x) + _YM(y) * _WIDTH )
//This is a device function to calculate the Pearson Correlation between
//two Objects. That is between two rows of the matrix
__device__ float nbrs(int x, int y, float * in)
{
 float sum_X = 0, sum_Y = 0, sum_XY = 0;
 float squareSum_X = 0, squareSum_Y = 0;
 int n=10;
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
//This is the kernel function that parallelize each calculation of
//coefficient for the resultant matrix N
__global__ void conway_ker(float* lattice_val,float * lattice_out, float *
lattice )
{
 // x, y are the appropriate values for the cell covered by this thread
 int x = _X, y = _Y;
 float n= nbrs(x,y,lattice);
 if(abs(n)<0.5)
 lattice_out[_INDEX(x,y)]=0;
 else
 lattice_out[_INDEX(x,y)]=1;
 lattice_val[_INDEX(x,y)]=n;
}
"""
)my_ker = ker.get_function("my_ker")
def update_gpu(valLattice_gpu, newLattice_gpu, lattice_gpu, N):
 my_ker(valLattice_gpu, newLattice_gpu, lattice_gpu, grid=(1,1,1),
block=( N,N,1))
 table = tabulate(lattice_gpu.get())
 print(table)
 table=tabulate(newLattice_gpu.get())
 print(table)
 table=tabulate(valLattice_gpu.get())
 print(table)
if __name__ == '__main__':
 # set matrix size
 N = 10
seed(1)
 # generate random numbers between 0-1
 values = rand(10)
 lattice=np.float32(np.random.choice(values, N*N).reshape(N, N))
 # Matrix M
 lattice_gpu = gpuarray.to_gpu(lattice)
 # Matrix N
 newLattice_gpu = gpuarray.empty_like(lattice_gpu)
 #Matrix containing values of coefficients for each (i,j) in Matrix N
 valLattice_gpu = gpuarray.empty_like(lattice_gpu)
 update_gpu(valLattice_gpu,newLattice_gpu, lattice_gpu,N)
