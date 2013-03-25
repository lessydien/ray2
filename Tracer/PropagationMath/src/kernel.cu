/***********************************************************************
 This file is part of ITO-MacroSim.

    ITO-MacroSim is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    ITO-MacroSim is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Foobar.  If not, see <http://www.gnu.org/licenses/>.
************************************************************************/

#include "kernel.h"
#include "math.h"

// Complex pointwise multiplication
static __global__ void ComplexPointwiseMul(cuDoubleComplex* a, const cuDoubleComplex* b, int size)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = threadID; i < size; i += numThreads)
        a[i] = cuCmul(a[i], b[i]);     
} 

// Complex pointwise multiplication
static __global__ void ComplexPointwiseMulandScale(cuDoubleComplex* a, const cuDoubleComplex* b, int size, double scale)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = threadID; i < size; i += numThreads)
	{
        a[i] = cuCmul(a[i], b[i]);     
		a[i].x = a[i].x*scale;
		a[i].y = a[i].y*scale;
	}
} 

//__global__ void kernel(int *a, int*b)
//{
//	int tx = threadIdx.x;
//	
//	switch(tx)
//	{
//		case 0:
//			*a=*a+10;
//			break;
//		case 1:
//			*b=*b+3;
//			break;
//		default:
//			break;
//	}
//}

static __global__ void scalar_RichardsonWolf_kernel(cuDoubleComplex* Uin_ptr, double* x1_ptr,  double* y1_ptr, double* x2_ptr, double*y2_ptr, unsigned int dimx, unsigned int dimy, unsigned int TileWidth, unsigned int TileHeight, double wvl, double f, double Dz)
{
	unsigned int jx=blockIdx.x*TileWidth+threadIdx.x;
	unsigned int jy=blockIdx.y*TileHeight+threadIdx.y;

	double dx1=abs(x1_ptr[1]-x1_ptr[0]);
	double dy1=abs(y1_ptr[1]-y1_ptr[0]);
	x2_ptr[jx]=(-1.0*dimx/2+jx)/(dimx*dx1)*wvl*f;
	y2_ptr[jy]=(-1.0*dimy/2+jy)/(dimy*dy1)*wvl*f;

	double sigmaX=-x1_ptr[jx]/f;
	double sigmaY=-y1_ptr[jy]/f;

	double GktSqr=1-sigmaX*sigmaX-sigmaY*sigmaY;
	// free space propagation filters out evanescent waves...
	if (GktSqr<0)
	{
		GktSqr=0.0;
		Uin_ptr[jx+jy*dimy]=make_cuDoubleComplex(0.0,0.0);
	}
	else
	{
		// this looks kind of ugly because cudas complex<double> implementation doesn't have any operator notation...
		//Uin_ptr[jx+jy*dimy]=make_cuDoubleComplex(0.0,-1.0)*f*Uin_ptr[jx+jy*dimy]/pow(make_cuDoubleComplex(1-sigmaX*sigmaX-sigmaY*sigmaY,0.0),0.25)*make_cuDoubleComplex(cos(2*PI/wvl*Dz*sqrt(GktSqr)),sin(2*PI/wvl*Dz*sqrt(GktSqr)));
		cuDoubleComplex help=cuCmul(make_cuDoubleComplex(f/pow(1-sigmaX*sigmaX-sigmaY*sigmaY,0.25),0.0),Uin_ptr[jx+jy*dimy]);
		help=cuCmul(make_cuDoubleComplex(0.0,-1.0),help);
		Uin_ptr[jx+jy*dimy]=cuCmul(help,make_cuDoubleComplex(cos(2*PI/wvl*Dz*sqrt(GktSqr)),sin(2*PI/wvl*Dz*sqrt(GktSqr))));
	}
}

bool scalar_RichardsonWolf_wrapper(complex<double>* Uin_ptr, unsigned int dimx, unsigned int dimy, double wvl, double* x1_ptr, double* y1_ptr, double f, double Dz, double** x2_ptrptr, double** y2_ptrptr)
{
	// we handle only regularly squared grids here
	if (dimx!=dimy)
		return 0;

	double k=2*PI/wvl;
	double dx1=abs(x1_ptr[0]-x1_ptr[1]);
	double dy1=abs(y1_ptr[0]-y1_ptr[1]);
	if (dx1!=dy1)
		return 0;

	// transfer data to GPU
	double* x2_kernel_ptr;
	cutilSafeCall(cudaMalloc((void**)&x2_kernel_ptr, sizeof(double)*dimx));
	//(cudaMalloc((void**)&x2_kernel_ptr, sizeof(double)*dimx));
	double* y2_kernel_ptr;
	cutilSafeCall(cudaMalloc((void**)&y2_kernel_ptr, sizeof(double)*dimy));
	//(cudaMalloc((void**)&y2_kernel_ptr, sizeof(double)*dimy));

	double* x1_kernel_ptr;
	cutilSafeCall(cudaMalloc((void**)&x1_kernel_ptr, sizeof(double)*dimx));
	//(cudaMalloc((void**)&x1_kernel_ptr, sizeof(double)*dimx));
	cutilSafeCall(cudaMemcpy(x1_kernel_ptr, x1_ptr, sizeof(double)*dimx, cudaMemcpyHostToDevice));
	//(cudaMemcpy(x1_kernel_ptr, x1_ptr, sizeof(double)*dimx, cudaMemcpyHostToDevice));
	double* y1_kernel_ptr;
	cutilSafeCall(cudaMalloc((void**)&y1_kernel_ptr, sizeof(double)*dimy));
	//(cudaMalloc((void**)&y1_kernel_ptr, sizeof(double)*dimy));
	cutilSafeCall(cudaMemcpy(y1_kernel_ptr, y1_ptr, sizeof(double)*dimy, cudaMemcpyHostToDevice));
	//(cudaMemcpy(y1_kernel_ptr, y1_ptr, sizeof(double)*dimy, cudaMemcpyHostToDevice));

	complex<double>* Uin_kernel_ptr;
	cutilSafeCall(cudaMalloc((void**)&Uin_kernel_ptr, sizeof(complex<double>)*dimx*dimy));
	//(cudaMalloc((void**)&Uin_kernel_ptr, sizeof(complex<double>)*dimx*dimy));
	cutilSafeCall(cudaMemcpy(Uin_kernel_ptr, Uin_ptr, sizeof(complex<double>)*dimx*dimy, cudaMemcpyHostToDevice));
	//(cudaMemcpy(Uin_kernel_ptr, Uin_ptr, sizeof(complex<double>)*dimx*dimy, cudaMemcpyHostToDevice));

	unsigned int tileWidth=16;
	unsigned int tileHeight=16;

	dim3 dimBlock(tileWidth,tileHeight,1); // number of threads within each block in x,y,z (maximum of 512 in total. I.e. 512,1,1 or 8,16,2 or ...
	dim3 dimGrid(dimx/tileWidth,dimy/tileHeight,1); // number of blocks in x,y,z (maximum of 65535 for each dimension)

	scalar_RichardsonWolf_kernel<<<dimGrid, dimBlock>>>((cuDoubleComplex*)Uin_kernel_ptr, x1_kernel_ptr, y1_kernel_ptr, x2_kernel_ptr, y2_kernel_ptr, dimx, dimy, tileWidth, tileHeight, wvl, f, Dz);


	// allocate host memory for observation plane coordinates
	double *x2_l=(double*)calloc(dimx,sizeof(double));
	double *y2_l=(double*)calloc(dimy,sizeof(double));

	// transfer coordinates from GPU
	cutilSafeCall(cudaMemcpy(x2_l, x2_kernel_ptr, sizeof(double)*dimx, cudaMemcpyDeviceToHost));
	//(cudaMemcpy(x2_l, x2_kernel_ptr, sizeof(double)*dimx, cudaMemcpyDeviceToHost));
	cutilSafeCall(cudaMemcpy(y2_l, y2_kernel_ptr, sizeof(double)*dimy, cudaMemcpyDeviceToHost));
	//(cudaMemcpy(y2_l, y2_kernel_ptr, sizeof(double)*dimy, cudaMemcpyDeviceToHost));

	//deallocate coordinates on GPU
	cudaFree(x1_kernel_ptr);
	cudaFree(x2_kernel_ptr);
	cudaFree(y1_kernel_ptr);
	cudaFree(y2_kernel_ptr);

	// do fft
    // plan fft
    cufftHandle plan;
    //cufftSafeCall(cufftPlan2d(&plan,dimx, dimy, CUFFT_Z2Z));
	(cufftPlan2d(&plan,dimx, dimy, CUFFT_Z2Z));

    // execute fft
    cufftSafeCall(cufftExecZ2Z(plan, (cufftDoubleComplex *)Uin_kernel_ptr, (cufftDoubleComplex *)Uin_kernel_ptr, CUFFT_FORWARD));
	//(cufftExecZ2Z(plan, (cufftDoubleComplex *)Uin_kernel_ptr, (cufftDoubleComplex *)Uin_kernel_ptr, CUFFT_FORWARD));

	// transfer optical field from GPU
	cutilSafeCall(cudaMemcpy(Uin_ptr, Uin_kernel_ptr, sizeof(complex<double>)*dimy*dimx, cudaMemcpyDeviceToHost));
	//(cudaMemcpy(Uin_ptr, Uin_kernel_ptr, sizeof(complex<double>)*dimy*dimx, cudaMemcpyDeviceToHost));
	// deallocate optical field on GPU
	cudaFree(Uin_kernel_ptr);
	// destroy fft plan
	cufftDestroy(plan);


	// return pointer to new coordinates
	*x2_ptrptr=x2_l;
	*y2_ptrptr=y2_l;

	return 1;
}