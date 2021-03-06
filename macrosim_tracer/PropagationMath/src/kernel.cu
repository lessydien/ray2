#include "kernel.h"
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/copy.h>
#include <ctime>
#include <algorithm>

//#include "cutil_math.h"

//#include "cutil_inline.h"
#ifndef PI
	#define PI       3.14159265358979323846
#endif

#define THREADSPERBLOCK 1024

unsigned int nextPow2(unsigned int x)
{
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}

#ifndef MIN
#define MIN(x,y) ((x < y) ? x : y)
#endif

// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
template<class T>
struct SharedMemory
{
    __device__ inline operator       T *()
    {
        extern __shared__ T __smem[];
        return (T *)__smem;
    }

    __device__ inline operator const T *() const
    {
        extern __shared__ T __smem[];
        return (T *)__smem;
    }
};

// specialize for double to avoid unaligned memory
// access compile errors
//template<>
//struct SharedMemory<double>
//{
//    __device__ inline operator       double *()
//    {
//        extern __shared__ double __smem_d[];
//        return (double *)__smem_d;
//    }
//
//    __device__ inline operator const double *() const
//    {
//        extern __shared__ double __smem_d[];
//        return (double *)__smem_d;
//    }
//};

// specialize for cuDoubleComplex to avoid unaligned memory
// access compile errors
template<>
struct SharedMemory<cuDoubleComplex>
{
    __device__ inline operator       cuDoubleComplex *()
    {
        extern __shared__ cuDoubleComplex __smem_d[];
        return (cuDoubleComplex *)__smem_d;
    }

    __device__ inline operator const cuDoubleComplex *() const
    {
        extern __shared__ cuDoubleComplex __smem_d[];
        return (cuDoubleComplex *)__smem_d;
    }
};


struct squareCuDoubleComplex
{
    __host__ __device__
        cuDoubleComplex operator()(const cuDoubleComplex& x) const { 
            return make_cuDoubleComplex(cuCabs(x),0.0);
            //return cuCmul(x,cuCmul(make_cuDoubleComplex(9.5367e-7,0),x));
        }
};

struct addCuDoubleComplex
{
    __host__ __device__
        cuDoubleComplex operator()(const cuDoubleComplex& x, const cuDoubleComplex& y) const { 
            return cuCadd(x,y);
        }
};

////////////////////////////////////////////////////////////////////////////////
// Compute the number of threads and blocks to use for the given reduction kernel
// For the kernels >= 3, we set threads / block to the minimum of maxThreads and
// n/2. For kernels < 3, we set to the minimum of maxThreads and n.  For kernel
// 6, we observe the maximum specified number of blocks, because each thread in
// that kernel can process a variable number of elements.
////////////////////////////////////////////////////////////////////////////////
void getNumBlocksAndThreads(int whichKernel, int n, int maxBlocks, int maxThreads, int &blocks, int &threads)
{
    cudaError_t error;
    //get device capability, to avoid block/grid size excceed the upbound
    cudaDeviceProp prop;
    int device;
    (cudaGetDevice(&device));

	// check device
    error = cudaGetDeviceProperties(&prop, device);

    if (error != cudaSuccess)
    {
        std::cout << "error in cu_simConfPointRawSig_wrapper: cudaGetDeviceProperties returned error code " << error << " line: " << __LINE__ << "...\n";
    }

    if (whichKernel < 3)
    {
        threads = (n < maxThreads) ? nextPow2(n) : maxThreads;
        blocks = (n + threads - 1) / threads;
    }
    else
    {
        threads = (n < maxThreads*2) ? nextPow2((n + 1)/ 2) : maxThreads;
        blocks = (n + (threads * 2 - 1)) / (threads * 2);
    }

    if (threads*blocks > prop.maxGridSize[0] * prop.maxThreadsPerBlock)
    {
        printf("n is too large, please choose a smaller number!\n");
    }

    if (blocks > prop.maxGridSize[0])
    {
        printf("Grid size <%d> excceeds the device capability <%d>, set block size as %d (original %d)\n",
               blocks, prop.maxGridSize[0], threads*2, threads);

        blocks /= 2;
        threads *= 2;
    }

    if (whichKernel == 6)
    {
        blocks = MIN(maxBlocks, blocks);
    }
}

// Complex pointwise multiplication
//__global__ void complexPointwiseMul_kernel(cuDoubleComplex* a, const cuDoubleComplex* b, int size)
//{
//    const int numThreads = blockDim.x * gridDim.x;
//    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
//    for (int i = threadID; i < size; i += numThreads)
//        a[i] = cuCmul(a[i], b[i]);     
//} 

// Complex pointwise multiplication
//__global__ void complexPointwiseMulandScale_kernel(cuDoubleComplex* a, const cuDoubleComplex* b, int size, double scale)
//{
//    const int numThreads = blockDim.x * gridDim.x;
//    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
//    for (int i = threadID; i < size; i += numThreads)
//	{
//        a[i] = cuCmul(a[i], b[i]);     
//		a[i].x = a[i].x*scale;
//		a[i].y = a[i].y*scale;
//	}
//} 

// create Wavefront
// our field is a 1D-vector representation of a 2D-field in row major format. x is along rows
__global__ void defocField_kernel(cuDoubleComplex* d_pField, ConfPoint_KernelParams *d_pParams)
{
    unsigned int xGes=(blockIdx.x*blockDim.x+threadIdx.x);
    unsigned int yGes=(blockIdx.y*blockDim.y+threadIdx.y);
    //unsigned int yGesBase=(blockIdx.y*blockDim.y+threadIdx.y);
    //while (xGes<s_n)
    //{
	   // unsigned int yGes=yGesBase;
    //    while (yGes <s_n)
    //    {
	        if (xGes+yGes*d_pParams->n < d_pParams->n*d_pParams->n)
	        {
	            double x=double(xGes)*(d_pParams->gridWidth/d_pParams->n)-d_pParams->gridWidth/2;
	            double y=double(yGes)*(d_pParams->gridWidth/d_pParams->n)-d_pParams->gridWidth/2;
                double sigmaX=sin(atan(-x/(160/d_pParams->magnif)));
	            double sigmaY=sin(atan(-y/(160/d_pParams->magnif)));
                double sigmaZ=(1-sigmaX*sigmaX-sigmaY*sigmaY);
                if (sigmaZ<0)
                    sigmaZ=0;
                else
                    sigmaZ=sqrt(sigmaZ);
	            // calc defocus
	            if (xGes+yGes*d_pParams->n < d_pParams->n*d_pParams->n)
	            {
		            // calc defocus phase
		            double phase=-2*PI/d_pParams->wvl*sigmaZ*d_pParams->scanStep.z;
		            d_pField[xGes+d_pParams->n*yGes]=cuCmul(d_pField[xGes+d_pParams->n*yGes],make_cuDoubleComplex(cos(phase),sin(phase)));
		            //d_pField[xGes+s_n*yGes]=make_cuDoubleComplex(cos(phase),sin(phase));
	            }
            }
    //        yGes+=blockDim.y*gridDim.y;
    //    }
    //    xGes+=blockDim.x*gridDim.x;  
    //}
}

__global__ void objectInteractionTEA(cuDoubleComplex* pObjField, ConfPoint_KernelParams* pParams, ConfPoint_KernelObjectParams* pObjParams, unsigned int jx)
{
    __shared__ unsigned int s_n;
	__shared__ double s_k, s_NA, s_gridWidth, s_f, s_kN, s_A, s_delta1;

	// load shared memory
	if (threadIdx.x==0)
	{
		s_k=2*PI/pParams->wvl;
		s_NA=pParams->NA;
        s_gridWidth=pParams->gridWidth;
        s_f=160/pParams->magnif; // we assume a tubus length of 160mm here
        s_n=pParams->n;
        s_kN=pObjParams->kN;
        s_A=pObjParams->A;
        s_delta1=pParams->scanStep.x;
	}

	__syncthreads();

    unsigned int xGes=(blockIdx.x*blockDim.x+threadIdx.x);
    unsigned int yGes=(blockIdx.y*blockDim.y+threadIdx.y);
    unsigned int xGesObj;

	if (xGes+yGes*s_n < s_n*s_n)
	{
        // calc sampling distance in object grid
        double delta2=(2*PI/s_k)*s_f/s_gridWidth;

        // we have to consider the fact that pObjField is not fftshifted...
        if (xGes < s_n/2)
            xGesObj=xGes+s_n/2;
        else
            xGesObj=xGes-s_n/2;

	    // calc coordinates in fftshifted object grid
		double x=jx*s_delta1+((-double(s_n)/2)+xGesObj)*delta2;

        double phase_obj=-s_k*s_A*cos(s_kN*x);
		// multiply field with object phase phase
		pObjField[xGes+yGes*s_n]=cuCmul(pObjField[xGes+yGes*s_n],make_cuDoubleComplex(cos(phase_obj),sin(phase_obj))); 
	}
}

__global__ void createField_kernel(cuDoubleComplex* pPupField, ConfPoint_KernelParams* pParams)
{
	__shared__ double s_aberrVec[16];
	__shared__ double s_gridWidth, s_magnif, s_k, s_NA, s_deltaZ, s_apodRadius, s_f;
	__shared__ unsigned int s_n;

	// load aberration coeffs in shared memory
	if (threadIdx.x < 16)
	{
		s_aberrVec[threadIdx.x]=pParams->pAberrVec[threadIdx.x];
		if (threadIdx.x==0)
		{
			s_gridWidth=pParams->gridWidth;
			s_magnif=pParams->magnif;
			s_k=2*PI/pParams->wvl;
			s_NA=pParams->NA;
			s_n=pParams->n;
			s_deltaZ=-pParams->scanStep.z*pParams->scanNumber.z/2;
			s_apodRadius=pParams->apodisationRadius;
            s_f=160/pParams->magnif;// we assume a tubus length of 160mm here to calculate the focal length of the objective lens
		}
	}

	__syncthreads();

    unsigned int xGes=(blockIdx.x*blockDim.x+threadIdx.x);
    unsigned int yGes=(blockIdx.y*blockDim.y+threadIdx.y);
    //unsigned int xGes=(blockIdx.x*blockDim.x+threadIdx.x);
    //unsigned int yGesBase=(blockIdx.y*blockDim.y+threadIdx.y);
    //while (xGes<s_n)
    //{
	   // unsigned int yGes=yGesBase;
    //    while (yGes <s_n)
    //    {
	        if (xGes+yGes*s_n < s_n*s_n)
	        {
		        // calc coordinates in pupil grid
		        double x=double(xGes)*(s_gridWidth/s_n)-s_gridWidth/2;
		        double y=double(yGes)*(s_gridWidth/s_n)-s_gridWidth/2;
		        // calc width of pupil
		        double wPup=tan(asin(s_NA))*2*s_f;
		        double rho=sqrt(x*x+y*y)/(wPup/2); // normalized radial coordinate in pupil
		        double apodRad=s_apodRadius/(wPup/2); // normalized apodisation radius
		        double phi=atan2(y,x);

		        // calc initial defocus
		        double sigmaX=sin(atan(-x/s_f));
		        double sigmaY=sin(atan(-y/s_f));
		        double sigmaZ=(1-sigmaX*sigmaX-sigmaY*sigmaY);
		        if (sigmaZ>=0)
			        sigmaZ=sqrt(sigmaZ);
		        else
			        sigmaZ=0;

                double cosThetaZ=abs(s_f/sqrt(x*x+y*y+s_f*s_f));
                double apod=sqrt(cosThetaZ);

		        if (rho<=1)
		        {
			        // calc defocus phase
			        double phase_defoc=-s_k*sigmaZ*s_deltaZ;

			        double phase_aberr=s_k*(pParams->pAberrVec[0]
							        +pParams->pAberrVec[1]*rho*cos(phi)
							        +pParams->pAberrVec[2]*rho*sin(phi)
							        +pParams->pAberrVec[3]*(2*rho*rho-1)
							        +pParams->pAberrVec[4]*(rho*rho*cos(2*phi))
							        +pParams->pAberrVec[5]*(rho*rho*sin(2*phi))
							        +pParams->pAberrVec[6]*(3*pow(rho,3)-2*rho)*cos(phi)
							        +pParams->pAberrVec[7]*(3*pow(rho,3)-2*rho)*sin(phi)
							        +pParams->pAberrVec[8]*(6*pow(rho,4)-6*rho*rho+1)
							        +pParams->pAberrVec[9]*pow(rho,3)*cos(3*phi)
							        +pParams->pAberrVec[10]*pow(rho,3)*sin(3*phi)
							        +pParams->pAberrVec[11]*(4*pow(rho,4)-3*pow(rho,2))*cos(2*phi)
							        +pParams->pAberrVec[12]*(4*pow(rho,4)-3*pow(rho,2))*sin(2*phi)
							        +pParams->pAberrVec[13]*(10*pow(rho,5)-12*pow(rho,3)+3*rho)*cos(phi)
							        +pParams->pAberrVec[14]*(10*pow(rho,5)-12*pow(rho,3)+3*rho)*sin(phi)
							        +pParams->pAberrVec[15]*(20*pow(rho,6)-30*pow(rho,4)+12*pow(rho,2)-1));
			        double ampl=exp(-rho*rho/(apodRad*apodRad));
			        // create real and imaginary part of field with unity modulus and our phase
			        pPupField[xGes+yGes*s_n]=make_cuDoubleComplex(apod*ampl*cos(phase_aberr+phase_defoc),apod*ampl*sin(phase_aberr+phase_defoc)); 
                    //pPupField[xGes+yGes*pParams->n]=make_cuDoubleComplex(1.0,0.0); 
		        }
		        else
			        pPupField[xGes+yGes*s_n]=make_cuDoubleComplex(0, 0);
            }
    //        yGes+=blockDim.y*gridDim.y;

	   // }
    //    xGes+=blockDim.x*gridDim.x;        
    //}
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

__global__ void scalar_RichardsonWolf_kernel(cuDoubleComplex* Uin_ptr, double* x1_ptr,  double* y1_ptr, double* x2_ptr, double*y2_ptr, unsigned int dimx, unsigned int dimy, unsigned int TileWidth, unsigned int TileHeight, double wvl, double f, double Dz)
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

//****************************************************************/
// wrappers
//****************************************************************/

bool cu_scalarRichardsonWolf_wrapper(cuDoubleComplex* Uin_ptr, unsigned int dimx, unsigned int dimy, double wvl, double* x1_ptr, double* y1_ptr, double f, double Dz, double** x2_ptrptr, double** y2_ptrptr)
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
	(cudaMalloc((void**)&x2_kernel_ptr, sizeof(double)*dimx));
	double* y2_kernel_ptr;
	(cudaMalloc((void**)&y2_kernel_ptr, sizeof(double)*dimy));

	double* x1_kernel_ptr;
	(cudaMalloc((void**)&x1_kernel_ptr, sizeof(double)*dimx));
	(cudaMemcpy(x1_kernel_ptr, x1_ptr, sizeof(double)*dimx, cudaMemcpyHostToDevice));
	double* y1_kernel_ptr;
	(cudaMalloc((void**)&y1_kernel_ptr, sizeof(double)*dimy));
	(cudaMemcpy(y1_kernel_ptr, y1_ptr, sizeof(double)*dimy, cudaMemcpyHostToDevice));

	cuDoubleComplex* Uin_kernel_ptr;
	(cudaMalloc((void**)&Uin_kernel_ptr, sizeof(cuDoubleComplex)*dimx*dimy));
	(cudaMemcpy(Uin_kernel_ptr, Uin_ptr, sizeof(cuDoubleComplex)*dimx*dimy, cudaMemcpyHostToDevice));

	unsigned int tileWidth=16;
	unsigned int tileHeight=16;

	dim3 dimBlock(tileWidth,tileHeight,1); // number of threads within each block in x,y,z (maximum of 512 in total. I.e. 512,1,1 or 8,16,2 or ...
	dim3 dimGrid(dimx/tileWidth,dimy/tileHeight,1); // number of blocks in x,y,z (maximum of 65535 for each dimension)

	scalar_RichardsonWolf_kernel<<<dimGrid, dimBlock>>>((cuDoubleComplex*)Uin_kernel_ptr, x1_kernel_ptr, y1_kernel_ptr, x2_kernel_ptr, y2_kernel_ptr, dimx, dimy, tileWidth, tileHeight, wvl, f, Dz);


	// allocate host memory for observation plane coordinates
	double *x2_l=(double*)calloc(dimx,sizeof(double));
	double *y2_l=(double*)calloc(dimy,sizeof(double));

	// transfer coordinates from GPU
	(cudaMemcpy(x2_l, x2_kernel_ptr, sizeof(double)*dimx, cudaMemcpyDeviceToHost));
	(cudaMemcpy(y2_l, y2_kernel_ptr, sizeof(double)*dimy, cudaMemcpyDeviceToHost));

	//deallocate coordinates on GPU
	cudaFree(x1_kernel_ptr);
	cudaFree(x2_kernel_ptr);
	cudaFree(y1_kernel_ptr);
	cudaFree(y2_kernel_ptr);

	// do fft
    // plan fft
    cufftHandle plan;
    (cufftPlan2d(&plan,dimx, dimy, CUFFT_Z2Z));

    // execute fft
    (cufftExecZ2Z(plan, (cufftDoubleComplex *)Uin_kernel_ptr, (cufftDoubleComplex *)Uin_kernel_ptr, CUFFT_FORWARD));

	// transfer optical field from GPU
	(cudaMemcpy(Uin_ptr, Uin_kernel_ptr, sizeof(cuDoubleComplex)*dimy*dimx, cudaMemcpyDeviceToHost));
	// deallocate optical field on GPU
	cudaFree(Uin_kernel_ptr);
	// destroy fft plan
	cufftDestroy(plan);


	// return pointer to new coordinates
	*x2_ptrptr=x2_l;
	*y2_ptrptr=y2_l;

	return 1;
}

//
//void kernel_wrapper(int *a, int* b)
//{
//	int *d_1, *d_2;
//	dim3 threads( 2, 1);
//	dim3 blocks( 1, 1);
//	
//	cudaMalloc( (void **)&d_1, sizeof(int) );
//	cudaMalloc( (void **)&d_2, sizeof(int) );
//	
//    cudaMemcpy( d_1, a, sizeof(int), cudaMemcpyHostToDevice );
//    cudaMemcpy( d_2, b, sizeof(int), cudaMemcpyHostToDevice );
//
//    kernel<<< blocks, threads >>>( a, b );
//
//    cudaMemcpy( a, d_1, sizeof(int), cudaMemcpyDeviceToHost );
//    cudaMemcpy( b, d_2, sizeof(int), cudaMemcpyDeviceToHost );
//
//    cudaFree(d_1);
//    cudaFree(d_2);
//}
//
bool cu_angularSpectrumScaled_wrapper(cuDoubleComplex* Uin_ptr, unsigned int dimx, unsigned int dimy, double wvl, double* x1_ptr, double* y1_ptr, double Dz, double dx2, double** x2_ptrptr, double** y2_ptrptr)
{

    // Allocate device memory for filter kernel
    cuDoubleComplex* Uin_kernel_ptr;
    (cudaMalloc((void**)&Uin_kernel_ptr, sizeof(cuDoubleComplex)*dimx*dimy));

    // Copy host memory to device
    (cudaMemcpy(Uin_kernel_ptr, Uin_ptr, sizeof(cuDoubleComplex)*dimx*dimy,
                              cudaMemcpyHostToDevice));

    // CUFFT plan
    cufftHandle plan;
    (cufftPlan2d(&plan,dimx, dimy, CUFFT_Z2Z));

    // Transform signal and kernel
    printf("Transforming signal cufftExecC2C\n");
    (cufftExecZ2Z(plan, (cufftDoubleComplex *)Uin_kernel_ptr, (cufftDoubleComplex *)Uin_kernel_ptr, CUFFT_FORWARD));

	// copy device memory back to host
	(cudaMemcpy(Uin_ptr, Uin_kernel_ptr, sizeof(cuDoubleComplex)*dimx*dimy, cudaMemcpyDeviceToHost));

	return true;
}
//
//bool cu_angularSpectrum_ABCD(complex<double>* Uin_ptr, unsigned int dimx, unsigned int dimy, double wvl, double* x1_ptr, double* y1_ptr, double Dz, double dx2, double* ABCD, double** x2_ptrptr, double** y2_ptrptr)
//{
//	return true;
//}
//
//bool cu_fresnel_two_step_1D(complex<double>* Uin_ptr, unsigned int dimx, double wvl, double* x1_ptr, double dx2, double Dz, double** x2_ptrptr )
//{
//	return true;
//}
//
//bool cu_fresnel_two_step(complex<double>* Uin_ptr, unsigned int dimx, unsigned int dimy, double wvl, double* x1_ptr, double* y1_ptr, double dx2, double Dz, double** x2_ptrptr, double** y2_ptrptr)
//{
//	return true;
//}
//
//bool cu_fresnel_one_step_1D(complex<double>* Uin_ptr, unsigned int dimx, double wvl, double* x1_ptr, double Dz, double** x2_ptrptr )
//{
//	return true;
//}
//
//bool cu_fresnel_one_step(complex<double>* Uin_ptr, unsigned int dimx, unsigned int dimy, double wvl, double* x1_ptr, double* y1_ptr, double Dz, double** x2_ptrptr, double** y2_ptrptr)
//{
//	return true;
//}
//
//bool cu_scalar_RichardsonWolf(complex<double>* Uin_ptr, unsigned int dimx, unsigned int dimy, double wvl, double* x1_ptr, double* y1_ptr, double f, double Dz, double** x2_ptrptr, double** y2_ptrptr)
//{
//    // Allocate device memory 
//    complex<double>* Uin_kernel_ptr;
//    cutilSafeCall(cudaMalloc((void**)&Uin_kernel_ptr, sizeof(complex<double>)*dimx*dimy));
//
//    // Copy host memory to device
//    cutilSafeCall(cudaMemcpy(Uin_kernel_ptr, Uin_ptr, sizeof(complex<double>)*dimx*dimy, cudaMemcpyHostToDevice));
//
//    // Allocate device memory 
//    double* x1_kernel_ptr;
//    cutilSafeCall(cudaMalloc((void**)&x1_kernel_ptr, sizeof(double)*dimx));
//
//    // Copy host memory to device
//    cutilSafeCall(cudaMemcpy(x1_kernel_ptr, x1_ptr, sizeof(double)*dimx,
//                              cudaMemcpyHostToDevice));
//
//    // Allocate device memory 
//    double* y1_kernel_ptr;
//    cutilSafeCall(cudaMalloc((void**)&y1_kernel_ptr, sizeof(double)*dimy));
//
//    // Copy host memory to device
//    cutilSafeCall(cudaMemcpy(y1_kernel_ptr, y1_ptr, sizeof(double)*dimy,
//                              cudaMemcpyHostToDevice));
//
//	// allocate host memory
//	*x2_ptrptr=(double*)calloc(dimx,sizeof(double));
//	*y2_ptrptr=(double*)calloc(dimy,sizeof(double));
//
//    // Allocate device memory 
//    double* x2_kernel_ptr;
//    cutilSafeCall(cudaMalloc((void**)&x2_kernel_ptr, sizeof(double)*dimx));
//
//    // Allocate device memory 
//    double* y2_kernel_ptr;
//    cutilSafeCall(cudaMalloc((void**)&y2_kernel_ptr, sizeof(double)*dimy));
//	// do the scaling
//	cu_scalar_RichardsonWolf_kernel<<<32,512>>>(reinterpret_cast<cufftDoubleComplex*>(Uin_kernel_ptr), dimx, dimy, wvl, x1_kernel_ptr, y1_kernel_ptr, f, Dz, x2_kernel_ptr, y2_kernel_ptr);
//
//	// do the fft
//    // CUFFT plan
//    cufftHandle plan;
//    cufftSafeCall(cufftPlan2d(&plan,dimx, dimy, CUFFT_Z2Z));
//
//    // execution
//    cufftSafeCall(cufftExecZ2Z(plan, (cufftDoubleComplex *)Uin_kernel_ptr, (cufftDoubleComplex *)Uin_kernel_ptr, CUFFT_FORWARD));
//
//	// do the ffthift in a kernel....
//
//	// copy device memory back to host
//	cutilSafeCall(cudaMemcpy(Uin_ptr, Uin_kernel_ptr, sizeof(complex<double>)*dimx*dimy, cudaMemcpyDeviceToHost));
//	cutilSafeCall(cudaMemcpy(*x2_ptrptr, x2_kernel_ptr, sizeof(double)*dimx, cudaMemcpyDeviceToHost));
//	cutilSafeCall(cudaMemcpy(*y2_ptrptr, y2_kernel_ptr, sizeof(double)*dimy, cudaMemcpyDeviceToHost));
//
//	return true;
//}
//
//bool cu_fraunhofer(complex<double>* Uin_ptr, unsigned int dimx, unsigned int dimy, double wvl, double* x1_ptr, double* y1_ptr, double Dz, double** x2_ptrptr, double** y2_ptrptr)
//{
//	return true;
//}
//
//bool cu_fraunhofer_1D(complex<double>* Uin_ptr, unsigned int dimx, double wvl, double* x1_ptr, double Dz, double** x2_ptrptr)
//{
//	return true;
//}
//
//bool cu_fftshift(complex<double>* in, unsigned int dimx, unsigned int dimy)
//{
//	return true;
//}
//

/*
    This version uses n/2 threads --
    it performs the first level of reduction when reading from global memory.
*/
__global__ void reduce_kernel3_overlap(cuDoubleComplex *g_idata, cuDoubleComplex *g_odata, unsigned int n)
{
    cuDoubleComplex *sdata = SharedMemory<cuDoubleComplex>();

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;

    cuDoubleComplex mySum = (i < n) ? cuCmul(g_idata[i],g_idata[i]) : make_cuDoubleComplex(0.0,0.0);

    if (i + blockDim.x < n)
        mySum = cuCadd(mySum, cuCmul(g_idata[i+blockDim.x],g_idata[i+blockDim.x]));

    sdata[tid] = mySum;
    __syncthreads();

    // do reduction in shared mem
    for (unsigned int s=blockDim.x/2; s>0; s>>=1)
    {
        if (tid < s)
        {
            sdata[tid] = mySum = cuCadd(mySum, sdata[tid + s]);
        }

        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

/*
    This version uses n/2 threads --
    it performs the first level of reduction when reading from global memory.
*/
__global__ void reduce_kernel3_final(cuDoubleComplex *g_idata, double *g_odata, unsigned int *g_index, unsigned int n)
{
    cuDoubleComplex *sdata = SharedMemory<cuDoubleComplex>();

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;

    cuDoubleComplex mySum = (i < n) ? g_idata[i] : make_cuDoubleComplex(0.0,0.0);

    if (i + blockDim.x < n)
        mySum = cuCadd(mySum, g_idata[i+blockDim.x]);

    sdata[tid] = mySum;
    __syncthreads();

    // do reduction in shared mem
    for (unsigned int s=blockDim.x/2; s>0; s>>=1)
    {
        if (tid < s)
        {
            sdata[tid] = mySum = cuCadd(mySum, sdata[tid + s]);
        }

        __syncthreads();
    }

    // write result for this block to global mem for raw signal
    if (tid == 0) 
	{
		g_odata[(*g_index)] = cuCabs(sdata[0])*cuCabs(sdata[0]);
		*g_index=*g_index+1; // increment index
	}
}

/*
    This version uses n/2 threads --
    it performs the first level of reduction when reading from global memory.
*/
__global__ void reduce_kernel3(cuDoubleComplex *g_idata, cuDoubleComplex *g_odata, unsigned int n)
{
    cuDoubleComplex *sdata = SharedMemory<cuDoubleComplex>();

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;

    cuDoubleComplex mySum = (i < n) ? g_idata[i] : make_cuDoubleComplex(0.0,0.0);

    if (i + blockDim.x < n)
        mySum = cuCadd(mySum, g_idata[i+blockDim.x]);

    sdata[tid] = mySum;
    __syncthreads();

    // do reduction in shared mem
    for (unsigned int s=blockDim.x/2; s>0; s>>=1)
    {
        if (tid < s)
        {
            sdata[tid] = mySum = cuCadd(mySum, sdata[tid + s]);
        }

        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

/*
    This version adds multiple elements per thread sequentially.  This reduces the overall
    cost of the algorithm while keeping the work complexity O(n) and the step complexity O(log n).
    (Brent's Theorem optimization)

    Note, this kernel needs a minimum of 64*sizeof(T) bytes of shared memory.
    In other words if blockSize <= 32, allocate 64*sizeof(T) bytes.
    If blockSize > 32, allocate blockSize*sizeof(T) bytes.
*/
template <class T, unsigned int blockSize, bool nIsPow2> __global__ void reduce6(T *g_idata, T *g_odata, unsigned int n)
{
    T *sdata = SharedMemory<T>();

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;

    T mySum = 0;

    // we reduce multiple elements per thread.  The number is determined by the
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < n)
    {
        mySum += g_idata[i];

        // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
        if (nIsPow2 || i + blockSize < n)
            mySum += g_idata[i+blockSize];

        i += gridSize;
    }

    // each thread puts its local sum into shared memory
    sdata[tid] = mySum;
    __syncthreads();


    // do reduction in shared mem
    if (blockSize >= 512)
    {
        if (tid < 256)
        {
            sdata[tid] = mySum = mySum + sdata[tid + 256];
        }

        __syncthreads();
    }

    if (blockSize >= 256)
    {
        if (tid < 128)
        {
            sdata[tid] = mySum = mySum + sdata[tid + 128];
        }

        __syncthreads();
    }

    if (blockSize >= 128)
    {
        if (tid <  64)
        {
            sdata[tid] = mySum = mySum + sdata[tid +  64];
        }

        __syncthreads();
    }

    if (tid < 32)
    {
        // now that we are using warp-synchronous programming (below)
        // we need to declare our shared memory volatile so that the compiler
        // doesn't reorder stores to it and induce incorrect behavior.
        volatile T *smem = sdata;

        if (blockSize >=  64)
        {
            smem[tid] = mySum = mySum + smem[tid + 32];
        }

        if (blockSize >=  32)
        {
            smem[tid] = mySum = mySum + smem[tid + 16];
        }

        if (blockSize >=  16)
        {
            smem[tid] = mySum = mySum + smem[tid +  8];
        }

        if (blockSize >=   8)
        {
            smem[tid] = mySum = mySum + smem[tid +  4];
        }

        if (blockSize >=   4)
        {
            smem[tid] = mySum = mySum + smem[tid +  2];
        }

        if (blockSize >=   2)
        {
            smem[tid] = mySum = mySum + smem[tid +  1];
        }
    }

    // write result for this block to global mem
    if (tid == 0)
        g_odata[blockIdx.x] = sdata[0];
}

////////////////////////////////////////////////////////////////////////////////
// This function performs a reduction of the input data multiple times and
// measures the average reduction time.
////////////////////////////////////////////////////////////////////////////////
void reduceOverlap(int  n,
                  int  numThreads,
                  int  numBlocks,
                  int  maxThreads,
                  int  maxBlocks,
                  cuDoubleComplex *d_idata,
                  cuDoubleComplex *d_odata,
				  double *d_rawSig,
				  double *h_rawSig,
				  unsigned int *d_index)
{
	dim3 dimBlock(numThreads,1,1);
	dim3 dimGrid(numBlocks, 1, 1);

    // when there is only one warp per block, we need to allocate two warps
    // worth of shared memory so that we don't index shared memory out of bounds
    int smemSize = (numThreads <= 32) ? 2 * numThreads * sizeof(cuDoubleComplex) : numThreads * sizeof(cuDoubleComplex);
    
	// execute the kernel (with squaring of the field values)
    reduce_kernel3_overlap<<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, n);

	cuDoubleComplex *h_pOutData;
	h_pOutData=(cuDoubleComplex*)malloc(numBlocks*sizeof(cuDoubleComplex));
	cudaMemcpy(h_pOutData, d_odata, numBlocks*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);

    // sum partial block sums on GPU
    int s=numBlocks;

    while (s > 1)
    {
        int l_threads = 0, l_blocks = 0;
        getNumBlocksAndThreads(3, s, maxBlocks, maxThreads, l_blocks, l_threads);
		dim3 l_dimBlock(l_threads,1,1);
		dim3 l_dimGrid(l_blocks,1,1);

		// when there is only one warp per block, we need to allocate two warps
		// worth of shared memory so that we don't index shared memory out of bounds
		int smemSize = (numThreads <= 32) ? 2 * numThreads * sizeof(cuDoubleComplex) : numThreads * sizeof(cuDoubleComplex);


		// the last reduction executes in only one block. The result of this is the final result of our overlap integral and needs to be saved in global memory for the raw signal
		if (l_blocks==1)
			reduce_kernel3_final<<<l_dimGrid, l_dimBlock, smemSize>>>(d_odata, d_rawSig, d_index, s);
		else
			// execute pure reduction kernel
			reduce_kernel3<<<l_dimGrid, l_dimBlock, smemSize>>>(d_odata, d_odata, s);

		// update number of active blocks
        s = (s + (l_threads*2-1)) / (l_threads*2);
    }
	unsigned int index;
	cudaMemcpy(&index, d_index, sizeof(unsigned int), cudaMemcpyDeviceToHost);

    // copy final sum from device to host
    cudaMemcpy(h_rawSig, d_rawSig, sizeof(double), cudaMemcpyDeviceToHost);
}

///*
//    This version adds multiple elements per thread sequentially.  This reduces the overall
//    cost of the algorithm while keeping the work complexity O(n) and the step complexity O(log n).
//    (Brent's Theorem optimization)
//
//    Note, this kernel needs a minimum of 64*sizeof(T) bytes of shared memory.
//    In other words if blockSize <= 32, allocate 64*sizeof(T) bytes.
//    If blockSize > 32, allocate blockSize*sizeof(T) bytes.
//*/
//template <unsigned int blockSize, bool nIsPow2> __global__ void calcOverlap(cuDoubleComplex *g_idata, cuDoubleComplex *g_odata, unsigned int n)
//{
//    cuDoubleComplex *sdata = SharedMemory<cuDoubleComplex>();
//
//    // perform first level of reduction,
//    // reading from global memory, writing to shared memory
//    unsigned int tid = threadIdx.x;
//    unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
//    unsigned int gridSize = blockSize*2*gridDim.x;
//
//    cuDoubleComplex mySum = make_cuDoubleComplex(0.0,0.0);
//
//    // we reduce multiple elements per thread.  The number is determined by the
//    // number of active thread blocks (via gridDim).  More blocks will result
//    // in a larger gridSize and therefore fewer elements per thread
//    while (i < n)
//    {
//        mySum = cuCadd(mySum, cuCmul(g_idata[i],g_idata[i]));
//
//        // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
//        if (nIsPow2 || i + blockSize < n)
//            mySum = cuCadd(mySum, cuCmul(g_idata[i+blockSize],g_idata[i+blockSize]));
//
//        i += gridSize;
//    }
//
//    // each thread puts its local sum into shared memory
//    sdata[tid] = mySum;
//    __syncthreads();
//
//
//    // do reduction in shared mem
//    if (blockSize >= 512)
//    {
//        if (tid < 256)
//        {
//            sdata[tid] = mySum = cuCadd(mySum,sdata[tid + 256]);
//        }
//
//        __syncthreads();
//    }
//
//    if (blockSize >= 256)
//    {
//        if (tid < 128)
//        {
//            sdata[tid] = mySum = cuCadd(mySum, sdata[tid + 128]);
//        }
//
//        __syncthreads();
//    }
//
//    if (blockSize >= 128)
//    {
//        if (tid <  64)
//        {
//            sdata[tid] = mySum = cuCadd(mySum, sdata[tid +  64]);
//        }
//
//        __syncthreads();
//    }
//
//    if (tid < 32)
//    {
//        // now that we are using warp-synchronous programming (below)
//        // we need to declare our shared memory volatile so that the compiler
//        // doesn't reorder stores to it and induce incorrect behavior.
//        //volatile cuDoubleComplex *smem = sdata;
//		volatile cuDoubleComplex *smem = sdata;
//
//        if (blockSize >=  64)
//        {
//            smem[tid] = mySum = cuCadd(mySum, smem[tid + 32]);
//        }
//
//        if (blockSize >=  32)
//        {
//			mySum=cuCadd(mySum,  smem[tid + 16]);
//            smem[tid] = mySum;// = cuCadd(mySum, smem[tid + 16]);
//        }
//
//        if (blockSize >=  16)
//        {
//            smem[tid] = mySum = cuCadd(mySum, smem[tid +  8]);
//        }
//
//        if (blockSize >=   8)
//        {
//            smem[tid] = mySum = cuCadd(mySum, smem[tid +  4]);
//        }
//
//        if (blockSize >=   4)
//        {
//            smem[tid] = mySum = cuCadd(mySum, smem[tid +  2]);
//        }
//
//        if (blockSize >=   2)
//        {
//            smem[tid] = mySum = cuCadd(mySum, smem[tid +  1]);
//        }
//    }
//
//    // write result for this block to global mem
//    //if (tid == 0)
//    //    g_odata[blockIdx.x] = sdata[0];
//}

double cu_testReduce_wrapper()
{
    cudaDeviceProp deviceProp;
    cudaError_t error;

	// check device
    error = cudaGetDeviceProperties(&deviceProp, 0);

    if (error != cudaSuccess)
    {
        std::cout << "error in cu_simConfPointRawSig_wrapper: cudaGetDeviceProperties returned error code " << error << " line: " << __LINE__ << "...\n";
        return 0;
    }
	// use a larger block size for Fermi and above
	int block_size = (deviceProp.major < 2) ? 16 : 32;

	int n=1024*1024;
	cuDoubleComplex *inData;
	inData=(cuDoubleComplex*)malloc(n*sizeof(cuDoubleComplex));
	cuDoubleComplex *d_inData;

	for (unsigned int i=0; i<n; ++i)
	{
		inData[i]=make_cuDoubleComplex(2.0,0.0);
	}
	if (cudaSuccess != cudaMalloc((void**)&d_inData, n*sizeof(cuDoubleComplex)))
	{
		std::cout << "error in cu_simConfPointRawSig_wrapper: cudaMalloc returned an error " << error << " line: " << __LINE__ << "...\n";
		return 0;
	}

	// raw signal
	double *d_rawSig;
	double rawSig=0.0;

	if (cudaSuccess != cudaMalloc((void**)&d_rawSig, sizeof(double)))
	{
		std::cout << "error in cu_simConfPointRawSig_wrapper: cudaMalloc returned an error " << error << " line: " << __LINE__ << "...\n";
		return 0;
	}
	if (cudaSuccess != cudaMemcpy(d_rawSig, &rawSig, sizeof(double), cudaMemcpyHostToDevice))
	{
		std::cout << "error in cu_simConfPointRawSig_wrapper: cudaMemcpy returned an error " << error << " line: " << __LINE__ << "...\n";
		return 0;
	}

	// index of raw signal
	unsigned int *d_index;
	unsigned int index=0;

	if (cudaSuccess != cudaMalloc((void**)&d_index, sizeof(unsigned int)))
	{
		std::cout << "error in cu_simConfPointRawSig_wrapper: cudaMalloc returned an error " << error << " line: " << __LINE__ << "...\n";
		return 0;
	}
	if (cudaSuccess != cudaMemcpy(d_index, &index, sizeof(unsigned int), cudaMemcpyHostToDevice))
	{
		std::cout << "error in cu_simConfPointRawSig_wrapper: cudaMemcpy returned an error " << error << " line: " << __LINE__ << "...\n";
		return 0;
	}

	int blocksReduction;
	int threadsReduction;
	int maxBlocks=deviceProp.maxGridSize[0];//64; // why this number??
	int maxThreads=deviceProp.maxThreadsPerBlock;

	getNumBlocksAndThreads(3, n, maxBlocks, maxThreads, blocksReduction, threadsReduction);
	dim3 dimBlockReduction(threadsReduction,1,1); 
	dim3 dimGridReduction(blocksReduction,1,1);
    // when there is only one warp per block, we need to allocate two warps
    // worth of shared memory so that we don't index shared memory out of bounds
    int smemSize = (threadsReduction <= block_size) ? 2 * threadsReduction * sizeof(cuDoubleComplex) : threadsReduction * sizeof(cuDoubleComplex);

	cuDoubleComplex *d_outData;
	if (cudaSuccess != cudaMalloc((void**)&d_outData, blocksReduction*sizeof(cuDoubleComplex)))
	{
		std::cout << "error in cu_simConfPointRawSig_wrapper: cudaMalloc returned an error " << error << " line: " << __LINE__ << "...\n";
		return 0;
	}
	double outData=0;
	// do the summation
	reduceOverlap(n, threadsReduction, blocksReduction, maxThreads, maxBlocks, d_inData, d_outData, d_rawSig, &outData, d_index);
	//reduceOverlap(params.n*params.n, threadsReduction, blocksReduction, maxThreads, maxBlocks, d_pObjField, d_pOutData, h_pRawSig, d_index);
	//switch (n)
	//{
	//case 32:
	//	calcOverlap<32, true><<<dimGrid, dimBlock, smemSize>>>(d_inData,  d_outData, n); break;
	//case 64:
	//	calcOverlap<64, true><<<dimGrid, dimBlock, smemSize>>>(d_inData,  d_outData, n); break;
	//}
	cudaFree(d_outData);
	cudaFree(d_inData);
	delete inData;
	return 1.0;//outData;
}

__global__ void innerProduct(Lock lock, cuDoubleComplex *field, cuDoubleComplex *out, int *outIdx, int N)
{
    __shared__ cuDoubleComplex cache[THREADSPERBLOCK];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIdx=threadIdx.x;

    cuDoubleComplex temp=make_cuDoubleComplex(0.0, 0.0);

    // square each element
    while (tid < N)
    {
        temp=cuCadd(temp, cuCmul(field[tid], field[tid]));
        tid += blockDim.x*gridDim.x;
    }

    // set the cache values
    cache[cacheIdx]=temp;

    // synchronize threads in this block
    __syncthreads();

    // for reductions, threadsPerBlock must be a power of two because of the following code
    int i=blockDim.x/2;
    while (i != 0)
    {
        if (cacheIdx < i)
            cache[cacheIdx] = cuCadd(cache[cacheIdx], cache[cacheIdx+i]);
        __syncthreads();
        i = i/2;
    }

    if (cacheIdx==0)
    {
        lock.lock();
        out[*outIdx]=cuCadd(out[*outIdx], cache[0]);
        lock.unlock();
    }
}

__global__ void incrementIdx(int *idx)
{
    *idx+=1;
}

__global__ void myReduce(Lock lock, cuDoubleComplex *field, cuDoubleComplex *out, int N)
{
    __shared__ cuDoubleComplex cache[THREADSPERBLOCK];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIdx=threadIdx.x;

    cuDoubleComplex temp=make_cuDoubleComplex(0.0, 0.0);

    // square each element
    while (tid < N)
    {
        temp=cuCadd(temp, cuCmul(field[tid], field[tid]));
        tid += blockDim.x*gridDim.x;
    }

    // set the cache values
    cache[cacheIdx]=temp;

    // synchronize threads in this block
    __syncthreads();

    // for reductions, threadsPerBlock must be a power of two because of the following code
    int i=blockDim.x/2;
    while (i != 0)
    {
        if (cacheIdx < i)
            cache[cacheIdx] = cuCadd(cache[cacheIdx], cache[cacheIdx+i]);
        __syncthreads();
        i = i/2;
    }

    if (cacheIdx==0)
    {
        lock.lock();
        *out=cuCadd(*out, cache[0]);
        lock.unlock();
    }

}

bool cu_simConfPointRawSig_wrapperTest(double** ppRawSig, ConfPoint_KernelParams params)
{
    size_t N = params.n*params.n;
    cudaDeviceProp deviceProp;
    cudaError_t error;

	// check device
    error = cudaGetDeviceProperties(&deviceProp, 0);
    if (error != cudaSuccess)
    {
        std::cout << "error in cu_simConfPointRawSig_wrapper: cudaGetDeviceProperties returned error code " << error << " line: " << __LINE__ << "...\n";
        return 0;
    }

    cuDoubleComplex * l_pHostData=(cuDoubleComplex*)malloc(N*sizeof(cuDoubleComplex));
    for (int i=0;i<N;i++)
    {
	    l_pHostData[i]=make_cuDoubleComplex(10000.0,5000.0);
    }

    // obtain raw pointer to device memory
    cuDoubleComplex * l_pDeviceData;
    cudaMalloc((void **) &l_pDeviceData, N * sizeof(cuDoubleComplex));

    error=cudaMemcpy(l_pDeviceData, l_pHostData,N*sizeof(cuDoubleComplex),cudaMemcpyHostToDevice);
	// transfer data to GPU
	if (error != cudaSuccess)
    {
        std::cout << "error in cu_simConfPointRawSig_wrapper: cudaMemcpy returned error code " << error << " line: " << __LINE__ << "...\n";
        return 0;
    }

    *ppRawSig=(double*)calloc(1,sizeof(double));

    cuDoubleComplex l_hostOutData=make_cuDoubleComplex(0.0,0.0);
	cuDoubleComplex *l_pDeviceOutData;
	if (cudaSuccess != cudaMalloc((void**)&l_pDeviceOutData, sizeof(cuDoubleComplex)))
	{
		std::cout << "error in cu_simConfPointRawSig_wrapper: cudaMalloc returned an error " << error << " line: " << __LINE__ << "...\n";
		return 0;
	}
    if (cudaSuccess != cudaMemcpy(l_pDeviceOutData, &l_hostOutData, sizeof(cuDoubleComplex), cudaMemcpyHostToDevice) )
    {
		std::cout << "error in cu_simConfPointRawSig_wrapper: cudaMalloc returned an error " << error << " line: " << __LINE__ << "...\n";
		return 0;
    }

    Lock lock;
    int blocksPerGrid=(N+THREADSPERBLOCK-1)/THREADSPERBLOCK;
    myReduce<<<blocksPerGrid, 1024>>>(lock, l_pDeviceData, l_pDeviceOutData, N);

    // copy data back from GPU
    if (cudaSuccess != cudaMemcpy(&l_hostOutData, l_pDeviceOutData, sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost) )
    {
		std::cout << "error in cu_simConfPointRawSig_wrapper: cudaMalloc returned an error " << error << " line: " << __LINE__ << "...\n";
		return 0;
    }
    (*ppRawSig)[0]=cuCabs(l_hostOutData);

    // free memory
    cudaFree(l_pDeviceData);
    cudaFree(l_pDeviceOutData);
    delete l_pHostData;
    
    return true;
}

bool cu_simConfPointSensorSig_wrapper(double** ppSensorSig, ConfPoint_KernelParams params, ConfPoint_KernelObjectParams paramsObject)
{
    cudaDeviceProp deviceProp;
    cudaError_t error;

     // timing stuff
    clock_t start, end, startGes, endGes;
	double msecs_DataTransfer=0;
    double msecs_ObjectInteraction=0;
	double msecs_Defoc=0;
    double msecs_FFT=0;
    double msecs_Reduce=0;
    double msecs_createField=0;
	double msecs=0;
	// start timing
	start=clock();
    startGes=clock();

	// check device
    error = cudaGetDeviceProperties(&deviceProp, 0);

    if (error != cudaSuccess)
    {
        std::cout << "error in cu_simConfPointSensorSig_wrapper: cudaGetDeviceProperties returned error code " << error << " line: " << __LINE__ << "...\n";
        return false;
    }

	// allocate host memory for raw signal
	(*ppSensorSig)=(double*)calloc(params.scanNumber.x*params.scanNumber.y,sizeof(double));

    // allocate host memory for complex raw signal
    cuDoubleComplex* l_pRawSig=(cuDoubleComplex*)calloc(params.scanNumber.z,sizeof(cuDoubleComplex));
    cuDoubleComplex* l_pRawSigInit=(cuDoubleComplex*)calloc(params.scanNumber.z,sizeof(cuDoubleComplex));

    double* l_pAbsVal=(double*)calloc(params.scanNumber.z,sizeof(double));

    // allocate device memory for raw signal
	cuDoubleComplex* d_pRawSig;
	if (cudaSuccess != cudaMalloc((void**)&d_pRawSig, params.scanNumber.z*sizeof(cuDoubleComplex)))
	{
		std::cout << "error in cu_simConfPointSensorSig_wrapper: cudaMalloc returned an error " << error << " line: " << __LINE__ << "...\n";
		return false;
	}
	// transfer rawSig to device
	//if (cudaSuccess != cudaMemcpy(d_pRawSig, l_pRawSig, params.scanNumber.z*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice))
	//{
	//	std::cout << "error in cu_simConfPointSensorSig_wrapper: cudaMemcpy returned an error " << error << " line: " << __LINE__ << "...\n";
	//	return false;
	//}

	// use a larger block size for Fermi and above
	int block_size = (deviceProp.major < 2) ? 16 : 32;

	// allocate device meory for params
	ConfPoint_KernelParams* d_pParams;
	if (cudaSuccess != cudaMalloc((void**)&d_pParams, sizeof(ConfPoint_KernelParams)))
	{
		std::cout << "error in cu_simConfPointSensorSig_wrapper: cudaMalloc returned an error " << error << " line: " << __LINE__ << "...\n";
		return false;
	}
	// transfer params to device
	if (cudaSuccess != cudaMemcpy(d_pParams, &params, sizeof(ConfPoint_KernelParams), cudaMemcpyHostToDevice))
	{
		std::cout << "error in cu_simConfPointSensorSig_wrapper: cudaMemcpy returned an error " << error << " line: " << __LINE__ << "...\n";
		return false;
	}

	// allocate device meory for params
	ConfPoint_KernelObjectParams* d_pObjParams;
	if (cudaSuccess != cudaMalloc((void**)&d_pObjParams, sizeof(ConfPoint_KernelObjectParams)))
	{
		std::cout << "error in cu_simConfPointSensorSig_wrapper: cudaMalloc returned an error " << error << " line: " << __LINE__ << "...\n";
		return false;
	}
	// transfer params to device
	if (cudaSuccess != cudaMemcpy(d_pObjParams, &paramsObject, sizeof(ConfPoint_KernelObjectParams), cudaMemcpyHostToDevice))
	{
		std::cout << "error in cu_simConfPointSensorSig_wrapper: cudaMemcpy returned an error " << error << " line: " << __LINE__ << "...\n";
		return false;
	}

	// allocate device memory for pupil field
	cuDoubleComplex* d_pPupField;
	if (cudaSuccess != cudaMalloc((void**)&d_pPupField, params.n*params.n*sizeof(cuDoubleComplex)))
	{
		std::cout << "error in cu_simConfPointSensorSig_wrapper: cudaMalloc returned an error " << error << " line: " << __LINE__ << "...\n";
		return false;
	}
	// allocate device memory for object field
	cuDoubleComplex* d_pObjField;
	if (cudaSuccess != cudaMalloc((void**)&d_pObjField, params.n*params.n*sizeof(cuDoubleComplex)))
	{
		std::cout << "error in cu_simConfPointSensorSig_wrapper: cudaMalloc returned an error " << error << " line: " << __LINE__ << "...\n";
		return false;
	}


	// calc dimensions of kernel launch when have one kernel per element in the pupil field
    dim3 dimBlock(block_size,block_size,1); // number of threads within each block in x,y,z (maximum of 512 or 1024 in total. I.e. 512,1,1 or 8,16,2 or ...
	unsigned int mod= params.n % block_size;
	unsigned int dimGridx;
	if (mod==0)
		dimGridx=params.n/(1*block_size);
	else
		dimGridx=params.n/(1*block_size+1);
	unsigned int dimGridy;
	if (mod==0)
		dimGridy=params.n/(1*block_size);
	else
		dimGridy=params.n/(1*block_size+1);
	dim3 dimGrid(std::max(dimGridx,unsigned int(1)),std::max(dimGridy,unsigned int(1)),1); // number of blocks in x,y,z (maximum of 65535 for each dimension)

    cudaDeviceSynchronize();
    end=clock();
	msecs_DataTransfer=((end-start)/(double)CLOCKS_PER_SEC*1000.0);

    start=clock();

    cufftHandle plan;
	if (!myCufftSafeCall(cufftPlan2d(&plan,params.n, params.n, CUFFT_Z2Z)))
	{
		std::cout << "error in cu_simConfPointSensorSig_wrapper: cufftPlan2d returned an error " << error << " line: " << __LINE__ << "...\n";
		return false;
	}

    // create rawSigIdx on GPU
    int rawSigIdx=0;
    int *d_pRawSigIdx;
	if (cudaSuccess != cudaMalloc((void**)&d_pRawSigIdx, sizeof(int)))
	{
		std::cout << "error in cu_simConfPointSensorSig_wrapper: cudaMalloc returned an error " << error << " line: " << __LINE__ << "...\n";
		return 0;
	}
	// transfer rawSig to device
	if (cudaSuccess != cudaMemcpy(d_pRawSigIdx, &rawSigIdx, sizeof(int), cudaMemcpyHostToDevice))
	{
		std::cout << "error in cu_simConfPointSensorSig_wrapper: cudaMemcpy returned an error " << error << " line: " << __LINE__ << "...\n";
		return false;
	}
    // create cog on GPU
    double cog=0;
    double *d_pCog;
	if (cudaSuccess != cudaMalloc((void**)&d_pCog, sizeof(double)))
	{
		std::cout << "error in cu_simConfPointSensorSig_wrapper: cudaMalloc returned an error " << error << " line: " << __LINE__ << "...\n";
		return 0;
	}
	// transfer cog to device
	if (cudaSuccess != cudaMemcpy(d_pCog, &cog, sizeof(int), cudaMemcpyHostToDevice))
	{
		std::cout << "error in cu_simConfPointSensorSig_wrapper: cudaMemcpy returned an error " << error << " line: " << __LINE__ << "...\n";
		return false;
	}

    cudaDeviceSynchronize();
    end=clock();
    msecs_DataTransfer+=((end-start)/(double)CLOCKS_PER_SEC*1000.0);

    Lock lock;
    int blocksPerGrid=(params.n*params.n+THREADSPERBLOCK-1)/THREADSPERBLOCK;

   	// allocate host memory for pupil field
	cuDoubleComplex* h_pPupField=(cuDoubleComplex*)malloc(params.n*params.n*sizeof(cuDoubleComplex));


	// do the simulation
	for (unsigned int jy=0; jy<params.scanNumber.y; jy++)
	{
		for (unsigned int jx=0; jx<params.scanNumber.x; jx++)
		{
            start=clock();
	        // create pupil field according to aberrations
	        createField_kernel<<<dimGrid,dimBlock>>>(d_pPupField, d_pParams);
            cudaDeviceSynchronize();
            end=clock();
            msecs_createField+=((end-start)/(double)CLOCKS_PER_SEC*1000.0);
	        // reset rawSigIdx
	        if (cudaSuccess != cudaMemcpy(d_pRawSigIdx, &rawSigIdx, sizeof(int), cudaMemcpyHostToDevice))
	        {
		        std::cout << "error in cu_simConfPointSensorSig_wrapper: cudaMemcpy returned an error " << error << " line: " << __LINE__ << "...\n";
		        return false;
	        }
            cudaDeviceSynchronize();
            // reset rawSig
	        if (cudaSuccess != cudaMemcpy(d_pRawSig, l_pRawSigInit, params.scanNumber.z*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice))
	        {
		        std::cout << "error in cu_simConfPointSensorSig_wrapper: cudaMemcpy returned an error " << error << " line: " << __LINE__ << "...\n";
		        return false;
	        }
            cudaDeviceSynchronize();
            for (unsigned int jz=0; jz<params.scanNumber.z; jz++)
			{
                start=clock();
				// apply defocus
				defocField_kernel<<<dimGrid,dimBlock>>>(d_pPupField, d_pParams);
                cudaDeviceSynchronize();
                end=clock();
                msecs_Defoc+=((end-start)/(double)CLOCKS_PER_SEC*1000.0);

	// transfer pupil field from device
	if (cudaSuccess != cudaMemcpy(h_pPupField, d_pPupField, params.n*params.n*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost))
		return false;

	char t_filename[512];
	sprintf(t_filename, "E:\\pupReal%i.txt", jz);
	FILE* hFile;
	hFile = fopen( t_filename, "w" ) ;


	if ( (hFile == NULL) )
		return 1;


	for (unsigned int jy=0; jy<params.n; jy++)
	{
		for (unsigned int jx=0; jx<params.n; jx++)
		{
			fprintf(hFile, " %.16e;\n", h_pPupField[jx+jy*params.n].x);
		}
	}

	fclose(hFile);

	sprintf(t_filename, "E:\\pupImag%i.txt", jz);
	hFile = fopen( t_filename, "w" ) ;


	if ( (hFile == NULL) )
		return 1;


	for (unsigned int jy=0; jy<params.n; jy++)
	{
		for (unsigned int jx=0; jx<params.n; jx++)
		{
			fprintf(hFile, " %.16e;\n", h_pPupField[jx+jy*params.n].y);
		}
	}

	fclose(hFile);

                start=clock();
				// note that object field is not fftshifted after call to cufft !!
				if (!myCufftSafeCall(cufftExecZ2Z(plan, (cufftDoubleComplex *)d_pPupField, (cufftDoubleComplex *)d_pObjField, CUFFT_FORWARD)))
				{
                    // try again, just to be sure...
                    if (!myCufftSafeCall(cufftExecZ2Z(plan, (cufftDoubleComplex *)d_pPupField, (cufftDoubleComplex *)d_pObjField, CUFFT_FORWARD)))
                    {
					    cudaFree(d_pParams);
					    cudaFree(d_pPupField);
					    cudaFree(d_pObjField);
                        cudaFree(d_pRawSig);
                        cudaFree(d_pObjParams);
                        cudaFree(d_pRawSigIdx);
					    cufftDestroy (plan);
                        delete l_pRawSig;
                        delete l_pRawSigInit;
					    //thrust::device_free(d_pObjField_thrust);
					    std::cout << "error in cu_simConfPointSensorSig_wrapper: cufftExecZ2Z returned an error " << error << " line: " << __LINE__ << "...\n";
					    return false;
                    }
				}
                cudaDeviceSynchronize();
                end=clock();
                msecs_FFT+=((end-start)/(double)CLOCKS_PER_SEC*1000.0);

	// transfer object field from device
	if (cudaSuccess != cudaMemcpy(h_pPupField, d_pObjField, params.n*params.n*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost))
		return false;

	sprintf(t_filename, "E:\\objInReal%i.txt", jz);
	hFile = fopen( t_filename, "w" ) ;


	if ( (hFile == NULL) )
		return 1;


	for (unsigned int jy=0; jy<params.n; jy++)
	{
		for (unsigned int jx=0; jx<params.n; jx++)
		{
			fprintf(hFile, " %.16e;\n", h_pPupField[jx+jy*params.n].x);
		}
	}

	fclose(hFile);

	sprintf(t_filename, "E:\\objInImag%i.txt", jz);
	hFile = fopen( t_filename, "w" ) ;


	if ( (hFile == NULL) )
		return 1;


	for (unsigned int jy=0; jy<params.n; jy++)
	{
		for (unsigned int jx=0; jx<params.n; jx++)
		{
			fprintf(hFile, " %.16e;\n", h_pPupField[jx+jy*params.n].y);
		}
	}

	fclose(hFile);
    
                // do the object interaction in TEA
                objectInteractionTEA<<<dimGrid,dimBlock>>>(d_pObjField, d_pParams, d_pObjParams, jx);

	// allocate host memory for pupil field
	// transfer pupil field from device
	if (cudaSuccess != cudaMemcpy(h_pPupField, d_pObjField, params.n*params.n*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost))
		return false;

	sprintf(t_filename, "E:\\objOutReal%i.txt", jz);
	hFile = fopen( t_filename, "w" ) ;


	if ( (hFile == NULL) )
		return 1;


	for (unsigned int jy=0; jy<params.n; jy++)
	{
		for (unsigned int jx=0; jx<params.n; jx++)
		{
			fprintf(hFile, " %.16e;\n", h_pPupField[jx+jy*params.n].x);
		}
	}

	fclose(hFile);

	sprintf(t_filename, "E:\\objOutImag%i.txt", jz);
	hFile = fopen( t_filename, "w" ) ;


	if ( (hFile == NULL) )
		return 1;


	for (unsigned int jy=0; jy<params.n; jy++)
	{
		for (unsigned int jx=0; jx<params.n; jx++)
		{
			fprintf(hFile, " %.16e;\n", h_pPupField[jx+jy*params.n].y);
		}
	}

	fclose(hFile);

                start=clock();
                // calc the inner product on GPU
                innerProduct<<<blocksPerGrid, THREADSPERBLOCK>>>(lock, d_pObjField, d_pRawSig, d_pRawSigIdx, params.n*params.n);
                incrementIdx<<<1,1>>>(d_pRawSigIdx);
                cudaDeviceSynchronize();
                end=clock();
                msecs_Reduce+=((end-start)/(double)CLOCKS_PER_SEC*1000.0);

			}

            start=clock();
            // copy data back from GPU
            if (cudaSuccess != cudaMemcpy(l_pRawSig, d_pRawSig, params.scanNumber.z*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost) )
            {
                std::cout << "error in cu_simConfPointSensorSig_wrapper: cudaMemcpy returned an error " << error << " line: " << __LINE__ << "...\n";
                return 0;
            }
            cudaDeviceSynchronize();
            end=clock();
            msecs_DataTransfer+=((end-start)/(double)CLOCKS_PER_SEC*1000.0);


                    char t_filename[512];
	                sprintf(t_filename, "E:\\rawSigReal%i.txt", jx);
	                FILE* hFile;
	                hFile = fopen( t_filename, "w" ) ;


	                if ( (hFile == NULL) )
		                return 1;


                        for (unsigned int idz=0; idz<params.scanNumber.z; idz++)
		                {
                            fprintf(hFile, " %.16e;\n", l_pRawSig[idz].x);
		                }

	                fclose(hFile);

	                sprintf(t_filename, "E:\\rawSigImag%i.txt", jx);
	                hFile = fopen( t_filename, "w" ) ;


	                if ( (hFile == NULL) )
		                return 1;


                        for (unsigned int idz=0; idz<params.scanNumber.z; idz++)
		                {
                            fprintf(hFile, " %.16e;\n", l_pRawSig[idz].y);
		                }

	                fclose(hFile);

            // find signal maxmium
            double sigMax=0;
            for (unsigned int idx=0; idx<params.scanNumber.z; idx++)
            {
                l_pAbsVal[idx]=pow(cuCabs(l_pRawSig[idx]),2);
                sigMax=(sigMax > l_pAbsVal[idx]) ? sigMax : l_pAbsVal[idx];
            }

            // calc cog
            double nom=0;
            double denom=0;
            for (unsigned int idx=0; idx<params.scanNumber.z; idx++)
            {
                if (l_pAbsVal[idx] > sigMax/2)
                {
                    nom+=double(idx)*l_pAbsVal[idx];
                    denom+=l_pAbsVal[idx];
                }
            }
            double x=jx*params.scanStep.x;
            double z0=paramsObject.A*cos(paramsObject.kN*x);
            (*ppSensorSig)[jx+jy*params.scanNumber.y]=nom/denom*params.scanStep.z-params.scanStep.z*params.scanNumber.z/2+z0;
		}
	}

    std::cout << msecs_DataTransfer << "msec for data transfer between CPU and GPU" << "\n";
    std::cout << msecs_FFT << "msec for fft" << "\n";
    std::cout << msecs_Defoc << "msec for defocus kernel" << "\n";
    std::cout << msecs_createField << "msec to create the field" << "\n";
    std::cout << msecs_Reduce << "msec to calculate the reduction" << "\n";    
    std::cout << msecs_ObjectInteraction << "msec for calculating object interaction" << "\n";

	// end timing
	endGes=clock();
	msecs=((endGes-startGes)/(double)CLOCKS_PER_SEC*1000.0);
	std::cout << msecs <<" ms to simulate confocal raw signal"<< "...\n";

    cudaFree(d_pObjParams);
	cudaFree(d_pParams);
	cudaFree(d_pPupField);
	cudaFree(d_pObjField);
    cudaFree(d_pObjParams);
    cudaFree(d_pRawSigIdx);
    delete l_pRawSig;
    delete l_pRawSigInit;

	cufftDestroy (plan);

	return true;
}

bool cu_simConfPointRawSig_wrapper(double** ppRawSig, ConfPoint_KernelParams params)
{
    cudaDeviceProp deviceProp;
    cudaError_t error;

     // timing stuff
    clock_t start, end, startGes, endGes;
	double msecs_DataTransfer=0;
    double msecs_DataTransfer1=0;
	double msecs_Defoc=0;
    double msecs_FFT=0;
    double msecs_Reduce=0;
    double msecs_createField=0;
	double msecs=0;
	// start timing
	start=clock();
    startGes=clock();

	// check device
    error = cudaGetDeviceProperties(&deviceProp, 0);

    if (error != cudaSuccess)
    {
        std::cout << "error in cu_simConfPointRawSig_wrapper: cudaGetDeviceProperties returned error code " << error << " line: " << __LINE__ << "...\n";
        return false;
    }

	// allocate host memory for raw signal
	*ppRawSig=(double*)calloc(params.scanNumber.x*params.scanNumber.y*params.scanNumber.z,sizeof(double));

    // allocate host memory for complex raw signal
    cuDoubleComplex* l_pRawSig=(cuDoubleComplex*)calloc(params.scanNumber.x*params.scanNumber.y*params.scanNumber.z,sizeof(cuDoubleComplex));

    // allocate device memory for raw signal
	cuDoubleComplex* d_pRawSig;
	if (cudaSuccess != cudaMalloc((void**)&d_pRawSig, params.scanNumber.x*params.scanNumber.y*params.scanNumber.z*sizeof(cuDoubleComplex)))
	{
		std::cout << "error in cu_simConfPointRawSig_wrapper: cudaMalloc returned an error " << error << " line: " << __LINE__ << "...\n";
		return false;
	}
	// transfer rawSig to device
	if (cudaSuccess != cudaMemcpy(d_pRawSig, l_pRawSig, params.scanNumber.x*params.scanNumber.y*params.scanNumber.z*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice))
	{
		std::cout << "error in cu_simConfPointRawSig_wrapper: cudaMemcpy returned an error " << error << " line: " << __LINE__ << "...\n";
		return false;
	}

	// use a larger block size for Fermi and above
	int block_size = (deviceProp.major < 2) ? 16 : 32;

	// allocate device meory for params
	ConfPoint_KernelParams* d_pParams;
	if (cudaSuccess != cudaMalloc((void**)&d_pParams, sizeof(ConfPoint_KernelParams)))
	{
		std::cout << "error in cu_simConfPointRawSig_wrapper: cudaMalloc returned an error " << error << " line: " << __LINE__ << "...\n";
		return false;
	}
	// transfer params to device
	if (cudaSuccess != cudaMemcpy(d_pParams, &params, sizeof(ConfPoint_KernelParams), cudaMemcpyHostToDevice))
	{
		std::cout << "error in cu_simConfPointRawSig_wrapper: cudaMemcpy returned an error " << error << " line: " << __LINE__ << "...\n";
		return false;
	}

	// allocate device memory for pupil field
	cuDoubleComplex* d_pPupField;
	if (cudaSuccess != cudaMalloc((void**)&d_pPupField, params.n*params.n*sizeof(cuDoubleComplex)))
	{
		std::cout << "error in cu_simConfPointRawSig_wrapper: cudaMalloc returned an error " << error << " line: " << __LINE__ << "...\n";
		return false;
	}
	// allocate device memory for object field
	cuDoubleComplex* d_pObjField;
	if (cudaSuccess != cudaMalloc((void**)&d_pObjField, params.n*params.n*sizeof(cuDoubleComplex)))
	{
		std::cout << "error in cu_simConfPointRawSig_wrapper: cudaMalloc returned an error " << error << " line: " << __LINE__ << "...\n";
		return false;
	}


	// calc dimensions of kernel launch when have one kernel per element in the pupil field
    dim3 dimBlock(block_size,block_size,1); // number of threads within each block in x,y,z (maximum of 512 or 1024 in total. I.e. 512,1,1 or 8,16,2 or ...
	unsigned int mod= params.n % block_size;
	unsigned int dimGridx;
	if (mod==0)
		//dimGridx=params.n/(8*block_size);
        dimGridx=params.n/(1*block_size);
	else
		dimGridx=params.n/(1*block_size+1);
	unsigned int dimGridy;
	if (mod==0)
		dimGridy=params.n/(1*block_size);
	else
		dimGridy=params.n/(1*block_size+1);
	dim3 dimGrid(std::max(dimGridx,unsigned int(1)),std::max(dimGridy,unsigned int(1)),1); // number of blocks in x,y,z (maximum of 65535 for each dimension)

    cudaDeviceSynchronize();
    end=clock();
	msecs_DataTransfer=((end-start)/(double)CLOCKS_PER_SEC*1000.0);

    start=clock();
	// create pupil field according to aberrations
	createField_kernel<<<dimGrid,dimBlock>>>(d_pPupField, d_pParams);
    cudaDeviceSynchronize();
    end=clock();
    msecs_createField=((end-start)/(double)CLOCKS_PER_SEC*1000.0);

	//// allocate host memory for pupil field
	//cuDoubleComplex* h_pPupField=(cuDoubleComplex*)malloc(params.n*params.n*sizeof(cuDoubleComplex));
	//// transfer pupil field from device
	//if (cudaSuccess != cudaMemcpy(h_pPupField, d_pPupField, params.n*params.n*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost))
	//	return false;

	//char t_filename[512];
	//sprintf(t_filename, "E:\\testReal.txt");
	//FILE* hFile;
	//hFile = fopen( t_filename, "w" ) ;


	//if ( (hFile == NULL) )
	//	return 1;


	//for (unsigned int jy=0; jy<params.n; jy++)
	//{
	//	for (unsigned int jx=0; jx<params.n; jx++)
	//	{
	//		fprintf(hFile, " %.16e;\n", h_pPupField[jx+jy*params.n].x);
	//	}
	//}

	//fclose(hFile);

	//sprintf(t_filename, "E:\\testImag.txt");
	//hFile = fopen( t_filename, "w" ) ;


	//if ( (hFile == NULL) )
	//	return 1;


	//for (unsigned int jy=0; jy<params.n; jy++)
	//{
	//	for (unsigned int jx=0; jx<params.n; jx++)
	//	{
	//		fprintf(hFile, " %.16e;\n", h_pPupField[jx+jy*params.n].y);
	//	}
	//}

	//fclose(hFile);

    start=clock();

    cufftHandle plan;
	if (!myCufftSafeCall(cufftPlan2d(&plan,params.n, params.n, CUFFT_Z2Z)))
	{
		std::cout << "error in cu_simConfPointRawSig_wrapper: cufftPlan2d returned an error " << error << " line: " << __LINE__ << "...\n";
		return false;
	}

    // create rawSigIdx on GPU
    int rawSigIdx=0;
    int *d_pRawSigIdx;
	if (cudaSuccess != cudaMalloc((void**)&d_pRawSigIdx, sizeof(int)))
	{
		std::cout << "error in cu_simConfPointRawSig_wrapper: cudaMalloc returned an error " << error << " line: " << __LINE__ << "...\n";
		return 0;
	}
	// transfer rawSig to device
	if (cudaSuccess != cudaMemcpy(d_pRawSigIdx, &rawSigIdx, sizeof(int), cudaMemcpyHostToDevice))
	{
		std::cout << "error in cu_simConfPointRawSig_wrapper: cudaMemcpy returned an error " << error << " line: " << __LINE__ << "...\n";
		return false;
	}
    cudaDeviceSynchronize();
    end=clock();
    msecs_DataTransfer+=((end-start)/(double)CLOCKS_PER_SEC*1000.0);

    Lock lock;
    int blocksPerGrid=(params.n*params.n+THREADSPERBLOCK-1)/THREADSPERBLOCK;

	// do the simulation
	for (unsigned int jy=0; jy<params.scanNumber.y; jy++)
	{
		for (unsigned int jx=0; jx<params.scanNumber.x; jx++)
		{
			for (unsigned int jz=0; jz<params.scanNumber.z; jz++)
			{
                start=clock();
				// apply defocus
				defocField_kernel<<<dimGrid,dimBlock>>>(d_pPupField, d_pParams);
                cudaDeviceSynchronize();
                end=clock();
                msecs_Defoc+=((end-start)/(double)CLOCKS_PER_SEC*1000.0);

                start=clock();
				// note that object field is not fftshifted after call to cufft !!
				if (!myCufftSafeCall(cufftExecZ2Z(plan, (cufftDoubleComplex *)d_pPupField, (cufftDoubleComplex *)d_pObjField, CUFFT_FORWARD)))
				{
                    // try again, just to be sure...
                    if (!myCufftSafeCall(cufftExecZ2Z(plan, (cufftDoubleComplex *)d_pPupField, (cufftDoubleComplex *)d_pObjField, CUFFT_FORWARD)))
                    {
					    cudaFree(d_pParams);
					    cudaFree(d_pPupField);
					    cudaFree(d_pObjField);
                        cudaFree(d_pRawSig);
                        cudaFree(d_pRawSigIdx);
					    cufftDestroy (plan);
					    //thrust::device_free(d_pObjField_thrust);
					    std::cout << "error in cu_simConfPointRawSig_wrapper: cufftExecZ2Z returned an error " << error << " line: " << __LINE__ << "...\n";
					    return false;
                    }
				}
                cudaDeviceSynchronize();
                end=clock();
                msecs_FFT+=((end-start)/(double)CLOCKS_PER_SEC*1000.0);

                start=clock();
                // calc the inner product on GPU
                innerProduct<<<blocksPerGrid, THREADSPERBLOCK>>>(lock, d_pObjField, d_pRawSig, d_pRawSigIdx, params.n*params.n);
                incrementIdx<<<1,1>>>(d_pRawSigIdx);
                cudaDeviceSynchronize();
                end=clock();
                msecs_Reduce+=((end-start)/(double)CLOCKS_PER_SEC*1000.0);
			}
		}
	}

    clock_t start1=clock();
    // copy data back from GPU
    if (cudaSuccess != cudaMemcpy(l_pRawSig, d_pRawSig, params.scanNumber.x*params.scanNumber.y*params.scanNumber.z*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost) )
    {
        std::cout << "error in cu_simConfPointRawSig_wrapper: cudaMemcpy returned an error " << error << " line: " << __LINE__ << "...\n";
        return 0;
    }
    cudaDeviceSynchronize();
    clock_t end1=clock();
    msecs_DataTransfer1=((end1-start1)/(double)CLOCKS_PER_SEC*1000.0);

    std::cout << msecs_DataTransfer << "msec for data transfer between CPU and GPU" << "\n";
    std::cout << msecs_DataTransfer1 << "msec for data transfer1 between CPU and GPU" << "\n";
    std::cout << msecs_FFT << "msec for fft" << "\n";
    std::cout << msecs_Defoc << "msec for defocus kernel" << "\n";
    std::cout << msecs_createField << "msec to create the field" << "\n";
    std::cout << msecs_Reduce << "msec to calculate the reduction" << "\n";    


    // calc magnitude square
    for (unsigned int idx=0; idx<params.scanNumber.x*params.scanNumber.y*params.scanNumber.z; idx++)
    {
        (*ppRawSig)[idx]=pow(cuCabs(l_pRawSig[idx]),2);
    }

	// end timing
	endGes=clock();
	msecs=((endGes-startGes)/(double)CLOCKS_PER_SEC*1000.0);
	std::cout << msecs <<" ms to simulate confocal raw signal"<< "...\n";

    cudaFree(d_pParams);
    cudaFree(d_pPupField);
    cudaFree(d_pObjField);
    cudaFree(d_pRawSig);
    cudaFree(d_pRawSigIdx);

    delete l_pRawSig;

	cufftDestroy (plan);

	return true;
}

bool cu_simConfPointRawSig_wrapper1(double** ppRawSig, ConfPoint_KernelParams params)
{
    cudaDeviceProp deviceProp;
    cudaError_t error;

	// check device
    error = cudaGetDeviceProperties(&deviceProp, 0);

    if (error != cudaSuccess)
    {
        std::cout << "error in cu_simConfPointRawSig_wrapper: cudaGetDeviceProperties returned error code " << error << " line: " << __LINE__ << "...\n";
        return false;
    }
	// use a larger block size for Fermi and above
	int block_size = (deviceProp.major < 2) ? 16 : 32;

	// allocate device meory for params
	ConfPoint_KernelParams* d_pParams;
	if (cudaSuccess != cudaMalloc((void**)&d_pParams, sizeof(ConfPoint_KernelParams)))
	{
		std::cout << "error in cu_simConfPointRawSig_wrapper: cudaMalloc returned an error " << error << " line: " << __LINE__ << "...\n";
		return false;
	}
	// transfer params to device
	if (cudaSuccess != cudaMemcpy(d_pParams, &params, sizeof(ConfPoint_KernelParams), cudaMemcpyHostToDevice))
	{
		std::cout << "error in cu_simConfPointRawSig_wrapper: cudaMemcpy returned an error " << error << " line: " << __LINE__ << "...\n";
		return false;
	}

	// allocate device memory for pupil field
	cuDoubleComplex* d_pPupField;
	if (cudaSuccess != cudaMalloc((void**)&d_pPupField, params.n*params.n*sizeof(cuDoubleComplex)))
	{
		std::cout << "error in cu_simConfPointRawSig_wrapper: cudaMalloc returned an error " << error << " line: " << __LINE__ << "...\n";
		return false;
	}
	// allocate device memory for object field
	cuDoubleComplex* d_pObjField;
	if (cudaSuccess != cudaMalloc((void**)&d_pObjField, params.n*params.n*sizeof(cuDoubleComplex)))
	{
		std::cout << "error in cu_simConfPointRawSig_wrapper: cudaMalloc returned an error " << error << " line: " << __LINE__ << "...\n";
		return false;
	}

	// calc dimensions of kernel launch when have one kernel per element in the pupil field
    dim3 dimBlock(block_size,block_size,1); // number of threads within each block in x,y,z (maximum of 512 or 1024 in total. I.e. 512,1,1 or 8,16,2 or ...
	unsigned int mod= params.n % block_size;
	unsigned int dimGridx;
	if (mod==0)
		dimGridx=params.n/block_size;
	else
		dimGridx=params.n/block_size+1;
	unsigned int dimGridy;
	if (mod==0)
		dimGridy=params.n/block_size;
	else
		dimGridy=params.n/block_size+1;
	dim3 dimGrid(std::max(dimGridx,unsigned int(1)),std::max(dimGridy,unsigned int(1)),1); // number of blocks in x,y,z (maximum of 65535 for each dimension)
	// create pupil field according to aberrations
	createField_kernel<<<dimGrid,dimBlock>>>(d_pPupField, d_pParams);
	
    cufftHandle plan;
	if (!myCufftSafeCall(cufftPlan2d(&plan,params.n, params.n, CUFFT_Z2Z)))
	{
		std::cout << "error in cu_simConfPointRawSig_wrapper: cufftPlan2d returned an error " << error << " line: " << __LINE__ << "...\n";
		return false;
	}
	
	// calc dimensions of kernel launch for reduction
	int blocksReduction;
	int threadsReduction;
	int maxBlocks=1024; // why this number??
	int maxThreads=deviceProp.maxThreadsPerBlock;
	getNumBlocksAndThreads(3, params.n*params.n, maxBlocks, maxThreads, blocksReduction, threadsReduction);
	dim3 dimBlockReduction(threadsReduction,1,1); 
	dim3 dimGridReduction(blocksReduction,1,1);

	cuDoubleComplex *d_pOutData;
	if (cudaSuccess != cudaMalloc((void**)&d_pOutData, blocksReduction*sizeof(cuDoubleComplex)))
	{
		std::cout << "error in cu_simConfPointRawSig_wrapper: cudaMalloc returned an error " << error << " line: " << __LINE__ << "...\n";
		return 100;
	}

	// alloacte host memory fpr raw signal
	double rawSig=0.0;
	double *h_pRawSig=(double*)malloc(params.scanNumber.x*params.scanNumber.y*params.scanNumber.z*sizeof(double));

	// allocate device memory for raw signal
	*ppRawSig=(double*)malloc(params.scanNumber.x*params.scanNumber.y*params.scanNumber.z*sizeof(double));
	// allocate device memory for raw signal
	double* d_pRawSig;
	if (cudaSuccess != cudaMalloc((void**)&d_pRawSig, params.scanNumber.x*params.scanNumber.y*params.scanNumber.z*sizeof(double)))
	{
		std::cout << "error in cu_simConfPointRawSig_wrapper: cudaMalloc returned an error " << error << " line: " << __LINE__ << "...\n";
		return false;
	}

	// index of raw signal
	unsigned int *d_index;
	unsigned int index=0;

	if (cudaSuccess != cudaMalloc((void**)&d_index, sizeof(unsigned int)))
	{
		std::cout << "error in cu_simConfPointRawSig_wrapper: cudaMalloc returned an error " << error << " line: " << __LINE__ << "...\n";
		return 0;
	}
	if (cudaSuccess != cudaMemcpy(d_index, &index, sizeof(unsigned int), cudaMemcpyHostToDevice))
	{
		std::cout << "error in cu_simConfPointRawSig_wrapper: cudaMemcpy returned an error " << error << " line: " << __LINE__ << "...\n";
		return 0;
	}

    // when there is only one warp per block, we need to allocate two warps
    // worth of shared memory so that we don't index shared memory out of bounds
    int smemSize = (threadsReduction <= block_size) ? 2 * threadsReduction * sizeof(cuDoubleComplex) : threadsReduction * sizeof(cuDoubleComplex);

	clock_t start, end;
	double msecs=0;
	// start timing
	start=clock();

	// do the simulation
	for (unsigned int jy=0; jy<params.scanNumber.y; jy++)
	{
		for (unsigned int jx=0; jx<params.scanNumber.x; jx++)
		{
			for (unsigned int jz=0; jz<params.scanNumber.z; jz++)
			{
				// apply defocus
				defocField_kernel<<<dimGrid,dimBlock>>>(d_pPupField, d_pParams);
				// note that object field is not fftshifted after call to cufft !!
				if (!myCufftSafeCall(cufftExecZ2Z(plan, (cufftDoubleComplex *)d_pPupField, (cufftDoubleComplex *)d_pObjField, CUFFT_FORWARD)))
				{
					cudaFree(d_pParams);
					cudaFree(d_pPupField);
					cudaFree(d_pObjField);
					cudaFree(d_pRawSig);
					cufftDestroy (plan);
					{
						std::cout << "error in cu_simConfPointRawSig_wrapper: cufftExecZ2Z returned an error " << error << " line: " << __LINE__ << "...\n";
						return false;
					}
				}
				// do the summation
				reduceOverlap(params.n*params.n, threadsReduction, blocksReduction, maxThreads, maxBlocks, d_pObjField, d_pOutData, d_pRawSig, h_pRawSig, d_index);
			}
		}
	}

	cudaMemcpy(h_pRawSig, d_pRawSig, params.scanNumber.x*params.scanNumber.y*params.scanNumber.z*sizeof(double), cudaMemcpyDeviceToHost);

	// end timing
	end=clock();
	msecs=((end-start)/(double)CLOCKS_PER_SEC*1000.0);
	std::cout << msecs <<" ms to simulate confocal raw signal"<< "...\n";

	cudaFree(d_pParams);
	cudaFree(d_pPupField);
	cudaFree(d_pObjField);
	cudaFree(d_pRawSig);
	cufftDestroy (plan);

	return true;
}