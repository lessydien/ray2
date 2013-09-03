#ifndef __kernel__
	#define __kernel__

#include "vector_types.h"
#include "cuComplex.h"
#include "cufft.h"
#include <iostream>
//#include "PropagationMath.h"

//#define PI ((double)3.141592653589793238462643383279502884197169399375105820)


/* declare class */
/**
  *\class	ConfPoint_KernelParams
  *\brief	  
  *
  *         
  *
  *         \todo
  *         \remarks           
  *         \sa       NA
  *         \date     17.041.2013
  *         \author  Mauch
  *
  */
class ConfPoint_KernelParams
{
public:
	double NA;						// numerical aperture of objective lens
	double magnif;					// magnification of objective lens
	double wvl;						// wavlength of illumination in mm
	unsigned int n;					// number of samples of pupil field per dimensions
	double gridWidth;				// physical width of grid of pupil field in mm
	double3 scanStep;				// physical step width of scanning in x, y and z in mm
	uint3 scanNumber;				// number of scan steps in x, y and z
	double pAberrVec[16];			// vector containing the zernike coefficients of the aberrations of the pupil field
	double apodisationRadius;		// 1/e-radius of a gaussian apodisation function

	//ConfPoint_KernelParams& ConfPoint_KernelParams::operator=(const ConfPoint_Params& op)
	//{
	//	this->gridWidth=op.gridWidth;
	//	this->magnif=op.magnif;
	//	this->wvl=op.wvl;
	//	this->n=op.n;
	//	this->scanNumber=op.scanNumber;
	//	this->scanStep=op.scanStep;
	//	memcpy(&(this->pAberrVec[0]),&(op.pAberrVec[0]),16*sizeof(double));
	//	return *this;
	//}
};

double cu_testReduce_wrapper();
bool cu_simConfPointRawSig_wrapper(double** ppRawSig, ConfPoint_KernelParams params);
bool cu_simConfPointRawSig_wrapper1(double** ppRawSig, ConfPoint_KernelParams params);
//extern "C" void kernel_wrapper(int *a, int *b);
bool cu_angularSpectrumScaled_wrapper(cuDoubleComplex* Uin_ptr, unsigned int dimx, unsigned int dimy, double wvl, double* x1_ptr, double* y1_ptr, double Dz, double dx2, double** x2_ptrptr, double** y2_ptrptr);
//extern "C" bool cu_angularSpectrum_ABCD(complex<double>* Uin_ptr, unsigned int dimx, unsigned int dimy, double wvl, double* x1_ptr, double* y1_ptr, double Dz, double dx2, double* ABCD, double** x2_ptrptr, double** y2_ptrptr);
//extern "C" bool cu_fresnel_two_step(complex<double>* Uin_ptr, unsigned int dimx, unsigned int dimy, double wvl, double* x1_ptr, double* y1_ptr, double dx2, double Dz, double** x2_ptrptr, double** y2_ptrptr);
//extern "C" bool cu_fresnel_one_step_1D(complex<double>* Uin_ptr, unsigned int dimx, double wvl, double* x1_ptr, double Dz, double** x2_ptrptr );
//extern "C" bool cu_fresnel_one_step(complex<double>* Uin_ptr, unsigned int dimx, unsigned int dimy, double wvl, double* x1_ptr, double* y1_ptr, double Dz, double** x2_ptrptr, double** y2_ptrptr);
bool cu_scalarRichardsonWolf_wrapper(cuDoubleComplex* Uin_ptr, unsigned int dimx, unsigned int dimy, double wvl, double* x1_ptr, double* y1_ptr, double f, double Dz, double** x2_ptrptr, double** y2_ptrptr);
//extern "C" bool cu_fraunhofer(complex<double>* Uin_ptr, unsigned int dimx, unsigned int dimy, double wvl, double* x1_ptr, double* y1_ptr, double Dz, double** x2_ptrptr, double** y2_ptrptr);
//extern "C" bool cu_fraunhofer_1D(complex<double>* Uin_ptr, unsigned int dimx, double wvl, double* x1_ptr, double Dz, double** x2_ptrptr);
//extern "C" bool cu_fftshift(complex<double>* in, unsigned int dimx, unsigned int dimy);
//extern "C" bool cu_calcWavefront(complex<double>* Uin_ptr, unsigned int dimx, unsigned int dimy, double* x_ptr, double *y_ptr, double D, double *coeffVec_ptr);

inline __host__ __device__ uint2 calc2DIndices(uint3 blockIndex, dim3 blockDimension, uint3 threadIndex, unsigned int dimX)
{
	uint2 idx2D;
	// calc 2D indices
	unsigned int iGes = blockIndex.x * blockDimension.x + threadIndex.x;
	idx2D.x = iGes % dimX;
	idx2D.y = floorf(iGes/dimX);
	return idx2D;
}

#define myCufftSafeCall(err)           __myCufftSafeCall     (err, __FILE__, __LINE__)

inline bool __myCufftSafeCall( cufftResult err, const char *file, const int line )
{
    if( CUFFT_SUCCESS != err) {
		std::cout << "cufftSafeCall() Runtime API error in file " << file << " line " << line << std::endl;
        return false;
    }
	return true;
}

#endif