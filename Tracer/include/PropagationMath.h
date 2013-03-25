#define _USE_MATH_DEFINES
#include <cmath>

//#include "cuda.h"
//#include "cuda_runtime.h"
//#include <cufft.h>
//#include "cutil_inline.h"

#include <complex>
#include "fftw3.h"
#include "kernel.h"
using namespace std;

// prototypes for cuda wrappers
//extern void kernel_wrapper(int *a, int *b);


typedef enum 
{
  PROP_NO_ERR,
  PROP_ERR
} propError;


//extern bool cu_angularSpectrum_scaled(complex<double>* Uin_ptr, unsigned int dimx, unsigned int dimy, double wvl, double* x1_ptr, double* y1_ptr, double Dz, double dx2, double** x2_ptrptr, double** y2_ptrptr);
//extern bool cu_angularSpectrum_ABCD(complex<double>* Uin_ptr, unsigned int dimx, unsigned int dimy, double wvl, double* x1_ptr, double* y1_ptr, double Dz, double dx2, double* ABCD, double** x2_ptrptr, double** y2_ptrptr);
//extern bool cu_fresnel_two_step_1D(complex<double>* Uin_ptr, unsigned int dimx, double wvl, double* x1_ptr, double dx2, double Dz, double** x2_ptrptr );
//extern bool cu_fresnel_two_step(complex<double>* Uin_ptr, unsigned int dimx, unsigned int dimy, double wvl, double* x1_ptr, double* y1_ptr, double dx2, double Dz, double** x2_ptrptr, double** y2_ptrptr);
//extern bool cu_fresnel_one_step_1D(complex<double>* Uin_ptr, unsigned int dimx, double wvl, double* x1_ptr, double Dz, double** x2_ptrptr );
//extern bool cu_fresnel_one_step(complex<double>* Uin_ptr, unsigned int dimx, unsigned int dimy, double wvl, double* x1_ptr, double* y1_ptr, double Dz, double** x2_ptrptr, double** y2_ptrptr);
//extern bool cu_scalar_RichardsonWolf(complex<double>* Uin_ptr, unsigned int dimx, unsigned int dimy, double wvl, double* x1_ptr, double* y1_ptr, double f, double Dz, double** x2_ptrptr, double** y2_ptrptr);
//extern bool cu_fraunhofer(complex<double>* Uin_ptr, unsigned int dimx, unsigned int dimy, double wvl, double* x1_ptr, double* y1_ptr, double Dz, double** x2_ptrptr, double** y2_ptrptr);
//extern bool cu_fraunhofer_1D(complex<double>* Uin_ptr, unsigned int dimx, double wvl, double* x1_ptr, double Dz, double** x2_ptrptr);
//extern bool cu_fftshift(complex<double>* in, unsigned int dimx, unsigned int dimy);
//extern bool cu_calcWavefront(complex<double>* Uin_ptr, unsigned int dimx, unsigned int dimy, double* x_ptr, double *y_ptr, double D, double *coeffVec_ptr);

propError cu_scalar_RichardsonWolf2(complex<double>* Uin_ptr, unsigned int dimx, unsigned int dimy, double wvl, double* x1_ptr, double* y1_ptr, double f, double Dz, double** x2_ptrptr, double** y2_ptrptr);

propError cu_scalar_RichardsonWolf(complex<double>* Uin_ptr, unsigned int dimx, unsigned int dimy, double wvl, double* x1_ptr, double* y1_ptr, double f, double Dz, double** x2_ptrptr, double** y2_ptrptr);

propError cu_ft2(complex<double>* Uin_ptr, unsigned int dimx, unsigned int dimy, double delta);

//propError fftshift(complex<double>* Uin_ptr, unsigned int dimx, unsigned int dimy);
propError fresnel_two_step_1D(complex<double>* Uin_ptr, unsigned int dimx, double wvl, double* x1_ptr, double dx2, double Dz, double** x2_ptrptr );
propError angularSpectrum_scaled(complex<double>* Uin_ptr, unsigned int dimx, unsigned int dimy, double wvl, double* x1_ptr, double* y1_ptr, double Dz, double dx2, double** x2_ptrptr, double** y2_ptrptr);
propError angularSpectrum_ABCD(complex<double>* Uin_ptr, unsigned int dimx, unsigned int dimy, double wvl, double* x1_ptr, double* y1_ptr, double Dz, double dx2, double* ABCD, double** x2_ptrptr, double** y2_ptrptr);
propError fresnel_two_step_1D(complex<double>* Uin_ptr, unsigned int dimx, double wvl, double* x1_ptr, double dx2, double Dz, double** x2_ptrptr );
propError fresnel_two_step(complex<double>* Uin_ptr, unsigned int dimx, unsigned int dimy, double wvl, double* x1_ptr, double* y1_ptr, double dx2, double Dz, double** x2_ptrptr, double** y2_ptrptr);
propError fresnel_one_step_1D(complex<double>* Uin_ptr, unsigned int dimx, double wvl, double* x1_ptr, double Dz, double** x2_ptrptr );
propError fresnel_one_step(complex<double>* Uin_ptr, unsigned int dimx, unsigned int dimy, double wvl, double* x1_ptr, double* y1_ptr, double Dz, double** x2_ptrptr, double** y2_ptrptr);
propError scalar_RichardsonWolf(complex<double>* Uin_ptr, unsigned int dimx, unsigned int dimy, double wvl, double* x1_ptr, double* y1_ptr, double f, double Dz, double** x2_ptrptr, double** y2_ptrptr);
propError fraunhofer(complex<double>* Uin_ptr, unsigned int dimx, unsigned int dimy, double wvl, double* x1_ptr, double* y1_ptr, double Dz, double** x2_ptrptr, double** y2_ptrptr);
propError fraunhofer_1D(complex<double>* Uin_ptr, unsigned int dimx, double wvl, double* x1_ptr, double Dz, double** x2_ptrptr);
propError fftshift(complex<double>* in, unsigned int dimx, unsigned int dimy);
propError calcWavefront(complex<double>* Uin_ptr, unsigned int dimx, unsigned int dimy, double* x_ptr, double *y_ptr, double D, double wvl, double *coeffVec_ptr);

// all the functions that use calls to fftw need to have an extra implementation that is thread safe by taking a global plan as an additional input
propError fresnel_two_step_1D_ts(complex<double>* Uin_ptr, unsigned int dimx, double wvl, double* x1_ptr, double dx2, double Dz, double** x2_ptrptr , fftw_plan &p);
propError angularSpectrum_scaled_ts(complex<double>* Uin_ptr, unsigned int dimx, unsigned int dimy, double wvl, double* x1_ptr, double* y1_ptr, double Dz, double dx2, double** x2_ptrptr, double** y2_ptrptr, fftw_plan &p_fw, fftw_plan &p_bw);
propError angularSpectrum_ABCD_ts(complex<double>* Uin_ptr, unsigned int dimx, unsigned int dimy, double wvl, double* x1_ptr, double* y1_ptr, double Dz, double dx2, double* ABCD, double** x2_ptrptr, double** y2_ptrptr, fftw_plan &p_fw, fftw_plan &p_bw);
propError fresnel_two_step_1D_ts(complex<double>* Uin_ptr, unsigned int dimx, double wvl, double* x1_ptr, double dx2, double Dz, double** x2_ptrptr , fftw_plan &p);
propError fresnel_two_step_ts(complex<double>* Uin_ptr, unsigned int dimx, unsigned int dimy, double wvl, double* x1_ptr, double* y1_ptr, double dx2, double Dz, double** x2_ptrptr, double** y2_ptrptr, fftw_plan &p);
propError fresnel_one_step_1D_ts(complex<double>* Uin_ptr, unsigned int dimx, double wvl, double* x1_ptr, double Dz, double** x2_ptrptr , fftw_plan &p);
propError fresnel_one_step_ts(complex<double>* Uin_ptr, unsigned int dimx, unsigned int dimy, double wvl, double* x1_ptr, double* y1_ptr, double Dz, double** x2_ptrptr, double** y2_ptrptr, fftw_plan &p);
propError scalar_RichardsonWolf_ts(complex<double>* Uin_ptr, unsigned int dimx, unsigned int dimy, double wvl, double* x1_ptr, double* y1_ptr, double f, double Dz, double** x2_ptrptr, double** y2_ptrptr, fftw_plan &p);
propError fraunhofer_ts(complex<double>* Uin_ptr, unsigned int dimx, unsigned int dimy, double wvl, double* x1_ptr, double* y1_ptr, double Dz, double** x2_ptrptr, double** y2_ptrptr, fftw_plan &p);
propError fraunhofer_1D_ts(complex<double>* Uin_ptr, unsigned int dimx, double wvl, double* x1_ptr, double Dz, double** x2_ptrptr, fftw_plan &p);


inline propError ft1(complex<double>* Uin_ptr, unsigned int dimx, double delta)
{
	fftshift(Uin_ptr, dimx, 1);

	fftw_complex *in=reinterpret_cast<fftw_complex*>(Uin_ptr);
	fftw_plan p;

	p = fftw_plan_dft_1d(dimx, in, in, FFTW_FORWARD, FFTW_ESTIMATE);

	fftw_execute(p);

	fftshift(Uin_ptr, dimx, 1);

	for (unsigned long jx=0; jx<dimx; jx++)
	{
		Uin_ptr[jx]=Uin_ptr[jx]*delta;
	}
	return PROP_NO_ERR;
}

inline propError ft2(complex<double>* Uin_ptr, unsigned int dimx, unsigned int dimy, double delta)
{
	if (dimx!=dimy)
		return PROP_ERR;
	fftshift(Uin_ptr, dimx, dimy);

	fftw_complex *in=reinterpret_cast<fftw_complex*>(Uin_ptr);
	fftw_plan p;

	p = fftw_plan_dft_2d(dimx, dimy, in, in, FFTW_FORWARD, FFTW_ESTIMATE);

	fftw_execute(p);

	fftshift(Uin_ptr, dimx, dimy);

	for (unsigned long jx=0; jx<dimx; jx++)
	{
		for (unsigned long jy=0; jy<dimy;jy++)
		{
			Uin_ptr[jx+jy*dimy]=Uin_ptr[jx+jy*dimy]*delta*delta;
		}
	}
	return PROP_NO_ERR;
}

inline propError ift2(complex<double>* Uin_ptr, unsigned int dimx, unsigned int dimy, double delta)
{
	if (dimx!=dimy)
		return PROP_ERR;
	fftshift(Uin_ptr, dimx, dimy);

	fftw_complex *in=reinterpret_cast<fftw_complex*>(Uin_ptr);
	fftw_plan p;

	p = fftw_plan_dft_2d(dimx, dimy, in, in, FFTW_BACKWARD, FFTW_ESTIMATE);

	fftw_execute(p);

	fftshift(Uin_ptr, dimx, dimy);

	for (unsigned long jx=0; jx<dimx; jx++)
	{
		for (unsigned long jy=0; jy<dimy;jy++)
		{
			Uin_ptr[jx+jy*dimy]=dimx*dimy*delta*delta*Uin_ptr[jx+jy*dimy];
		}
	}
	return PROP_NO_ERR;
}

inline propError ift1(complex<double>* Uin_ptr, unsigned int dimx, double delta)
{
	fftshift(Uin_ptr, dimx, 1);

	fftw_complex *in=reinterpret_cast<fftw_complex*>(Uin_ptr);
	fftw_plan p;

	p = fftw_plan_dft_1d(dimx, in, in, FFTW_BACKWARD, FFTW_ESTIMATE);

	fftw_execute(p);

	fftshift(Uin_ptr, dimx, 1);

	for (unsigned long jx=0; jx<dimx; jx++)
	{
		Uin_ptr[jx]=dimx*delta*Uin_ptr[jx];
	}
	return PROP_NO_ERR;
}

// thread safe implementations...

inline propError ft1_ts(complex<double>* Uin_ptr, unsigned int dimx, double delta, fftw_plan &p)
{
	fftshift(Uin_ptr, dimx, 1);

	fftw_complex *in=reinterpret_cast<fftw_complex*>(Uin_ptr);
//	fftw_plan p;

//	p = fftw_plan_dft_1d(dimx, in, in, FFTW_FORWARD, FFTW_ESTIMATE);

	fftw_execute_dft(p, in, in);

	fftshift(Uin_ptr, dimx, 1);

	for (unsigned long jx=0; jx<dimx; jx++)
	{
		Uin_ptr[jx]=Uin_ptr[jx]*delta;
	}
	return PROP_NO_ERR;
}

inline propError ft2_ts(complex<double>* Uin_ptr, unsigned int dimx, unsigned int dimy, double delta, fftw_plan &p)
{
	if (dimx!=dimy)
		return PROP_ERR;
	fftshift(Uin_ptr, dimx, dimy);

	fftw_complex *in=reinterpret_cast<fftw_complex*>(Uin_ptr);
//	fftw_plan p;

//	p = fftw_plan_dft_2d(dimx, dimy, in, in, FFTW_FORWARD, FFTW_ESTIMATE);

	fftw_execute_dft(p, in, in);

	fftshift(Uin_ptr, dimx, dimy);

	for (unsigned long jx=0; jx<dimx; jx++)
	{
		for (unsigned long jy=0; jy<dimy;jy++)
		{
			Uin_ptr[jx+jy*dimy]=Uin_ptr[jx+jy*dimy]*delta*delta;
		}
	}
	return PROP_NO_ERR;
}

inline propError ift2_ts(complex<double>* Uin_ptr, unsigned int dimx, unsigned int dimy, double delta, fftw_plan &p)
{
	if (dimx!=dimy)
		return PROP_ERR;
	fftshift(Uin_ptr, dimx, dimy);

	fftw_complex *in=reinterpret_cast<fftw_complex*>(Uin_ptr);
//	fftw_plan p;

//	p = fftw_plan_dft_2d(dimx, dimy, in, in, FFTW_BACKWARD, FFTW_ESTIMATE);

	fftw_execute_dft(p, in, in);

	fftshift(Uin_ptr, dimx, dimy);

	for (unsigned long jx=0; jx<dimx; jx++)
	{
		for (unsigned long jy=0; jy<dimy;jy++)
		{
			Uin_ptr[jx+jy*dimy]=dimx*dimy*delta*delta*Uin_ptr[jx+jy*dimy];
		}
	}
	return PROP_NO_ERR;
}

inline propError ift1_ts(complex<double>* Uin_ptr, unsigned int dimx, double delta, fftw_plan &p)
{
	fftshift(Uin_ptr, dimx, 1);

	fftw_complex *in=reinterpret_cast<fftw_complex*>(Uin_ptr);
//	fftw_plan p;

//	p = fftw_plan_dft_1d(dimx, in, in, FFTW_BACKWARD, FFTW_ESTIMATE);

	fftw_execute_dft(p, in, in);

	fftshift(Uin_ptr, dimx, 1);

	for (unsigned long jx=0; jx<dimx; jx++)
	{
		Uin_ptr[jx]=dimx*delta*Uin_ptr[jx];
	}
	return PROP_NO_ERR;
}