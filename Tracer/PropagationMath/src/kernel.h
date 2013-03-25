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

#ifndef __kernel__
	#define __kernel__

#include "cuda.h"
#include "cuda_runtime.h"
#include <cufft.h>
//#include "cutil_inline.h"
#include <complex>
#include "cuComplex.h"

#define PI ((double)3.141592653589793238462643383279502884197169399375105820)

inline bool myCufftSafeCall( cufftResult err, const char *file, const int line )
{
    if( CUFFT_SUCCESS != err) {
        fprintf(stderr, "%s(%i) : CUFFT error.\n", file, line);
        return false;
    }
	return true;
};

inline bool myCudaSafeCall( cudaError err, const char *file, const int line )
{
    if( cudaSuccess != err) {
		fprintf(stderr, "%s(%i) : cudaSafeCall() Runtime API error : %s.\n",
                file, line, cudaGetErrorString( err) );
        return false;
    }
	return true;
}

using namespace std;

//extern "C" void kernel_wrapper(int *a, int *b);
//extern "C" bool cu_angularSpectrum_scaled(complex<double>* Uin_ptr, unsigned int dimx, unsigned int dimy, double wvl, double* x1_ptr, double* y1_ptr, double Dz, double dx2, double** x2_ptrptr, double** y2_ptrptr);
//extern "C" bool cu_angularSpectrum_ABCD(complex<double>* Uin_ptr, unsigned int dimx, unsigned int dimy, double wvl, double* x1_ptr, double* y1_ptr, double Dz, double dx2, double* ABCD, double** x2_ptrptr, double** y2_ptrptr);
//extern "C" bool cu_fresnel_two_step(complex<double>* Uin_ptr, unsigned int dimx, unsigned int dimy, double wvl, double* x1_ptr, double* y1_ptr, double dx2, double Dz, double** x2_ptrptr, double** y2_ptrptr);
//extern "C" bool cu_fresnel_one_step_1D(complex<double>* Uin_ptr, unsigned int dimx, double wvl, double* x1_ptr, double Dz, double** x2_ptrptr );
//extern "C" bool cu_fresnel_one_step(complex<double>* Uin_ptr, unsigned int dimx, unsigned int dimy, double wvl, double* x1_ptr, double* y1_ptr, double Dz, double** x2_ptrptr, double** y2_ptrptr);
//extern "C" bool cu_fraunhofer(complex<double>* Uin_ptr, unsigned int dimx, unsigned int dimy, double wvl, double* x1_ptr, double* y1_ptr, double Dz, double** x2_ptrptr, double** y2_ptrptr);
//extern "C" bool cu_fraunhofer_1D(complex<double>* Uin_ptr, unsigned int dimx, double wvl, double* x1_ptr, double Dz, double** x2_ptrptr);
//extern "C" bool cu_fftshift(complex<double>* in, unsigned int dimx, unsigned int dimy);
//extern "C" bool cu_calcWavefront(complex<double>* Uin_ptr, unsigned int dimx, unsigned int dimy, double* x_ptr, double *y_ptr, double D, double *coeffVec_ptr);

//extern "C" void scalar_RichardsonWolf_kernel(cuDoubleComplex* Uin_ptr, double* x1_ptr,  double* y1_ptr, double* x2_ptr, double*y2_ptr, unsigned int dimx, unsigned int dimy, unsigned int TileWidth, unsigned int TileHeight, double wvl, double f, double Dz);
//extern "C" bool scalar_RichardsonWolf_wrapper(complex<double>* Uin_ptr, unsigned int dimx, unsigned int dimy, double wvl, double* x1_ptr, double* y1_ptr, double f, double Dz, double** x2_ptrptr, double** y2_ptrptr);


#endif