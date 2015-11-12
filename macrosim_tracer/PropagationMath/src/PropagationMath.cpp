#include "vector_types.h"
#include "PropagationMath.h"
#include "kernel.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include <cufft.h>
#include <omp.h>
#include <complex>

#include <ctime>

//#include <iostream>
//using namespace std;

propError cu_scalarRichardsonWolf(cuDoubleComplex* Uin_ptr, unsigned int dimx, unsigned int dimy, double wvl, double* x1_ptr, double* y1_ptr, double f, double Dz, double** x2_ptrptr, double** y2_ptrptr)
{
	if (!cu_scalarRichardsonWolf_wrapper(Uin_ptr, dimx, dimy, wvl, x1_ptr, y1_ptr, f, Dz, x2_ptrptr, y2_ptrptr))
		return PROP_ERR;
	return PROP_NO_ERR;
}

propError cu_ft2(cuDoubleComplex* Uin_ptr, unsigned int dimx, unsigned int dimy)
{
 //   clock_t start, end, startGes, endGes;
//	double msecs_DataTransfer=0;
//    double msecs_fft=0;
//    double msecs_ges=0;

//    startGes=clock();


//    for (unsigned int idx=0; idx<1000;idx++)
//    {
//    start=clock();*/

	if (dimx!=dimy)
		return PROP_ERR;

    // Allocate device memory 
    cuDoubleComplex* Uin_kernel_ptr;
    (cudaMalloc((void**)&Uin_kernel_ptr, sizeof(cuDoubleComplex)*dimx*dimy));

    // Copy host memory to device
    (cudaMemcpy(Uin_kernel_ptr, Uin_ptr, sizeof(cuDoubleComplex)*dimx*dimy, cudaMemcpyHostToDevice));

	// do the fft
    // CUFFT plan
    cufftHandle plan;
    (cufftPlan2d(&plan,dimx, dimy, CUFFT_Z2Z));
    //cudaDeviceSynchronize();

    //end=clock();
    //msecs_DataTransfer+=((end-start)/(double)CLOCKS_PER_SEC*1000.0);

    //start=clock();
    // execution
    (cufftExecZ2Z(plan, (cufftDoubleComplex *)Uin_kernel_ptr, (cufftDoubleComplex *)Uin_kernel_ptr, CUFFT_FORWARD));
    //cudaDeviceSynchronize();
    //end=clock();

    //msecs_fft+=((end-start)/(double)CLOCKS_PER_SEC*1000.0);

    //start=clock();
	// copy device memory back to host
	(cudaMemcpy(Uin_ptr, Uin_kernel_ptr, sizeof(complex<double>)*dimx*dimy, cudaMemcpyDeviceToHost));
    //cudaDeviceSynchronize();


//	fftshift(Uin_ptr, dimx, dimy);

	//for (unsigned long jx=0; jx<dimx; jx++)
	//{
	//	for (unsigned long jy=0; jy<dimy;jy++)
	//	{
	//		Uin_ptr[jx+jy*dimy]=Uin_ptr[jx+jy*dimy]*delta*delta;
	//	}
	//}

	cudaFree(Uin_kernel_ptr);
    cufftDestroy (plan);
    //end=clock();
    //msecs_DataTransfer+=((end-start)/(double)CLOCKS_PER_SEC*1000.0);

    //}

    //endGes=clock();
    //msecs_ges=(endGes-startGes)/(double)CLOCKS_PER_SEC*1000.0;

    //double msecs_fft2=msecs_ges-msecs_DataTransfer;

    //cout << "fft time in ms: " << msecs_fft << "\n";
    //cout << "fft2 time in ms: " << msecs_fft2 << "\n";
    //cout << "transfer time in ms: " << msecs_DataTransfer << "\n";
    //cout << "resulting time in ms: " << msecs_ges << "\n";

	return PROP_NO_ERR;
};

/**
 * \detail angularSpectrum_scaled 
 *
 * computes scaled angular spectrum propagation. see Jason D. Schmidt, Numerical Simulation of Optical Wave Propagation for reference
 *
 * \param[in]	complex<double>*	Uin_ptr
 *				unsigned long		dimx
 *				unsigned long		dimy
 *				double				wvl
 *				double*				x1_ptr
 *				double*				y1_ptr
 *				double				Dz
 *				double				dx2
 *				double**			x2_ptrptr
 *				double**			y2_ptrptr
 * 
 * \return propErr
 * \sa 
 * \remarks 
 * \author Mauch
 */
/*propError angularSpectrum_scaled(complex<double>* Uin_ptr, unsigned int dimx, unsigned int dimy, double wvl, double* x1_ptr, double* y1_ptr, double Dz, double dx2, double** x2_ptrptr, double** y2_ptrptr)
{
	// we can only handle regularly squared grids here
	if (dimx!=dimy)
		return PROP_ERR;
	double k=2*M_PI/wvl;
	double dx1=abs(x1_ptr[0]-x1_ptr[1]);
	double dy1=abs(y1_ptr[0]-y1_ptr[1]);
	if (dx1!=dy1)
		return PROP_ERR;

	// spectral plane
	// coordinates
	double dfx=1/(dimx*dx1);
	double *fx_l=(double*)calloc(dimx,sizeof(double));
	double *fy_l=fx_l;
	for (unsigned long jx=0; jx<dimx; jx++)
	{
		fx_l[jx]=(-1.0*dimx/2+jx)/dfx;
	}
	// scaling parameter
	double m=dx2/dx1;
	// observation plane
	// coordinates
	double* x2_l=(double*)calloc(dimx, sizeof(double));
	double* y2_l=x2_l;
	for (unsigned long jx=0; jx<dimx; jx++)
	{
		x2_l[jx]=(-1.0*dimx/2+jx)*dx2;
	}
	// evaluate Fresnel integral
	for (unsigned long jx=0; jx<dimx;jx++)
	{
		for (unsigned long jy=0; jy<dimy;jy++)
		{
			Uin_ptr[jx+jy*dimy]=Uin_ptr[jx+jy*dimy]/m*polar(1.0,k/2*(1-m)/Dz*(x1_ptr[jx]*x1_ptr[jx]+y1_ptr[jy]*y1_ptr[jy]));
		}
	}
	ft2(Uin_ptr, dimx, dimy, dx1);
	for (unsigned long jx=0; jx<dimx;jx++)
	{
		for (unsigned long jy=0; jy<dimy; jy++)
		{
			Uin_ptr[jx+jy*dimy]=Uin_ptr[jx+jy*dimy]*polar(1.0,-1.0*M_PI*M_PI*Dz/m/k*(fx_l[jx]*fx_l[jx]+fy_l[jy]*fy_l[jy]));
		}
	}
	ift2(Uin_ptr, dimx, dimy, dfx);
	for (unsigned long jx=0; jx<dimx; jx++)
	{
		for (unsigned long jy=0; jy<dimy; jy++)
		{
			Uin_ptr[jx+jy*dimy]=Uin_ptr[jx+jy*dimy]*polar(1.0,k/2*(m-1)/(m*Dz)*(x2_l[jx]*x2_l[jx]+y2_l[jx]*y2_l[jx]));
		}
	}
	*x2_ptrptr=x2_l; // return coordinates of output field in respective pointer
	memcpy((*y2_ptrptr),x2_l,dimx*sizeof(double));
	return PROP_NO_ERR;
}
*/
/**
 * \detail angularSpectrum_ABCD 
 *
 * computes Collins Integral diffraction. see Jason D. Schmidt, Numerical Simulation of Optical Wave Propagation for reference
 *
 * \param[in]	complex<double>*	Uin_ptr
 *				unsigned long		dimx
 *				unsigned long		dimy
 *				double				wvl
 *				double*				x1_ptr
 *				double*				y1_ptr
 *				double				Dz
 *				double				dx2
 *				double*				ABCD
 *				double**			x2_ptrptr
 *				double**			y2_ptrptr
 * 
 * \return propErr
 * \sa 
 * \remarks 
 * \author Mauch
 */
/*propError angularSpectrum_ABCD(complex<double>* Uin_ptr, unsigned int dimx, unsigned int dimy, double wvl, double* x1_ptr, double* y1_ptr, double Dz, double dx2, double* ABCD, double** x2_ptrptr, double** y2_ptrptr)
{
	// we can only handle regularly squared grids here
	if (dimx!=dimy)
		return PROP_ERR;
	double k=2*M_PI/wvl;
	double dx1=abs(x1_ptr[0]-x1_ptr[1]);
	double dy1=abs(y1_ptr[0]-y1_ptr[1]);
	if (dx1!=dy1)
		return PROP_ERR;

	// spectral plane
	// coordinates
	double dfx=1/(dimx*dx1);
	double *fx_l=(double*)calloc(dimx,sizeof(double));
	double *fy_l=fx_l;
	for (unsigned long jx=0; jx<dimx; jx++)
	{
		fx_l[jx]=(-1.0*dimx/2+jx)/dfx;
	}
	// scaling parameter
	double m=dx2/dx1;
	// observation plane
	// coordinates
	double* x2_l=(double*)calloc(dimx, sizeof(double));
	double* y2_l=x2_l;
	for (unsigned long jx=0; jx<dimx; jx++)
	{
		x2_l[jx]=(-1.0*dimx/2+jx)*dx2;
	}
	// evaluate Fresnel integral
	for (unsigned long jx=0; jx<dimx;jx++)
	{
		for (unsigned long jy=0; jy<dimy;jy++)
		{
			Uin_ptr[jx+jy*dimy]=Uin_ptr[jx+jy*dimy]/m*polar(1.0,M_PI/(wvl*ABCD[1])*(ABCD[0]-m)*(x1_ptr[jx]*x1_ptr[jx]+y1_ptr[jy]*y1_ptr[jy]));
		}
	}
	ft2(Uin_ptr, dimx, dimy, dx1);
	for (unsigned long jx=0; jx<dimx;jx++)
	{
		for (unsigned long jy=0; jy<dimy; jy++)
		{
			Uin_ptr[jx+jy*dimy]=Uin_ptr[jx+jy*dimy]*polar(1.0,-1.0*M_PI*wvl*ABCD[1]/m*(fx_l[jx]*fx_l[jx]+fy_l[jy]*fy_l[jy]));
		}
	}
	ift2(Uin_ptr, dimx, dimy, dfx);
	for (unsigned long jx=0; jx<dimx; jx++)
	{
		for (unsigned long jy=0; jy<dimy; jy++)
		{
			Uin_ptr[jx+jy*dimy]=Uin_ptr[jx+jy*dimy]*polar(1.0,M_PI/(wvl*ABCD[1])*ABCD[0]*(ABCD[1]*ABCD[2]-ABCD[0]*(ABCD[0]-m)/m)*(x2_l[jx]*x2_l[jx]+y2_l[jx]*y2_l[jx]));
		}
	}
	*x2_ptrptr=x2_l; // return coordinates of output field in respective pointer
	memcpy((*y2_ptrptr),x2_l,dimx*sizeof(double));
	return PROP_NO_ERR;
}
*/
/**
 * \detail fresnel_two_step_1D 
 *
 * computes one dimensional fresnel propagation in two steps. see Jason D. Schmidt, Numerical Simulation of Optical Wave Propagation for reference
 *
 * \param[in]	complex<double>*	Uin_ptr
 *				unsigned long		dimx
 *				double				wvl
 *				double*				x1_ptr
 *				double				Dz
 *				double				dx2
 *				double**			x2_ptrptr
 * 
 * \return propErr
 * \sa 
 * \remarks 
 * \author Mauch
 */
/*propError fresnel_two_step_1D(complex<double>* Uin_ptr, unsigned int dimx, double wvl, double* x1_ptr, double dx2, double Dz, double** x2_ptrptr )
{
	double k=2*M_PI/wvl;
	double dx1=abs(x1_ptr[0]-x1_ptr[1]);
	// magnification
	double m=dx2/dx1;
	// intermediate plane
	double Dz1=Dz/(1-m); // propagation distance
	double dx1a=wvl*abs(Dz1)/(dimx*dx1); // coordinates
	double* x1a=(double*)malloc(dimx*sizeof(double));
	// evaluate Fresnel integral
	for (unsigned long jx=0;jx<dimx;jx++)
	{
		x1a[jx]=-1.0*dimx/2*dx1a+jx*dx1a;
		Uin_ptr[jx]=Uin_ptr[jx]*polar(1.0,k/(2*Dz1)*x1_ptr[jx]*x1_ptr[jx]);
	}
	ft1(Uin_ptr, dimx, dx1);
	complex<double> fac1;
	if (Dz1<0)
		fac1=complex<double>(sqrt(abs(0.5*wvl*Dz1)),-1.0*sqrt(abs(0.5*wvl*Dz1)));
	else
		fac1=complex<double>(sqrt(0.5*wvl*Dz1),sqrt(0.5*wvl*Dz1));
	for (unsigned long jx=0; jx<dimx;jx++)
	{
		Uin_ptr[jx]=1.0/fac1*Uin_ptr[jx]*polar(1.0,k/(2*Dz1)*x1a[jx]*x1a[jx]);
	}

	// observation plane
	double Dz2=Dz-Dz1; // propagation distance
	// coordinates
	double *x2_l=(double*)calloc(dimx,sizeof(double));
	// evaluate Fresnel integral
	for (unsigned long jx=0; jx<dimx;jx++)
	{
		x2_l[jx]=-1.0*dimx/2*dx2+jx*dx2;
		Uin_ptr[jx]=Uin_ptr[jx]*polar(1.0,k/(2*Dz2)*x1a[jx]*x1a[jx]);
	}
	ft1(Uin_ptr, dimx, dx1a);
	if (Dz2<0)
		fac1=complex<double>(sqrt(abs(0.5*wvl*Dz2)),-1.0*sqrt(abs(0.5*wvl*Dz2)));
	else
		fac1=complex<double>(sqrt(0.5*wvl*Dz2),sqrt(0.5*wvl*Dz2));
	for (unsigned long jx=0; jx<dimx;jx++)
	{
		Uin_ptr[jx]=1.0/fac1*Uin_ptr[jx]*polar(1.0,k/(2*Dz2)*x2_l[jx]*x2_l[jx]);
	}
	*x2_ptrptr=x2_l; // return coordinates of output field in respective pointer
	delete x1a;

	return PROP_NO_ERR;
}
*/
/**
 * \detail fresnel_two_step 
 *
 * computes two dimensional fresnel propagation in two steps. see Jason D. Schmidt, Numerical Simulation of Optical Wave Propagation for reference
 *
 * \param[in]	complex<double>*	Uin_ptr
 *				unsigned long		dimx
 *				unsigned long		dimy
 *				double				wvl
 *				double*				x1_ptr
 *				double*				y1_ptr
 *				double				Dz
 *				double				dx2
 *				double**			x2_ptrptr
 *				double**			y2_ptrptr
 * 
 * \return propErr
 * \sa 
 * \remarks 
 * \author Mauch
 */
/*propError fresnel_two_step(complex<double>* Uin_ptr, unsigned int dimx, unsigned int dimy, double wvl, double* x1_ptr, double* y1_ptr, double dx2, double Dz, double** x2_ptrptr, double** y2_ptrptr)
{
	// we can only handle squared grids here
	if (dimx!=dimy)
		return PROP_ERR;
	double k=2*M_PI/wvl;
	double dx1=abs(x1_ptr[0]-x1_ptr[1]);
	double dy1=abs(y1_ptr[0]-y1_ptr[1]);
	// we can only handle regularly spaced grids so far
	if (dx1!=dy1)
		return PROP_ERR;
	// magnification
	double m=dx2/dx1;
	// intermediate plane
	double Dz1=Dz/(1-m); // propagation distance
	double dx1a=wvl*abs(Dz1)/(dimx*dx1); // coordinates
	double* x1a=(double*)malloc(dimx*sizeof(double));
	double* y1a=x1a; // as we assume the grid to be regularly squared we just use the same point as for x here...
	for (unsigned long jx=0; jx<dimx; jx++)
	{
		x1a[jx]=-1.0*dimx/2*dx1a+jx*dx1a;
	}
	// evaluate Fresnel integral
	for (unsigned long jx=0;jx<dimx;jx++)
	{
		for (unsigned long jy=0;jy<dimy;jy++)
		{
			Uin_ptr[jx+jy*dimy]=Uin_ptr[jx+jy*dimy]*polar(1.0,k/(2*Dz1)*(x1_ptr[jx]*x1_ptr[jx]+y1_ptr[jy]*y1_ptr[jy]));
		}
	}
	ft2(Uin_ptr, dimx, dimy, dx1);
	for (unsigned long jx=0; jx<dimx;jx++)
	{
		for (unsigned long jy=0; jy<dimy;jy++)
		{
			Uin_ptr[jx+jy*dimy]=1.0/complex<double>(0,wvl*Dz1)*Uin_ptr[jx+jy*dimy]*polar(1.0,k/(2*Dz1)*(x1a[jx]*x1a[jx]+y1a[jy]*y1a[jy]));
		}
	}

	// observation plane
	double Dz2=Dz-Dz1; // propagation distance
	// coordinates
	double *x2_l=(double*)calloc(dimx,sizeof(double));
	double *y2_l=x2_l;
	// evaluate Fresnel integral
	for (unsigned long jx=0; jx<dimx;jx++)
	{
		x2_l[jx]=(-1.0*dimx/2+jx)*dx2;
	}
	for (unsigned long jx=0;jx<dimx;jx++)
	{
		for (unsigned long jy=0;jy<dimy;jy++)
		{
			Uin_ptr[jx+jy*dimy]=Uin_ptr[jx+jy*dimy]*polar(1.0,k/(2*Dz2)*(x1a[jx]*x1a[jx]+y1a[jy]*y1a[jy]));
		}
	}
	ft2(Uin_ptr, dimx, dimy, dx1a);
	for (unsigned long jx=0; jx<dimx;jx++)
	{
		for (unsigned long jy=0; jy<dimy; jy++)
		{
			Uin_ptr[jx+jy*dimy]=1.0/complex<double>(0,wvl*Dz2)*Uin_ptr[jx+jy*dimy]*polar(1.0,k/(2*Dz2)*(x2_l[jx]*x2_l[jx]+y2_l[jy]*y2_l[jy]));
		}
	}
	*x2_ptrptr=x2_l; // return coordinates of output field in respective pointer
	(*y2_ptrptr)=(double*)calloc(dimx,sizeof(double));
	memcpy((*y2_ptrptr), x2_l, dimx*sizeof(double));
	delete x1a;
	return PROP_NO_ERR;
}
*/
/**
 * \detail fresnel_one_step_1D 
 *
 * computes one dimensional fresnel propagation in one step. see Jason D. Schmidt, Numerical Simulation of Optical Wave Propagation for reference
 *
 * \param[in]	complex<double>*	Uin_ptr
 *				unsigned long		dimx
 *				double				wvl
 *				double*				x1_ptr
 *				double				Dz
 *				double**			x2_ptrptr
 * 
 * \return propErr
 * \sa 
 * \remarks 
 * \author Mauch
 */
/*propError fresnel_one_step_1D(complex<double>* Uin_ptr, unsigned int dimx, double wvl, double* x1_ptr, double Dz, double** x2_ptrptr )
{
	double k=2*M_PI/wvl;
	double dx1=abs(x1_ptr[0]-x1_ptr[1]);

	// observation plane
	// coordinates
	double *x2_l=(double*)calloc(dimx,sizeof(double));
	// evaluate Fresnel integral
	for (unsigned long jx=0; jx<dimx;jx++)
	{
		x2_l[jx]=(-1.0*dimx/2+jx)/(dimx*dx1)*wvl*Dz;
		Uin_ptr[jx]=Uin_ptr[jx]*polar(1.0,k/(2*Dz)*x1_ptr[jx]*x1_ptr[jx]);
	}
	ft1(Uin_ptr, dimx, dx1);
	complex<double> fac1;
	if (Dz<0)
		fac1=complex<double>(sqrt(abs(0.5*wvl*Dz)),-1.0*sqrt(abs(0.5*wvl*Dz)));
	else
		fac1=complex<double>(sqrt(0.5*wvl*Dz),sqrt(0.5*wvl*Dz));
	for (unsigned long jx=0; jx<dimx;jx++)
	{
		Uin_ptr[jx]=1.0/fac1*Uin_ptr[jx]*polar(1.0,k/(2*Dz)*x2_l[jx]*x2_l[jx]);
	}
	*x2_ptrptr=x2_l; // return coordinates of output field in respective pointer

	return PROP_NO_ERR;
}
*/
/**
 * \detail fresnel_one_step 
 *
 * computes two dimensional fresnel propagation in one step. see Jason D. Schmidt, Numerical Simulation of Optical Wave Propagation for reference
 *
 * \param[in]	complex<double>*	Uin_ptr
 *				unsigned long		dimx
 *				unsigned long		dimy
 *				double				wvl
 *				double*				x1_ptr
 *				double*				y1_ptr
 *				double				Dz
 *				double**			x2_ptrptr
 *				double**			y2_ptrptr
 * 
 * \return propErr
 * \sa 
 * \remarks 
 * \author Mauch
 */
/*propError fresnel_one_step(complex<double>* Uin_ptr, unsigned int dimx, unsigned int dimy, double wvl, double* x1_ptr, double* y1_ptr, double Dz, double** x2_ptrptr, double** y2_ptrptr)
{
	// we can only handle regularly squared grids here
	if (dimx!=dimy)
		return PROP_ERR;
	double k=2*M_PI/wvl;
	double dx1=abs(x1_ptr[0]-x1_ptr[1]);
	double dy1=abs(y1_ptr[0]-y1_ptr[1]);
	if (dx1!=dy1)
		return PROP_ERR;

	// observation plane
	// coordinates
	double *x2_l=(double*)calloc(dimx,sizeof(double));
	double *y2_l=x2_l;
	for (unsigned long jx=0; jx<dimx; jx++)
	{
		x2_l[jx]=(-1.0*dimx/2+jx)/(dimx*dx1)*wvl*Dz;
	}
	// evaluate Fresnel integral
	for (unsigned long jx=0; jx<dimx;jx++)
	{
		for (unsigned long jy=0; jy<dimy;jy++)
		{
			Uin_ptr[jx+jy*dimy]=Uin_ptr[jx+jy*dimy]*polar(1.0,k/(2*Dz)*(x1_ptr[jx]*x1_ptr[jx]+y1_ptr[jy]*y1_ptr[jy]));
		}
	}
	ft2(Uin_ptr, dimx, dimy, dx1);
	for (unsigned long jx=0; jx<dimx;jx++)
	{
		for (unsigned long jy=0; jy<dimy; jy++)
		{
			Uin_ptr[jx+jy*dimy]=1.0/complex<double>(0,wvl*Dz)*Uin_ptr[jx+jy*dimy]*polar(1.0,k/(2*Dz)*(x2_l[jx]*x2_l[jx]+y2_l[jy]*y2_l[jy]));
		}
	}
	*x2_ptrptr=x2_l; // return coordinates of output field in respective pointer
	(*y2_ptrptr)=(double*)calloc(dimx,sizeof(double));
	memcpy((*y2_ptrptr),x2_l,dimx*sizeof(double));
	return PROP_NO_ERR;
}
*/

/**
 * \detail scalar_RichardsonWolf 
 *
 * computes scalar propagation to the focus of a lense. see Masud Mansuripur, Classical Optics and its Applications for reference
 *
 * \param[in]	complex<double>*	Uin_ptr
 *				unsigned long		dimx
 *				unsigned long		dimy
 *				double				wvl
 *				double*				x1_ptr
 *				double*				y1_ptr
 *				double				f
 *				double				Dz
 *				double**			x2_ptrptr
 *				double**			y2_ptrptr
 * 
 * \return propErr
 * \sa 
 * \remarks 
 * \author Mauch
 */
/*propError scalar_RichardsonWolf(complex<double>* Uin_ptr, unsigned int dimx, unsigned int dimy, double wvl, double* x1_ptr, double* y1_ptr, double f, double Dz, double** x2_ptrptr, double** y2_ptrptr)
{
	// we can only handle regularly squared grids here
	if (dimx!=dimy)
		return PROP_ERR;
	double k=2*M_PI/wvl;
	double dx1=abs(x1_ptr[0]-x1_ptr[1]);
	double dy1=abs(y1_ptr[0]-y1_ptr[1]);
	if (dx1!=dy1)
		return PROP_ERR;

	// observation plane
	// coordinates
	double *x2_l=(double*)calloc(dimx,sizeof(double));
	double *y2_l=x2_l;
	for (unsigned long jx=0; jx<dimx; jx++)
	{
		x2_l[jx]=(-1.0*dimx/2+jx)/(dimx*dx1)*wvl*f;
	}

	// evaluate scalar Richardson Wolf integral
	for (unsigned long jx=0; jx<dimx;jx++)
	{
		double sigmaX=-x1_ptr[jx]/f;
		for (unsigned long jy=0; jy<dimy;jy++)
		{
			double sigmaY=-y1_ptr[jy]/f;
			double GktSqr=1-sigmaX*sigmaX-sigmaY*sigmaY;
			// cut off evanescent waves
			if (GktSqr<0)
			{
				GktSqr=0.0;
				Uin_ptr[jx+jy*dimy]=0;
			}
			else
				Uin_ptr[jx+jy*dimy]=complex<double>(0.0,-1.0)*f*Uin_ptr[jx+jy*dimy]/pow(polar(1-sigmaX*sigmaX-sigmaY*sigmaY,0.0),0.25)*polar(1.0,k*Dz*sqrt(GktSqr));
		}
	}
	ft2(Uin_ptr, dimx, dimy, dx1/f);

	*x2_ptrptr=x2_l; // return coordinates of output field in respective pointer
	(*y2_ptrptr)=(double*)calloc(dimx,sizeof(double));
	memcpy((*y2_ptrptr),x2_l,dimx*sizeof(double));
	return PROP_NO_ERR;
}
*/
/**
 * \detail fraunhofer 
 *
 * computes fraunhofer propagation. see Jason D. Schmidt, Numerical Simulation of Optical Wave Propagation for reference
 *
 * \param[in]	complex<double>*	Uin_ptr
 *				unsigned long		dimx
 *				unsigned long		dimy
 *				double				wvl
 *				double*				x1_ptr
 *				double*				y1_ptr
 *				double				Dz
 *				double**			x2_ptrptr
 *				double**			y2_ptrptr
 * 
 * \return propErr
 * \sa 
 * \remarks 
 * \author Mauch
 */
/*propError fraunhofer(complex<double>* Uin_ptr, unsigned int dimx, unsigned int dimy, double wvl, double* x1_ptr, double* y1_ptr, double Dz, double** x2_ptrptr, double** y2_ptrptr)
{
	// we can only handle regularly squared grids here
	if (dimx!=dimy)
		return PROP_ERR;
	double k=2*M_PI/wvl;
	double dx1=abs(x1_ptr[0]-x1_ptr[1]);
	double dy1=abs(y1_ptr[0]-y1_ptr[1]);
	if (dx1!=dy1)
		return PROP_ERR;

	// observation plane
	// coordinates
	double *x2_l=(double*)calloc(dimx,sizeof(double));
	double *y2_l=x2_l;
	for (unsigned long jx=0; jx<dimx; jx++)
	{
		x2_l[jx]=(-1.0*dimx/2+jx)/(dimx*dx1)*wvl*Dz;
	}
	
	ft2(Uin_ptr, dimx, dimy, dx1);
	for (unsigned long jx=0; jx<dimx;jx++)
	{
		for (unsigned long jy=0; jy<dimy; jy++)
		{
			Uin_ptr[jx+jy*dimy]=Uin_ptr[jx+jy*dimy]/complex<double>(0,wvl*Dz)*polar(1.0,k/(2*Dz)*(x2_l[jx]*x2_l[jx]+y2_l[jy]*y2_l[jy]));
		}
	}
	*x2_ptrptr=x2_l; // return coordinates of output field in respective pointer
	(*y2_ptrptr)=(double*)calloc(dimx,sizeof(double));
	memcpy((*y2_ptrptr),x2_l,dimx*sizeof(double));
	return PROP_NO_ERR;
}
*/
/**
 * \detail fraunhofer 
 *
 * computes 1D fraunhofer propagation. see Jason D. Schmidt, Numerical Simulation of Optical Wave Propagation for reference
 *
 * \param[in]	complex<double>*	Uin_ptr
 *				unsigned long		dimx
 *				double				wvl
 *				double*				x1_ptr
 *				double				Dz
 *				double**			x2_ptrptr
 * 
 * \return propErr
 * \sa 
 * \remarks 
 * \author Mauch
 */
/*propError fraunhofer_1D(complex<double>* Uin_ptr, unsigned int dimx, double wvl, double* x1_ptr, double Dz, double** x2_ptrptr)
{
	double k=2*M_PI/wvl;
	double dx1=abs(x1_ptr[0]-x1_ptr[1]);

	// observation plane
	// coordinates
	double *x2_l=(double*)calloc(dimx,sizeof(double));
	for (unsigned long jx=0; jx<dimx; jx++)
	{
		x2_l[jx]=(-1.0*dimx/2+jx)/(dimx*dx1)*wvl*Dz;
	}
	
	ft1(Uin_ptr, dimx, dx1);
	for (unsigned long jx=0; jx<dimx;jx++)
	{
		Uin_ptr[jx]=Uin_ptr[jx]/complex<double>(0,wvl*Dz)*polar(1.0,k/(2*Dz)*x2_l[jx]*x2_l[jx]);
	}
	*x2_ptrptr=x2_l; // return coordinates of output field in respective pointer
	return PROP_NO_ERR;
};
*/
propError calcWavefront(complex<double>* Uin_ptr, unsigned int dimx, unsigned int dimy, double* x_ptr, double *y_ptr, double D, double wvl, double *coeffVec_ptr)
{
//	int nrThreads=omp_get_max_threads();
//	nrThreads=1;
//
//	omp_set_num_threads(nrThreads);
//
//	std::cout << "calculating on " << nrThreads << " cores of CPU." << "...\n";
//
//#pragma omp parallel default(shared)
//{
//	#pragma omp for schedule(dynamic, 10)
//
//	for (signed long long j=0; j<dimx*dimy; j++)
//	{
//		// calc indices of field. The loop is organized in x first
//		signed long jx=j % dimx;
//		signed long jy=(j-jx)/dimx;

	for (signed int jy=0; jy<dimy; jy++)
	{
		for (signed int jx=0; jx<dimx; jx++)
		{
			double rho=sqrt(x_ptr[jx]*x_ptr[jx]+y_ptr[jy]*y_ptr[jy])/(D/2);
			double phi=atan2(x_ptr[jx],y_ptr[jy]);
			// add the contributions of all aberrations
			double phase = 0;
			phase=coeffVec_ptr[0];
			phase=phase+coeffVec_ptr[1]*(rho*cos(phi));
			phase=phase+coeffVec_ptr[2]*(rho*sin(phi));
			phase=phase+coeffVec_ptr[3]*(2*rho*rho-1);
			phase=phase+coeffVec_ptr[4]*(rho*rho*cos(2*phi));
			phase=phase+coeffVec_ptr[5]*(rho*rho*sin(2*phi));
			phase=phase+coeffVec_ptr[6]*((3*pow(rho,3)-2*rho)*cos(phi));
			phase=phase+coeffVec_ptr[7]*((3*pow(rho,3)-2*rho)*sin(phi));
			phase=phase+coeffVec_ptr[8]*(6*pow(rho,4)-6*pow(rho,2)+1);
			phase=phase+coeffVec_ptr[9]*(pow(rho,3)*cos(phi));
			phase=phase+coeffVec_ptr[10]*(pow(rho,3)*sin(phi));
			phase=phase+coeffVec_ptr[11]*((4*pow(rho,4)-3*pow(rho,2))*cos(phi));
			phase=phase+coeffVec_ptr[12]*((4*pow(rho,4)-3*pow(rho,2))*sin(phi));
			phase=phase+coeffVec_ptr[13]*((10*pow(rho,5)-12*pow(rho,3)+3*rho)*cos(phi));
			phase=phase+coeffVec_ptr[14]*((10*pow(rho,5)-12*pow(rho,3)+3*rho)*sin(phi));
			phase=phase+coeffVec_ptr[15]*(20*pow(rho,6)-30*pow(rho,4)+12*pow(rho,2)-1);

			if ( abs(rho)<=1 )
			{
				//double k=2*M_PI/wvl;
				//cout << k*phase << endl;
				
				Uin_ptr[jx+jy*dimx]=Uin_ptr[jx+jy*dimx]*polar(1.0,-2*M_PI/wvl*phase);
				
				//Uin_ptr[jx+jy*dimx]=Uin_ptr[jx+jy*dimx]*polar(1.0,-phase);
				//Uin_ptr[1048575]=polar(1.0,0.0);
			}
			else
				Uin_ptr[jx+jy*dimx]=polar(1.0,0.0);
		}
	}

	//}
//}

	return PROP_NO_ERR;
}

// thread safe implementations
/**
 * \detail angularSpectrum_scaled_ts 
 *
 * computes scaled angular spectrum propagation thread safe. see Jason D. Schmidt, Numerical Simulation of Optical Wave Propagation for reference
 *
 * \param[in]	complex<double>*	Uin_ptr
 *				unsigned long		dimx
 *				unsigned long		dimy
 *				double				wvl
 *				double*				x1_ptr
 *				double*				y1_ptr
 *				double				Dz
 *				double				dx2
 *				double**			x2_ptrptr
 *				double**			y2_ptrptr
 *				fftw_plan			&p_fw
 *				fftw_plan			&p_bw
 * 
 * \return propErr
 * \sa 
 * \remarks 
 * \author Mauch
 */
/*propError angularSpectrum_scaled_ts(complex<double>* Uin_ptr, unsigned int dimx, unsigned int dimy, double wvl, double* x1_ptr, double* y1_ptr, double Dz, double dx2, double** x2_ptrptr, double** y2_ptrptr, fftw_plan &p_fw, fftw_plan &p_bw)
{
	// we can only handle regularly squared grids here
	if (dimx!=dimy)
		return PROP_ERR;
	double k=2*M_PI/wvl;
	double dx1=abs(x1_ptr[0]-x1_ptr[1]);
	double dy1=abs(y1_ptr[0]-y1_ptr[1]);
	if (dx1!=dy1)
		return PROP_ERR;

	// spectral plane
	// coordinates
	double dfx=1/(dimx*dx1);
	double *fx_l=(double*)calloc(dimx,sizeof(double));
	double *fy_l=fx_l;
	for (unsigned long jx=0; jx<dimx; jx++)
	{
		fx_l[jx]=(-1.0*dimx/2+jx)/dfx;
	}
	// scaling parameter
	double m=dx2/dx1;
	// observation plane
	// coordinates
	double* x2_l=(double*)calloc(dimx, sizeof(double));
	double* y2_l=x2_l;
	for (unsigned long jx=0; jx<dimx; jx++)
	{
		x2_l[jx]=(-1.0*dimx/2+jx)*dx2;
	}
	// evaluate Fresnel integral
	for (unsigned long jx=0; jx<dimx;jx++)
	{
		for (unsigned long jy=0; jy<dimy;jy++)
		{
			Uin_ptr[jx+jy*dimy]=Uin_ptr[jx+jy*dimy]/m*polar(1.0,k/2*(1-m)/Dz*(x1_ptr[jx]*x1_ptr[jx]+y1_ptr[jy]*y1_ptr[jy]));
		}
	}
	ft2_ts(Uin_ptr, dimx, dimy, dx1, p_fw);
	for (unsigned long jx=0; jx<dimx;jx++)
	{
		for (unsigned long jy=0; jy<dimy; jy++)
		{
			Uin_ptr[jx+jy*dimy]=Uin_ptr[jx+jy*dimy]*polar(1.0,-1.0*M_PI*M_PI*Dz/m/k*(fx_l[jx]*fx_l[jx]+fy_l[jy]*fy_l[jy]));
		}
	}
	ift2_ts(Uin_ptr, dimx, dimy, dfx, p_bw);
	for (unsigned long jx=0; jx<dimx; jx++)
	{
		for (unsigned long jy=0; jy<dimy; jy++)
		{
			Uin_ptr[jx+jy*dimy]=Uin_ptr[jx+jy*dimy]*polar(1.0,k/2*(m-1)/(m*Dz)*(x2_l[jx]*x2_l[jx]+y2_l[jx]*y2_l[jx]));
		}
	}
	*x2_ptrptr=x2_l; // return coordinates of output field in respective pointer
	memcpy((*y2_ptrptr),x2_l,dimx*sizeof(double));
	return PROP_NO_ERR;
}
*/
/**
 * \detail angularSpectrum_ABCD_ts 
 *
 * computes Collins Integral diffraction thread safe. see Jason D. Schmidt, Numerical Simulation of Optical Wave Propagation for reference
 *
 * \param[in]	complex<double>*	Uin_ptr
 *				unsigned long		dimx
 *				unsigned long		dimy
 *				double				wvl
 *				double*				x1_ptr
 *				double*				y1_ptr
 *				double				Dz
 *				double				dx2
 *				double*				ABCD
 *				double**			x2_ptrptr
 *				double**			y2_ptrptr
 *				fftw_plan			&p_fw
 *				fftw_plan			&p_bw
 * 
 * \return propErr
 * \sa 
 * \remarks 
 * \author Mauch
 */
/*propError angularSpectrum_ABCD_ts(complex<double>* Uin_ptr, unsigned int dimx, unsigned int dimy, double wvl, double* x1_ptr, double* y1_ptr, double Dz, double dx2, double* ABCD, double** x2_ptrptr, double** y2_ptrptr, fftw_plan &p_fw, fftw_plan &p_bw)
{
	// we can only handle regularly squared grids here
	if (dimx!=dimy)
		return PROP_ERR;
	double k=2*M_PI/wvl;
	double dx1=abs(x1_ptr[0]-x1_ptr[1]);
	double dy1=abs(y1_ptr[0]-y1_ptr[1]);
	if (dx1!=dy1)
		return PROP_ERR;

	// spectral plane
	// coordinates
	double dfx=1/(dimx*dx1);
	double *fx_l=(double*)calloc(dimx,sizeof(double));
	double *fy_l=fx_l;
	for (unsigned long jx=0; jx<dimx; jx++)
	{
		fx_l[jx]=(-1.0*dimx/2+jx)/dfx;
	}
	// scaling parameter
	double m=dx2/dx1;
	// observation plane
	// coordinates
	double* x2_l=(double*)calloc(dimx, sizeof(double));
	double* y2_l=x2_l;
	for (unsigned long jx=0; jx<dimx; jx++)
	{
		x2_l[jx]=(-1.0*dimx/2+jx)*dx2;
	}
	// evaluate Fresnel integral
	for (unsigned long jx=0; jx<dimx;jx++)
	{
		for (unsigned long jy=0; jy<dimy;jy++)
		{
			Uin_ptr[jx+jy*dimy]=Uin_ptr[jx+jy*dimy]/m*polar(1.0,M_PI/(wvl*ABCD[1])*(ABCD[0]-m)*(x1_ptr[jx]*x1_ptr[jx]+y1_ptr[jy]*y1_ptr[jy]));
		}
	}
	ft2_ts(Uin_ptr, dimx, dimy, dx1, p_fw);
	for (unsigned long jx=0; jx<dimx;jx++)
	{
		for (unsigned long jy=0; jy<dimy; jy++)
		{
			Uin_ptr[jx+jy*dimy]=Uin_ptr[jx+jy*dimy]*polar(1.0,-1.0*M_PI*wvl*ABCD[1]/m*(fx_l[jx]*fx_l[jx]+fy_l[jy]*fy_l[jy]));
		}
	}
	ift2_ts(Uin_ptr, dimx, dimy, dfx, p_bw);
	for (unsigned long jx=0; jx<dimx; jx++)
	{
		for (unsigned long jy=0; jy<dimy; jy++)
		{
			Uin_ptr[jx+jy*dimy]=Uin_ptr[jx+jy*dimy]*polar(1.0,M_PI/(wvl*ABCD[1])*ABCD[0]*(ABCD[1]*ABCD[2]-ABCD[0]*(ABCD[0]-m)/m)*(x2_l[jx]*x2_l[jx]+y2_l[jx]*y2_l[jx]));
		}
	}
	*x2_ptrptr=x2_l; // return coordinates of output field in respective pointer
	memcpy((*y2_ptrptr),x2_l,dimx*sizeof(double));
	return PROP_NO_ERR;
}
*/
/**
 * \detail fresnel_two_step_1D_ts 
 *
 * computes one dimensional fresnel propagation in two steps thread safe. see Jason D. Schmidt, Numerical Simulation of Optical Wave Propagation for reference
 *
 * \param[in]	complex<double>*	Uin_ptr
 *				unsigned long		dimx
 *				double				wvl
 *				double*				x1_ptr
 *				double				Dz
 *				double				dx2
 *				double**			x2_ptrptr
 *				fftw_plan			&p
 * 
 * \return propErr
 * \sa 
 * \remarks 
 * \author Mauch
 */
/*propError fresnel_two_step_1D(complex<double>* Uin_ptr, unsigned int dimx, double wvl, double* x1_ptr, double dx2, double Dz, double** x2_ptrptr , fftw_plan &p)
{
	double k=2*M_PI/wvl;
	double dx1=abs(x1_ptr[0]-x1_ptr[1]);
	// magnification
	double m=dx2/dx1;
	// intermediate plane
	double Dz1=Dz/(1-m); // propagation distance
	double dx1a=wvl*abs(Dz1)/(dimx*dx1); // coordinates
	double* x1a=(double*)malloc(dimx*sizeof(double));
	// evaluate Fresnel integral
	for (unsigned long jx=0;jx<dimx;jx++)
	{
		x1a[jx]=-1.0*dimx/2*dx1a+jx*dx1a;
		Uin_ptr[jx]=Uin_ptr[jx]*polar(1.0,k/(2*Dz1)*x1_ptr[jx]*x1_ptr[jx]);
	}
	ft1_ts(Uin_ptr, dimx, dx1, p);
	complex<double> fac1;
	if (Dz1<0)
		fac1=complex<double>(sqrt(abs(0.5*wvl*Dz1)),-1.0*sqrt(abs(0.5*wvl*Dz1)));
	else
		fac1=complex<double>(sqrt(0.5*wvl*Dz1),sqrt(0.5*wvl*Dz1));
	for (unsigned long jx=0; jx<dimx;jx++)
	{
		Uin_ptr[jx]=1.0/fac1*Uin_ptr[jx]*polar(1.0,k/(2*Dz1)*x1a[jx]*x1a[jx]);
	}

	// observation plane
	double Dz2=Dz-Dz1; // propagation distance
	// coordinates
	double *x2_l=(double*)calloc(dimx,sizeof(double));
	// evaluate Fresnel integral
	for (unsigned long jx=0; jx<dimx;jx++)
	{
		x2_l[jx]=-1.0*dimx/2*dx2+jx*dx2;
		Uin_ptr[jx]=Uin_ptr[jx]*polar(1.0,k/(2*Dz2)*x1a[jx]*x1a[jx]);
	}
	ft1_ts(Uin_ptr, dimx, dx1a, p);
	if (Dz2<0)
		fac1=complex<double>(sqrt(abs(0.5*wvl*Dz2)),-1.0*sqrt(abs(0.5*wvl*Dz2)));
	else
		fac1=complex<double>(sqrt(0.5*wvl*Dz2),sqrt(0.5*wvl*Dz2));
	for (unsigned long jx=0; jx<dimx;jx++)
	{
		Uin_ptr[jx]=1.0/fac1*Uin_ptr[jx]*polar(1.0,k/(2*Dz2)*x2_l[jx]*x2_l[jx]);
	}
	*x2_ptrptr=x2_l; // return coordinates of output field in respective pointer
	delete x1a;

	return PROP_NO_ERR;
}
*/
/**
 * \detail fresnel_two_step_ts 
 *
 * computes two dimensional fresnel propagation in two steps thread safe. see Jason D. Schmidt, Numerical Simulation of Optical Wave Propagation for reference
 *
 * \param[in]	complex<double>*	Uin_ptr
 *				unsigned long		dimx
 *				unsigned long		dimy
 *				double				wvl
 *				double*				x1_ptr
 *				double*				y1_ptr
 *				double				Dz
 *				double				dx2
 *				double**			x2_ptrptr
 *				double**			y2_ptrptr
 *				fftw_plan			&p
 * 
 * \return propErr
 * \sa 
 * \remarks 
 * \author Mauch
 */
/*propError fresnel_two_step_ts(complex<double>* Uin_ptr, unsigned int dimx, unsigned int dimy, double wvl, double* x1_ptr, double* y1_ptr, double dx2, double Dz, double** x2_ptrptr, double** y2_ptrptr, fftw_plan &p)
{
	// we can only handle squared grids here
	if (dimx!=dimy)
		return PROP_ERR;
	double k=2*M_PI/wvl;
	double dx1=abs(x1_ptr[0]-x1_ptr[1]);
	double dy1=abs(y1_ptr[0]-y1_ptr[1]);
	// we can only handle regularly spaced grids so far
	if (dx1!=dy1)
		return PROP_ERR;
	// magnification
	double m=dx2/dx1;
	// intermediate plane
	double Dz1=Dz/(1-m); // propagation distance
	double dx1a=wvl*abs(Dz1)/(dimx*dx1); // coordinates
	double* x1a=(double*)malloc(dimx*sizeof(double));
	double* y1a=x1a; // as we assume the grid to be regularly squared we just use the same point as for x here...
	for (unsigned long jx=0; jx<dimx; jx++)
	{
		x1a[jx]=-1.0*dimx/2*dx1a+jx*dx1a;
	}
	// evaluate Fresnel integral
	for (unsigned long jx=0;jx<dimx;jx++)
	{
		for (unsigned long jy=0;jy<dimy;jy++)
		{
			Uin_ptr[jx+jy*dimy]=Uin_ptr[jx+jy*dimy]*polar(1.0,k/(2*Dz1)*(x1_ptr[jx]*x1_ptr[jx]+y1_ptr[jy]*y1_ptr[jy]));
		}
	}
	ft2_ts(Uin_ptr, dimx, dimy, dx1, p);
	for (unsigned long jx=0; jx<dimx;jx++)
	{
		for (unsigned long jy=0; jy<dimy;jy++)
		{
			Uin_ptr[jx+jy*dimy]=1.0/complex<double>(0,wvl*Dz1)*Uin_ptr[jx+jy*dimy]*polar(1.0,k/(2*Dz1)*(x1a[jx]*x1a[jx]+y1a[jy]*y1a[jy]));
		}
	}

	// observation plane
	double Dz2=Dz-Dz1; // propagation distance
	// coordinates
	double *x2_l=(double*)calloc(dimx,sizeof(double));
	double *y2_l=x2_l;
	// evaluate Fresnel integral
	for (unsigned long jx=0; jx<dimx;jx++)
	{
		x2_l[jx]=(-1.0*dimx/2+jx)*dx2;
	}
	for (unsigned long jx=0;jx<dimx;jx++)
	{
		for (unsigned long jy=0;jy<dimy;jy++)
		{
			Uin_ptr[jx+jy*dimy]=Uin_ptr[jx+jy*dimy]*polar(1.0,k/(2*Dz2)*(x1a[jx]*x1a[jx]+y1a[jy]*y1a[jy]));
		}
	}
	ft2_ts(Uin_ptr, dimx, dimy, dx1a, p);
	for (unsigned long jx=0; jx<dimx;jx++)
	{
		for (unsigned long jy=0; jy<dimy; jy++)
		{
			Uin_ptr[jx+jy*dimy]=1.0/complex<double>(0,wvl*Dz2)*Uin_ptr[jx+jy*dimy]*polar(1.0,k/(2*Dz2)*(x2_l[jx]*x2_l[jx]+y2_l[jy]*y2_l[jy]));
		}
	}
	*x2_ptrptr=x2_l; // return coordinates of output field in respective pointer
	(*y2_ptrptr)=(double*)calloc(dimx,sizeof(double));
	memcpy((*y2_ptrptr), x2_l, dimx*sizeof(double));
	delete x1a;
	return PROP_NO_ERR;
}
*/
/**
 * \detail fresnel_one_step_1D_ts 
 *
 * computes one dimensional fresnel propagation in one step thread safe. see Jason D. Schmidt, Numerical Simulation of Optical Wave Propagation for reference
 *
 * \param[in]	complex<double>*	Uin_ptr
 *				unsigned long		dimx
 *				double				wvl
 *				double*				x1_ptr
 *				double				Dz
 *				double**			x2_ptrptr
 *				fftw_plan			&p
 * 
 * \return propErr
 * \sa 
 * \remarks 
 * \author Mauch
 */
/*propError fresnel_one_step_1D_ts(complex<double>* Uin_ptr, unsigned int dimx, double wvl, double* x1_ptr, double Dz, double** x2_ptrptr , fftw_plan &p)
{
	double k=2*M_PI/wvl;
	double dx1=abs(x1_ptr[0]-x1_ptr[1]);

	// observation plane
	// coordinates
	double *x2_l=(double*)calloc(dimx,sizeof(double));
	// evaluate Fresnel integral
	for (unsigned long jx=0; jx<dimx;jx++)
	{
		x2_l[jx]=(-1.0*dimx/2+jx)/(dimx*dx1)*wvl*Dz;
		Uin_ptr[jx]=Uin_ptr[jx]*polar(1.0,k/(2*Dz)*x1_ptr[jx]*x1_ptr[jx]);
	}
	ft1_ts(Uin_ptr, dimx, dx1, p);
	complex<double> fac1;
	if (Dz<0)
		fac1=complex<double>(sqrt(abs(0.5*wvl*Dz)),-1.0*sqrt(abs(0.5*wvl*Dz)));
	else
		fac1=complex<double>(sqrt(0.5*wvl*Dz),sqrt(0.5*wvl*Dz));
	for (unsigned long jx=0; jx<dimx;jx++)
	{
		Uin_ptr[jx]=1.0/fac1*Uin_ptr[jx]*polar(1.0,k/(2*Dz)*x2_l[jx]*x2_l[jx]);
	}
	*x2_ptrptr=x2_l; // return coordinates of output field in respective pointer

	return PROP_NO_ERR;
}
*/
/**
 * \detail fresnel_one_step_ts 
 *
 * computes two dimensional fresnel propagation in one step thread safe. see Jason D. Schmidt, Numerical Simulation of Optical Wave Propagation for reference
 *
 * \param[in]	complex<double>*	Uin_ptr
 *				unsigned long		dimx
 *				unsigned long		dimy
 *				double				wvl
 *				double*				x1_ptr
 *				double*				y1_ptr
 *				double				Dz
 *				double**			x2_ptrptr
 *				double**			y2_ptrptr
 *				fftw_plan			&p
 * 
 * \return propErr
 * \sa 
 * \remarks 
 * \author Mauch
 */
/*propError fresnel_one_step_ts(complex<double>* Uin_ptr, unsigned int dimx, unsigned int dimy, double wvl, double* x1_ptr, double* y1_ptr, double Dz, double** x2_ptrptr, double** y2_ptrptr, fftw_plan &p)
{
	// we can only handle regularly squared grids here
	if (dimx!=dimy)
		return PROP_ERR;
	double k=2*M_PI/wvl;
	double dx1=abs(x1_ptr[0]-x1_ptr[1]);
	double dy1=abs(y1_ptr[0]-y1_ptr[1]);
	if (dx1!=dy1)
		return PROP_ERR;

	// observation plane
	// coordinates
	double *x2_l=(double*)calloc(dimx,sizeof(double));
	double *y2_l=x2_l;
	for (unsigned long jx=0; jx<dimx; jx++)
	{
		x2_l[jx]=(-1.0*dimx/2+jx)/(dimx*dx1)*wvl*Dz;
	}
	// evaluate Fresnel integral
	for (unsigned long jx=0; jx<dimx;jx++)
	{
		for (unsigned long jy=0; jy<dimy;jy++)
		{
			Uin_ptr[jx+jy*dimy]=Uin_ptr[jx+jy*dimy]*polar(1.0,k/(2*Dz)*(x1_ptr[jx]*x1_ptr[jx]+y1_ptr[jy]*y1_ptr[jy]));
		}
	}
	ft2_ts(Uin_ptr, dimx, dimy, dx1, p);
	for (unsigned long jx=0; jx<dimx;jx++)
	{
		for (unsigned long jy=0; jy<dimy; jy++)
		{
			Uin_ptr[jx+jy*dimy]=1.0/complex<double>(0,wvl*Dz)*Uin_ptr[jx+jy*dimy]*polar(1.0,k/(2*Dz)*(x2_l[jx]*x2_l[jx]+y2_l[jy]*y2_l[jy]));
		}
	}
	*x2_ptrptr=x2_l; // return coordinates of output field in respective pointer
	(*y2_ptrptr)=(double*)calloc(dimx,sizeof(double));
	memcpy((*y2_ptrptr),x2_l,dimx*sizeof(double));
	return PROP_NO_ERR;
}
*/
/**
 * \detail scalar_RichardsonWolf_ts 
 *
 * computes scalar propagation to the focus of a lense thread safe. see Masud Mansuripur, Classical Optics and its Applications for reference
 *
 * \param[in]	complex<double>*	Uin_ptr
 *				unsigned long		dimx
 *				unsigned long		dimy
 *				double				wvl
 *				double*				x1_ptr
 *				double*				y1_ptr
 *				double				f
 *				double				Dz
 *				double**			x2_ptrptr
 *				double**			y2_ptrptr
 *				fftw_plan			&p
 * 
 * \return propErr
 * \sa 
 * \remarks 
 * \author Mauch
 */
/*propError scalar_RichardsonWolf_ts(complex<double>* Uin_ptr, unsigned int dimx, unsigned int dimy, double wvl, double* x1_ptr, double* y1_ptr, double f, double Dz, double** x2_ptrptr, double** y2_ptrptr, fftw_plan &p)
{
	// we can only handle regularly squared grids here
	if (dimx!=dimy)
		return PROP_ERR;
	double k=2*M_PI/wvl;
	double dx1=abs(x1_ptr[0]-x1_ptr[1]);
	double dy1=abs(y1_ptr[0]-y1_ptr[1]);
	if (dx1!=dy1)
		return PROP_ERR;

	// observation plane
	// coordinates
	double *x2_l=(double*)calloc(dimx,sizeof(double));
	double *y2_l=x2_l;
	for (unsigned long jx=0; jx<dimx; jx++)
	{
		x2_l[jx]=(-1.0*dimx/2+jx)/(dimx*dx1)*wvl*f;
	}

	// evaluate scalar Richardson Wolf integral
	for (unsigned long jx=0; jx<dimx;jx++)
	{
		double sigmaX=-x1_ptr[jx]/f;
		for (unsigned long jy=0; jy<dimy;jy++)
		{
			double sigmaY=-y1_ptr[jy]/f;
			double GktSqr=1-sigmaX*sigmaX-sigmaY*sigmaY;
			// cut off evanescent waves
			if (GktSqr<0)
			{
				GktSqr=0.0;
				Uin_ptr[jx+jy*dimy]=0;
			}
			else
				Uin_ptr[jx+jy*dimy]=complex<double>(0.0,-1.0)*f*Uin_ptr[jx+jy*dimy]/pow(polar(1-sigmaX*sigmaX-sigmaY*sigmaY,0.0),0.25)*polar(1.0,k*Dz*sqrt(GktSqr));
		}
	}
	ft2_ts(Uin_ptr, dimx, dimy, dx1/f, p);

	*x2_ptrptr=x2_l; // return coordinates of output field in respective pointer
	(*y2_ptrptr)=(double*)calloc(dimx,sizeof(double));
	memcpy((*y2_ptrptr),x2_l,dimx*sizeof(double));
	return PROP_NO_ERR;
}
*/
/**
 * \detail fraunhofer_ts 
 *
 * computes fraunhofer propagation thread safe. see Jason D. Schmidt, Numerical Simulation of Optical Wave Propagation for reference
 *
 * \param[in]	complex<double>*	Uin_ptr
 *				unsigned long		dimx
 *				unsigned long		dimy
 *				double				wvl
 *				double*				x1_ptr
 *				double*				y1_ptr
 *				double				Dz
 *				double**			x2_ptrptr
 *				double**			y2_ptrptr
 *				fftw_plan			&p
 * 
 * \return propErr
 * \sa 
 * \remarks 
 * \author Mauch
 */
/*propError fraunhofer_ts(complex<double>* Uin_ptr, unsigned int dimx, unsigned int dimy, double wvl, double* x1_ptr, double* y1_ptr, double Dz, double** x2_ptrptr, double** y2_ptrptr, fftw_plan &p)
{
	// we can only handle regularly squared grids here
	if (dimx!=dimy)
		return PROP_ERR;
	double k=2*M_PI/wvl;
	double dx1=abs(x1_ptr[0]-x1_ptr[1]);
	double dy1=abs(y1_ptr[0]-y1_ptr[1]);
	if (dx1!=dy1)
		return PROP_ERR;

	// observation plane
	// coordinates
	double *x2_l=(double*)calloc(dimx,sizeof(double));
	double *y2_l=x2_l;
	for (unsigned long jx=0; jx<dimx; jx++)
	{
		x2_l[jx]=(-1.0*dimx/2+jx)/(dimx*dx1)*wvl*Dz;
	}
	
	ft2_ts(Uin_ptr, dimx, dimy, dx1, p);
	for (unsigned long jx=0; jx<dimx;jx++)
	{
		for (unsigned long jy=0; jy<dimy; jy++)
		{
			Uin_ptr[jx+jy*dimy]=Uin_ptr[jx+jy*dimy]/complex<double>(0,wvl*Dz)*polar(1.0,k/(2*Dz)*(x2_l[jx]*x2_l[jx]+y2_l[jy]*y2_l[jy]));
		}
	}
	*x2_ptrptr=x2_l; // return coordinates of output field in respective pointer
	(*y2_ptrptr)=(double*)calloc(dimx,sizeof(double));
	memcpy((*y2_ptrptr),x2_l,dimx*sizeof(double));
	return PROP_NO_ERR;
}
*/
/**
 * \detail fraunhofer_1D_ts 
 *
 * computes 1D fraunhofer propagation thread safe. see Jason D. Schmidt, Numerical Simulation of Optical Wave Propagation for reference
 *
 * \param[in]	complex<double>*	Uin_ptr
 *				unsigned long		dimx
 *				double				wvl
 *				double*				x1_ptr
 *				double				Dz
 *				double**			x2_ptrptr
 *				fftw_plan			&p
 * 
 * \return propErr
 * \sa 
 * \remarks 
 * \author Mauch
 */
/*propError fraunhofer_1D_ts(complex<double>* Uin_ptr, unsigned int dimx, double wvl, double* x1_ptr, double Dz, double** x2_ptrptr, fftw_plan &p)
{
	double k=2*M_PI/wvl;
	double dx1=abs(x1_ptr[0]-x1_ptr[1]);

	// observation plane
	// coordinates
	double *x2_l=(double*)calloc(dimx,sizeof(double));
	for (unsigned long jx=0; jx<dimx; jx++)
	{
		x2_l[jx]=(-1.0*dimx/2+jx)/(dimx*dx1)*wvl*Dz;
	}
	
	ft1_ts(Uin_ptr, dimx, dx1, p);
	for (unsigned long jx=0; jx<dimx;jx++)
	{
		Uin_ptr[jx]=Uin_ptr[jx]/complex<double>(0,wvl*Dz)*polar(1.0,k/(2*Dz)*x2_l[jx]*x2_l[jx]);
	}
	*x2_ptrptr=x2_l; // return coordinates of output field in respective pointer
	return PROP_NO_ERR;
}
*/

/**
 * \detail fftshift 
 *
 * performes fftshift equivalent to matlabs fftshift command. see William H. Press, Numerical recipes, 3rd edition for reference
 *
 * \param[in]	complex<double>*	in
 *				unsigned long		dimx
 *				unsigned long		dimy
 * 
 * \return propErr
 * \sa 
 * \remarks 
 * \author Mauch
 */
propError fftshift(complex<double>* in, unsigned int dimx, unsigned int dimy)
{
	complex<double>* tmp=(complex<double>*)malloc(dimx*dimy*sizeof(complex<double>));
	for (unsigned int jx=0;jx<dimx;jx++)
	{
		unsigned int jjx=(jx+dimx/2) % dimx;
		for (unsigned int jy=0;jy<dimy;jy++)
		{
			unsigned int jjy=(jy+dimy/2) % dimy;
			tmp[jjx*dimy+jjy]=in[jx * dimy + jy];
		}
	}
	memcpy(in, tmp, dimx*dimy*sizeof(complex<double>));
	delete tmp;
	return PROP_NO_ERR;
}

/**
 * \detail simConfSensorSig 
 *
 * simulates the sensor signal of a confocal point sensor. see F. Mauch, Improved signal model for confocal sensors accounting for object depending artifacts, Optics Express, 20, 19936-19945 (2012)
 *
 * \param[in]	double				NA			:	numerical aperture of objective lens
 *				double				magnif		:	magnification of objective lens
 *				double				wvl			:	wavelength of illumination in µm
 *				unsigned int		n			:	number of sample points per dimension
 *				double				gridWidth   :   width of the square grid in the objective lens aperture
 *				double*				pScanWidth	:	vector containing the scan widths in x, y and z in µm
 *				unsigned int*		pScanNumber :	number of steps of the scan in x, y and z
 *				double*				pAberrVec	:	vector containing the zernike coefficients of the aberrations of the confocal sensor
 * \param[out]	double**			pRawSig		:	3D-array containing the raw signals. The layout is: first z, then x, then y.
 * 
 * \return propError
 * \sa 
 * \remarks 
 * \author Mauch
 */
propError simConfSensorSig(double **ppRawSig, ConfPoint_Params params, ConfPointObject_Params paramsObject, bool runOnCPU)
{
	if (runOnCPU)
	{
		cout << "error in simConfSensorSig: running on CPU not implemented yet" << endl;
		return PROP_ERR;
	}
	else
	{
		ConfPoint_KernelParams l_kernelParams;
		l_kernelParams.gridWidth=params.gridWidth;
		l_kernelParams.magnif=params.magnif;
		l_kernelParams.wvl=params.wvl;
		l_kernelParams.n=params.n;
		l_kernelParams.NA=params.NA;
		l_kernelParams.scanNumber=params.scanNumber;
		l_kernelParams.scanStep=params.scanStep;
		l_kernelParams.apodisationRadius=params.apodisationRadius;
		memcpy(&(l_kernelParams.pAberrVec[0]),&(params.pAberrVec[0]),16*sizeof(double));

        ConfPoint_KernelObjectParams l_kernelParamsObject;
        l_kernelParamsObject.A=paramsObject.A;
        l_kernelParamsObject.kN=paramsObject.kN;
        if (!cu_simConfPointSensorSig_wrapper(ppRawSig, l_kernelParams, l_kernelParamsObject))
		{
			cout << "error in simConfRawSig: cu_simConfPointSensorSig_wrapper() returned an error" << endl;
			return PROP_ERR;
		}

		//unsigned int test=cu_testReduce_wrapper();
		//cout << "result of reduce: " << test << endl;

		return PROP_NO_ERR;
	}
	return PROP_NO_ERR;
}

propError simConfSensorSig(double **ppRawSig, ConfPoint_Params params, bool runOnCPU)
{
    return PROP_NO_ERR;
}
/**
 * \detail simConfRawSig 
 *
 * simulates the raw signal of a confocal point sensor. see F. Mauch, Improved signal model for confocal sensors accounting for object depending artifacts, Optics Express, 20, 19936-19945 (2012)
 *
 * \param[in]	double				NA			:	numerical aperture of objective lens
 *				double				magnif		:	magnification of objective lens
 *				double				wvl			:	wavelength of illumination in µm
 *				unsigned int		n			:	number of sample points per dimension
 *				double				gridWidth   :   width of the square grid in the objective lens aperture
 *				double*				pScanWidth	:	vector containing the scan widths in x, y and z in µm
 *				unsigned int*		pScanNumber :	number of steps of the scan in x, y and z
 *				double*				pAberrVec	:	vector containing the zernike coefficients of the aberrations of the confocal sensor
 * \param[out]	double**			pRawSig		:	3D-array containing the raw signals. The layout is: first z, then x, then y.
 * 
 * \return propError
 * \sa 
 * \remarks 
 * \author Mauch
 */
propError simConfRawSig(double **ppRawSig, ConfPoint_Params params, bool runOnCPU)
{
	if (runOnCPU)
	{
		cout << "error in simConfRawSig: running on CPU not implemented yet" << endl;
		return PROP_ERR;
	}
	else
	{
		ConfPoint_KernelParams l_kernelParams;
		l_kernelParams.gridWidth=params.gridWidth;
		l_kernelParams.magnif=params.magnif;
		l_kernelParams.wvl=params.wvl;
		l_kernelParams.n=params.n;
		l_kernelParams.NA=params.NA;
		l_kernelParams.scanNumber=params.scanNumber;
		l_kernelParams.scanStep=params.scanStep;
		l_kernelParams.apodisationRadius=params.apodisationRadius;
		memcpy(&(l_kernelParams.pAberrVec[0]),&(params.pAberrVec[0]),16*sizeof(double));
        if (!cu_simConfPointRawSig_wrapper(ppRawSig, l_kernelParams))
        //if (!cu_simConfPointRawSig_wrapperTest(ppRawSig, l_kernelParams))
		{
			cout << "error in simConfRawSig: cu_simConfPointRawSig_wrapper() returned an error" << endl;
			return PROP_ERR;
		}

		//unsigned int test=cu_testReduce_wrapper();
		//cout << "result of reduce: " << test << endl;

		return PROP_NO_ERR;
	}
	return PROP_NO_ERR;
}

//int main(int argc, char* argv[])
//{
//	double deltaZ=10e-3; //[mm]
//	ConfPoint_KernelParams l_params;
//	l_params.gridWidth=100; // [mm]
//	l_params.magnif=50;
//	l_params.n=1024;
//	l_params.NA=0.5;
//	l_params.scanNumber=make_uint3(1,1,501);
//	l_params.scanStep=make_double3(0/l_params.scanNumber.x,0/l_params.scanNumber.y,deltaZ/l_params.scanNumber.z);
//	l_params.wvl=830e-6; //[mm]
//	l_params.pAberrVec[0]=0;
//	l_params.pAberrVec[1]=0;
//	l_params.pAberrVec[2]=0;
//	l_params.pAberrVec[3]=0;
//	l_params.pAberrVec[4]=0;
//	l_params.pAberrVec[5]=0;
//	l_params.pAberrVec[6]=0;
//	l_params.pAberrVec[7]=0;
//	l_params.pAberrVec[8]=0; // primary spherical
//	l_params.pAberrVec[9]=0;
//	l_params.pAberrVec[10]=0;
//	l_params.pAberrVec[11]=0;
//	l_params.pAberrVec[12]=0;
//	l_params.pAberrVec[13]=0;
//	l_params.pAberrVec[14]=0;
//	l_params.pAberrVec[15]=0;
//
//	double *l_pRawSig;
//
//	bool test=cu_simConfPointRawSig_wrapper(&l_pRawSig,l_params);
//
//	char t_filename[512];
//	sprintf(t_filename, "E:\\rawSig.txt");
//	FILE* hFile;
//	hFile = fopen( t_filename, "w" ) ;
//
//
//	if ( (hFile == NULL) )
//		return 1;
//
//
//	for (unsigned int jy=0; jy<l_params.scanNumber.y; jy++)
//	{
//		for (unsigned int jx=0; jx<l_params.scanNumber.x; jx++)
//		{
//			for (unsigned int jz=0; jz<l_params.scanNumber.z; jz++)
//			{
//				fprintf(hFile, " %.16e;\n", l_pRawSig[jz+jx*l_params.scanNumber.z+jy*l_params.scanNumber.z*l_params.scanNumber.x]);
//			}
//		}
//	}
//
//	fclose(hFile);


//*************************************************************************************

	 //int N=256;

	 //complex<double> test=complex<double>(1.0,0.0);
	 //test=test*polar(1.0,M_PI/2);

	 //complex<double> *Uin_ptr=(complex<double>*)calloc(N*N,sizeof(complex<double>));

	 //for (unsigned int jx=120;jx<149;jx++)
	 //{
		// for (unsigned int jy=120;jy<149;jy++)
		// {
		//	Uin_ptr[jx+jy*N]=complex<double>(1.0,0.0);
		// }
	 //}

	 //double* x1_ptr=(double*)malloc(N*sizeof(double));
	 //double* y1_ptr=(double*)malloc(N*sizeof(double));
	 //double dx1=1E-6;
	 //double dy1=1E-6;
	 //double* x2_ptrptr;
	 //double* y2_ptrptr;
	 //for (unsigned long jx=0; jx<N; jx++)
	 //{
		//x1_ptr[jx]=(-1.0*N/2+jx)*dx1;
	 //}
	 //memcpy(y1_ptr,x1_ptr,N*sizeof(double));

	 //double wvl=833E-9;
	 //double dx2=10E-6;
	 //double dz= 10E-3;
	 //double f=3.6E-3;
	 //double deFoc=1E-6;
	 ////ft1(Uin_ptr, N, dx1);
	 ////fresnel_two_step(Uin_ptr, N, N, wvl, x1_ptr, y1_ptr, dx2, dz, &x2_ptrptr, &y2_ptrptr);
	 ////fraunhofer(Uin_ptr, N, N, wvl, x1_ptr, y1_ptr, dz, &x2_ptrptr, &y2_ptrptr);
	 //scalar_RichardsonWolf(Uin_ptr, N, N, wvl, x1_ptr, y1_ptr, f, deFoc, &x2_ptrptr, &y2_ptrptr);
	 ////ft2(Uin_ptr, N, N, dx1);
	 //
	 //char t_filename[512] = "Real.txt";

	 //FILE* hFileReal;
	 //hFileReal = fopen( t_filename, "w" ) ;

	 //sprintf(t_filename, "Imag.txt");
	 //FILE* hFileImag;
	 //hFileImag = fopen( t_filename, "w" ) ;


	 //if ( (hFileReal == NULL) || (hFileImag == NULL) )
		// return 1;

	 //for (unsigned int jy=0;jy<N;jy++)
	 //{
		// for (unsigned int jx=0;jx<N;jx++)
		// {
		//	 if (jx+1 == N)
		//	 {
		//		fprintf(hFileReal, " %.16e;\n", Uin_ptr[jx+jy*N].real());
		//		fprintf(hFileImag, " %.16e;\n", Uin_ptr[jx+jy*N].imag());
		//	 }
		//	 else
		//	 {
		//		fprintf(hFileReal, " %.16e;", Uin_ptr[jx+jy*N].real());
		//		fprintf(hFileImag, " %.16e;", Uin_ptr[jx+jy*N].imag());
		//	 }

		// }
	 //}

	 //fclose(hFileReal);
	 //fclose(hFileImag);

  //   delete Uin_ptr;
	 //delete x1_ptr;
	 //delete x2_ptrptr;


//	 return 0;
//}
