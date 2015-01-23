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

/**\file Interpolator.cpp
* \brief 
* 
*           
* \author Mauch
*/

/**
 *\defgroup Interpolator
 */

#include "Interpolator.h"

/**
 * \detail nearestNeighbour 
 *
 * Given array yin containing tabulated function on a regular, i.e. yin=f(x1,x2), 
 * where x1=x1_0+index*delta_x1. The value y of the nearest neighbour to x1_out, x2_out is returned in yout_ptr
 *
 * \param[in]	const double&				x1_0
 *				const double&				x2_0
 *				const double&				delta_x1
 *				const double&				delta_x2
 *				const double*				yin_tpr
 *				const unsigned int			dim_x1
 *				const unsigned int			dim_x2
 *				const double&				x1_out
 *				const double&				x2_out
 *				double*						yout_ptr
 * 
 * \return propErr
 * \sa 
 * \remarks 
 * \author Mauch
 */
interpErr Interpolator::nearestNeighbour(const double& x1_0, const double& x2_0, const double& delta_x1, const double& delta_x2, const double *yin_ptr, const unsigned int dim_x1, const unsigned int dim_x2, const double &x1_out, const double &x2_out, double *yout_ptr)
{
	if (!nearestNeighbour_hostdevice(x1_0, x2_0, delta_x1, delta_x2, yin_ptr, dim_x1, dim_x2, x1_out, x2_out, yout_ptr) )
		return INTERP_ERR;
	return INTERP_NO_ERR;
};
/**
 * \detail spline 
 *
 * Given arrays x and y containing tabulated function, i.e. yi=f(xi),
 * with x0<x1<...xN-1, and given values yp1 and ypN for the first 
 * derivative of the interpolating function at points 0 and N-1, respectively, 
 * this routine returns an array y2 that contaions the second derivatives of 
 * the interpolating function at the tabulated points xi. If yp1 and/or
 * ypN are equal to 1E30 or larger, the routine is signaled to set the 
 * corresponding boundary condition for a natural spline, with zero second
 * derivative on that boundary
 * see Numerical recipes in C++ second edition pp.118 for reference
 *
 * \param[in]	double*				x
 *				double*				y
 *				const unsigned int	width
 *				conste double		yp1
 *				conste double		ypn
 *				double*				y2
 * 
 * \return propErr
 * \sa 
 * \remarks 
 * \author Mauch
 */
interpErr Interpolator::spline(double *x, double *y, const unsigned int width, const double yp1,  const double ypn, double *y2)
{
	int i,k;
	double p,qn,sig,un;

	double *u;
	u=(double*)malloc(width*sizeof(double));
	if (yp1 > 0.99e30)
		y2[0]=u[0]=0.0;
	else
	{
		y2[0]=-0.5;
		u[0]=(3.0/(x[1]-x[0]))*((y[1]-y[0])/(x[1]-x[0])-yp1);
	}
	for (i=1;i<width-1;i++)
	{
		sig=(x[i]-x[i-1])/(x[i+1]-x[i-1]);
		p=sig*y2[i-1]+2.0;
		y2[i]=(sig-1.0)/p;
		u[i]=(y[i+1]-y[i])/(x[i+1]-x[i])-(y[i]-y[i-1])/(x[i]-x[i-1]);
		u[i]=(6.0*u[i]/(x[i+1]-x[i-1])-sig*u[i-1])/p;
	}
	if (ypn > 0.99e30)
		qn=un=0.0;
	else
	{
		qn=0.5;
		un=(3.0/(x[width-1]-x[width-2]))*(ypn-(y[width-1]-y[width-2])/(x[width-1]-x[width-2]));
	}
	y2[width-1]=(un-qn*u[width-2])/(qn*y2[width-2]+1.0);
	for (k=width-2;k>=0;k--)
		y2[k]=y2[k]*y2[k+1]+u[k];

	// cleanup
	delete u;
	u=NULL;

	return INTERP_NO_ERR;
}

/**
 * \detail spline 
 *
 * Given arrays x and y containing tabulated function, i.e. yi=f(xi),
 * with x0<x1<...xN-1, and given values yp1 and ypN for the first 
 * derivative of the interpolating function at points 0 and N-1, respectively, 
 * this routine returns an array y2 that contaions the second derivatives of 
 * the interpolating function at the tabulated points xi. If yp1 and/or
 * ypN are equal to 1E30 or larger, the routine is signaled to set the 
 * corresponding boundary condition for a natural spline, with zero second
 * derivative on that boundary
 * see Numerical recipes in C++ second edition pp.118 for reference
 *
 * \param[in]	double*				x
 *				complex<double>*	y
 *				const unsigned int	width
 *				conste double		yp1
 *				conste double		ypn
 *				complex<double>*	y2
 * 
 * \return propErr
 * \sa 
 * \remarks 
 * \author Mauch
 */
interpErr Interpolator::spline(double *x, complex<double> *y, const unsigned int width, const double yp1,  const double ypn, complex<double> *y2)
{
	int i,k;
	double sig;
	complex<double> p, qn, un;

	complex<double> *u;
	u=(complex<double>*)malloc(width*sizeof(complex<double>));
	if (yp1 > 0.99e30)
		y2[0]=u[0]=polar(0.0,0.0);
	else
	{
		y2[0]=complex<double>(-0.5,-0.5);
		u[0]=(3.0/(x[1]-x[0]))*((y[1]-y[0])/(x[1]-x[0])-yp1);
	}
	for (i=1;i<width-1;i++)
	{
		sig=(x[i]-x[i-1])/(x[i+1]-x[i-1]);
		p=sig*y2[i-1]+2.0;
		y2[i]=(sig-1.0)/p;
		u[i]=(y[i+1]-y[i])/(x[i+1]-x[i])-(y[i]-y[i-1])/(x[i]-x[i-1]);
		u[i]=(6.0*u[i]/(x[i+1]-x[i-1])-sig*u[i-1])/p;
	}
	if (ypn > 0.99e30)
		qn=un=complex<double>(0.0,0.0);
	else
	{
		qn=complex<double>(0.5,0.5);
		un=(3.0/(x[width-1]-x[width-2]))*(ypn-(y[width-1]-y[width-2])/(x[width-1]-x[width-2]));
	}
	y2[width-1]=(un-qn*u[width-2])/(qn*y2[width-2]+1.0);
	for (k=width-2;k>=0;k--)
		y2[k]=y2[k]*y2[k+1]+u[k];

	// cleanup
	delete u;
	u=NULL;

	return INTERP_NO_ERR;
}

/**
 * \detail splint 
 *
 * Given the arrays xa and ya, which tabulate a function, and given the array y2a, 
 * which is the output from spline above, and given a value of x, this routine returns a 
 * cubic spline interpolated value of y
 * see Numerical recipes in C++ second edition pp.119 for reference
 *
 * \param[in]	double*				xa
 *				double*				ya
 *				double*				y2a
 *				const unsigned int	width
 *				conste double		x
 *				double&				y
 * 
 * \return propErr
 * \sa 
 * \remarks 
 * \author Mauch
 */
interpErr Interpolator::splint(double *xa, double *ya, double *y2a, const unsigned int width, const double x,  double *y)
{
	int k;
	double h,b,a;

	int klo=0;
	int khi=width-1;
	while (khi-klo > 1) // find bucket of x in xa
	{
		k=(khi+klo) >> 1;
		if (xa[k] > x) khi=k;
		else klo=k;
	}
	h=xa[khi]-xa[klo];
	if (h==0.0) 
	{
		cout << "error in DiffRayField_Freeform.splint(): the tabulated x-values must be distinct" << endl;
	}
	a=(xa[khi]-x)/h;
	b=(x-xa[klo])/h;
	*y=a*ya[klo]+b*ya[khi]+((a*a*a-a)*y2a[klo]+(b*b*b-b)*y2a[khi])*(h*h)/6.0;
	return INTERP_NO_ERR;
}

/**
 * \detail splint 
 *
 * Given the arrays xa and ya, which tabulate a function, and given the array y2a, 
 * which is the output from spline above, and given a value of x, this routine returns a 
 * cubic spline interpolated value of y
 * see Numerical recipes in C++ second edition pp.119 for reference
 *
 * \param[in]	double*				xa
 *				double*				ya
 *				double*				y2a
 *				const unsigned int	width
 *				conste double		x
 *				double&				y
 * 
 * \return propErr
 * \sa 
 * \remarks 
 * \author Mauch
 */
interpErr Interpolator::splint(double *xa, complex<double> *ya, complex<double> *y2a, const unsigned int width, const double x, complex<double> *y)
{
	int k;
	double h,b,a;

	int klo=0;
	int khi=width-1;
	while (khi-klo > 1) // find bucket of x in xa
	{
		k=(khi+klo) >> 1;
		if (xa[k] > x) khi=k;
		else klo=k;
	}
	h=xa[khi]-xa[klo];
	if (h==0.0) 
	{
		cout << "error in DiffRayField_Freeform.splint(): the tabulated x-values must be distinct" << endl;
	}
	a=(xa[khi]-x)/h;
	b=(x-xa[klo])/h;
	*y=a*ya[klo]+b*ya[khi]+((a*a*a-a)*y2a[klo]+(b*b*b-b)*y2a[khi])*(h*h)/6.0;
	return INTERP_NO_ERR;
}

/**
 * \detail splie2 
 *
 * Given an m by n tabulated function ya, and tabulated independent variables x1a and x2a,
 * this routine constructs one dimensional natural cubic splines of the rows of ya
 * and returns the second derivatives in the array y2a.
 * see Numerical recipes in C++ 2nd edition second edition pp.131 for reference
 *
 * \param[in]	double*				xa
 *				double*				ya
 *				double*				y2a
 *				const unsigned int	width
 *				conste double		x
 *				double&				y
 * 
 * \return propErr
 * \sa 
 * \remarks 
 * \author Mauch
 */
// see Numerical recipes in C++ second edition pp.131 for reference
interpErr Interpolator::splie2(double *x1a, double *x2a, double *ya, double *y2a, const unsigned int width, const unsigned int height)
{
	unsigned int j, k;

	double *ya_t, *y2a_t;
	ya_t=(double*)malloc(width*sizeof(double));
	y2a_t=(double*)malloc(width*sizeof(double));

	for (j=0;j<height;j++)
	{
		for (k=0;k<width;k++)
			ya_t[k]=ya[k+j*width];
		spline(x2a,ya_t,width,1.0e30,1.0e30,y2a_t);
		for (k=0;k<width;k++)
			y2a[k+j*width]=y2a_t[k];
	}

	// cleanup
	delete ya_t;
	ya_t=NULL;
	delete y2a_t;
	y2a_t=NULL;
	return INTERP_NO_ERR;
}

/**
 * \detail splie2 
 *
 * Given an m by n tabulated function ya, and tabulated independent variables x1a and x2a,
 * this routine constructs one dimensional natural cubic splines of the rows of ya
 * and returns the second derivatives in the array y2a.
 * see Numerical recipes in C++ 2nd edition second edition pp.131 for reference
 *
 * \param[in]	double*				xa
 *				complex<double>*	ya
 *				complex<double>*	y2a
 *				const unsigned int	width
 *				conste double		x
 *				double&				y
 * 
 * \return propErr
 * \sa 
 * \remarks 
 * \author Mauch
 */
// see Numerical recipes in C++ second edition pp.131 for reference
interpErr Interpolator::splie2(double *x1a, double *x2a, complex<double> *ya, complex<double> *y2a, const unsigned int width, const unsigned int height)
{
	unsigned int j, k;

	complex<double> *ya_t, *y2a_t;
	ya_t=(complex<double>*)malloc(width*sizeof(complex<double>));
	y2a_t=(complex<double>*)malloc(width*sizeof(complex<double>));

	for (j=0;j<height;j++)
	{
		for (k=0;k<width;k++)
			ya_t[k]=ya[k+j*width];
		spline(x1a,ya_t,width,1.0e30,1.0e30,y2a_t);
		for (k=0;k<width;k++)
			y2a[k+j*width]=y2a_t[k];
	}

	// cleanup
	delete ya_t;
	ya_t=NULL;
	delete y2a_t;
	y2a_t=NULL;

	return INTERP_NO_ERR;
}

/**
 * \detail splin2 
 *
 * Given x1a, x2a, ya, m, n as described in splie2 and y2a as produced by that routine;
 * and given a desired interpolating point x1,x2; this routine returns an interpolated 
 * function value y by bicubic spline interpolation.
 * see Numerical recipes in C++ 2nd edition second edition pp.131 for reference
 *
 * \param[in]	double*				x1a
 *				double*				x2a
 *				const unsigned int	width
 *				const unsigned int	height
 *				double*				ya
 *				double*				y2a
 *				conste double		x1
 *				conste double		x2
 *				double&				y
 * 
 * \return propErr
 * \sa 
 * \remarks 
 * \author Mauch
 */
interpErr Interpolator::splin2(double *x1a, double *x2a, const unsigned int width, const unsigned int height, double *ya, double *y2a, const double x1, const double x2, double *y)
{
	int j,k;

	double *ya_t, *y2a_t, *yytmp, *ytmp;
	ya_t=(double*)malloc(width*sizeof(double));
	y2a_t=(double*)malloc(width*sizeof(double));
	yytmp=(double*)malloc(height*sizeof(double));
	ytmp=(double*)malloc(height*sizeof(double));

	for (j=0;j<height;j++)
	{
		for (k=0;k<width;k++)
		{
			ya_t[k]=ya[k+j*width];
			y2a_t[k]=y2a[k+j*width];
		}
		this->splint(x1a,ya_t,y2a_t,width, x1,&yytmp[j]); 
	}
	spline(x2a,yytmp,height,1.0e30,1.0e30,ytmp);
	splint(x2a,yytmp,ytmp,height,x2,y);

	// cleanup
	delete ya_t;
	ya_t=NULL;
	delete y2a_t;
	y2a_t=NULL;
	delete yytmp;
	yytmp=NULL;
	delete ytmp;
	ytmp=NULL;

	return INTERP_NO_ERR;
}

/**
 * \detail splin2 
 *
 * Given x1a, x2a, ya, m, n as described in splie2 and y2a as produced by that routine;
 * and given a desired interpolating point x1,x2; this routine returns an interpolated 
 * function value y by bicubic spline interpolation.
 * see Numerical recipes in C++ 2nd edition second edition pp.131 for reference
 *
 * \param[in]	double*				x1a
 *				double*				x2a
 *				const unsigned int	width
 *				const unsigned int	height
 *				double*				ya
 *				double*				y2a
 *				conste double		x1
 *				conste double		x2
 *				double&				y
 * 
 * \return propErr
 * \sa 
 * \remarks 
 * \author Mauch
 */
interpErr Interpolator::splin2(double *x1a, double *x2a, const unsigned int width, const unsigned int height, complex<double> *ya, complex<double> *y2a, const double x1, const double x2, complex<double> *y)
{
	int j,k;

	complex<double> *ya_t, *y2a_t, *yytmp, *ytmp;
	ya_t=(complex<double>*)malloc(width*sizeof(complex<double>));
	y2a_t=(complex<double>*)malloc(width*sizeof(complex<double>));
	yytmp=(complex<double>*)malloc(height*sizeof(complex<double>));
	ytmp=(complex<double>*)malloc(height*sizeof(complex<double>));

	for (j=0;j<height;j++)
	{
		for (k=0;k<width;k++)
		{
			ya_t[k]=ya[k+j*width];
			y2a_t[k]=y2a[k+j*width];
		}
		this->splint(x1a,ya_t,y2a_t,width, x1,&yytmp[j]);
	}
	spline(x2a,yytmp,height,1.0e30,1.0e30,ytmp);
	splint(x2a,yytmp,ytmp,height,x2,y);

	// cleanup
	delete ya_t;
	ya_t=NULL;
	delete y2a_t;
	y2a_t=NULL;
	delete yytmp;
	yytmp=NULL;
	delete ytmp;
	ytmp=NULL;

	return INTERP_NO_ERR;
}

/**
 * \detail initInterpolation 
 *
 * this routine basically wraps the private function splie2
 *
 * \param[in]	double*				x1a
 *				double*				x2a
 *				const unsigned int	width
 *				const unsigned int	height
 *				double*				ya
 *				conste double		x1
 *				conste double		x2
 *				double&				y
 * 
 * \return propErr
 * \sa 
 * \remarks 
 * \author Mauch
 */
interpErr Interpolator::initInterpolation(double *x1a, double *x2a, double *ya, const unsigned int width, const unsigned int height)
{
	// if y2a pointer was initialized before, discard old data
	if (this->y2a_ptr != NULL)
	{
		delete this->y2a_ptr;
	}
	this->y2a_ptr=(double*)calloc(width*height,sizeof(double));
	this->splie2(x1a, x2a, ya, this->y2a_ptr, width, height);
	return INTERP_NO_ERR;
};

/**
 * \detail initInterpolation 
 *
 * this routine basically wraps the private function spline
 *
 * \param[in]	double*				x1a
 *				double*				x2a
 *				const unsigned int	width
 *				const unsigned int	height
 *				double*				ya
 *				conste double		x1
 *				conste double		x2
 *				double&				y
 * 
 * \return propErr
 * \sa 
 * \remarks 
 * \author Mauch
 */
interpErr Interpolator::initInterpolation(double *x, double *y, const unsigned int width, const double yp1,  const double ypn)
{
	// if y2 pointer was initialized before, discard old data
	if (this->y2_ptr != NULL)
	{
		delete this->y2_ptr;
	}
	this->y2_ptr=(double*)calloc(width,sizeof(double));
	this->spline(x, y, width, yp1, ypn, this->y2_ptr);
	return INTERP_NO_ERR;
};

/**
 * \detail initInterpolation 
 *
 * this routine basically wraps the private function splie2 for complex data
 *
 * \param[in]	double*				x1a
 *				double*				x2a
 *				const unsigned int	width
 *				const unsigned int	height
 *				complex<double>*	ya
 *				conste double		x1
 *				conste double		x2
 *				double&				y
 * 
 * \return propErr
 * \sa 
 * \remarks 
 * \author Mauch
 */
interpErr Interpolator::initInterpolation(double *x1a, double *x2a, complex<double> *ya, const unsigned int width, const unsigned int height)
{
	// if y2a pointer was initialized before, discard old data
	if (this->y2ca_ptr != NULL)
	{
		delete this->y2ca_ptr;
	}
	this->y2ca_ptr=(complex<double>*)calloc(width*height,sizeof(complex<double>));
	this->splie2(x1a, x2a, ya, this->y2ca_ptr, width, height);
	return INTERP_NO_ERR;
};

/**
 * \detail initInterpolation 
 *
 * this routine basically wraps the private function spline for complex data
 *
 * \param[in]	double*				x1a
 *				double*				x2a
 *				const unsigned int	width
 *				const unsigned int	height
 *				complex<double>*	ya
 *				conste double		x1
 *				conste double		x2
 *				double&				y
 * 
 * \return propErr
 * \sa 
 * \remarks 
 * \author Mauch
 */
interpErr Interpolator::initInterpolation(double *x, complex<double> *y, const unsigned int width, const double yp1,  const double ypn)
{
	// if y2 pointer was initialized before, discard old data
	if (this->y2c_ptr != NULL)
	{
		delete this->y2c_ptr;
	}
	this->y2c_ptr=(complex<double>*)calloc(width,sizeof(complex<double>));
	this->spline(x, y, width, yp1, ypn, this->y2c_ptr);
	return INTERP_NO_ERR;
};

/**
 * \detail doInterpolation 
 *
 * this routine basically wraps the private function splin2
 *
 * \param[in]	double*				x1a
 *				double*				x2a
 *				const unsigned int	width
 *				const unsigned int	height
 *				double*				ya
 *				conste double		x1
 *				conste double		x2
 *				double&				y
 * 
 * \return propErr
 * \sa 
 * \remarks 
 * \author Mauch
 */
interpErr Interpolator::doInterpolation(double *x1a, double *x2a, const unsigned int width, const unsigned int height, double *ya, const double x1, const double x2, double *y)
{
	if (this->y2a_ptr == NULL)
	{
		cout << "error in Interpolator.doInterpolation(): array containing second derivative data is not initialized. Try calling initInterpolation... " << endl;
		return INTERP_ERR;
	}
	else
		this->splin2(x1a, x2a, width, height, ya, this->y2a_ptr, x1, x2, y);
	return INTERP_NO_ERR;
};

/**
 * \detail doInterpolation 
 *
 * this routine basically wraps the private function splint
 *
 * \param[in]	double*				xa
 *				double*				ya
 *				const unsigned int	width
 *				conste double		x
 *				double&				y
 * 
 * \return propErr
 * \sa 
 * \remarks 
 * \author Mauch
 */
interpErr Interpolator::doInterpolation(double *xa, double *ya, const unsigned int width, const double x,  double *y)
{
	if (this->y2_ptr == NULL)
	{
		cout << "error in Interpolator.doInterpolation(): array containing second derivative data is not initialized. Try calling initInterpolation... " << endl;
		return INTERP_ERR;
	}
	else
		this->splint(xa, ya, this->y2_ptr, width, x, y);
	return INTERP_NO_ERR;
};

/**
 * \detail doInterpolation 
 *
 * this routine basically wraps the private function splin2 for complex data
 *
 * \param[in]	double*				x1a
 *				double*				x2a
 *				const unsigned int	width
 *				const unsigned int	height
 *				complex<double>*	ya
 *				conste double		x1
 *				conste double		x2
 *				complex<double>&	y
 * 
 * \return propErr
 * \sa 
 * \remarks 
 * \author Mauch
 */
interpErr Interpolator::doInterpolation(double *x1a, double *x2a, const unsigned int width, const unsigned int height, complex<double> *ya, const double x1, const double x2, complex<double> *y)
{
	if (this->y2ca_ptr == NULL)
	{
		cout << "error in Interpolator.doInterpolation(): array containing second derivative data is not initialized. Try calling initInterpolation... " << endl;
		return INTERP_ERR;	
	}
	else
		this->splin2(x1a, x2a, width, height, ya, this->y2ca_ptr, x1, x2, y);

	return INTERP_NO_ERR;
};

/**
 * \detail doInterpolation 
 *
 * this routine basically wraps the private function splint for complex data
 *
 * \param[in]	double*				xa
 *				complex<double>*	ya
 *				const unsigned int	width
 *				conste double		x
 *				complex<double>&	y
 * 
 * \return propErr
 * \sa 
 * \remarks 
 * \author Mauch
 */
interpErr Interpolator::doInterpolation(double *xa, complex<double> *ya, const unsigned int width, const double x,  complex<double> *y)
{
	if (this->y2c_ptr == NULL)
	{
		cout << "error in Interpolator.doInterpolation(): array containing second derivative data is not initialized. Try calling initInterpolation... " << endl;
		return INTERP_ERR;	
	}
	else
		splint(xa, ya, this->y2c_ptr, width, x, y);
	return INTERP_NO_ERR;
};
