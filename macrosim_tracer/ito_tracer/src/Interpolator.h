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

/**\file Interpolator.h
* \brief 
* 
*           
* \author Mauch
*/
#ifndef INTERPOLATOR_H
  #define INTERPOLATOR_H

#include "Interpolator_host_device.h"

#include <complex>
#include <iostream>
using namespace std;

typedef enum 
{
  INTERP_NO_ERR,
  INTERP_ERR
} interpErr;

/* declare class */
/**
  *\class   Interpolator
  *\ingroup Interpolator
  *\brief   
  *
  *         
  *
  *         \todo
  *         \remarks           
  *         \sa       NA
  *         \date     11.10.2011
  *         \author  Mauch
  *
  */


class Interpolator
{
private:

	interpErr spline(double *x, double *y, const unsigned int width, const double yp1,  const double ypn, double *y2);
	interpErr splint(double *xa, double *ya, double *y2a, const unsigned int width, const double x,  double *y);
	//interpErr splint(double *xa, double *ya, double *y2a, const unsigned int width, const double x);
	interpErr splie2(double *x1a, double *x2a, double *ya, double *y2a, const unsigned int width, const unsigned int height);
	interpErr splin2(double *x1a, double *x2a, const unsigned int width, const unsigned int height, double *ya, double *y2a, const double x1, const double x2, double *y);

	// same functions for complex data
	interpErr spline(double *x, complex<double> *y, const unsigned int width, const double yp1,  const double ypn, complex<double> *y2);
	interpErr splint(double *xa, complex<double> *ya, complex<double> *y2a, const unsigned int width, const double x,  complex<double> *y);
	//interpErr splint(double *xa, complex<double> *ya, complex<double> *y2a, const unsigned int width, const double x);
	interpErr splie2(double *x1a, double *x2a, complex<double> *ya, complex<double> *y2a, const unsigned int width, const unsigned int height);
	interpErr splin2(double *x1a, double *x2a, const unsigned int width, const unsigned int height, complex<double> *ya, complex<double> *y2a, const double x1, const double x2, complex<double> *y);


public:
	double* y2_ptr;				//!> pointer for the 1D second derivative
	complex<double>* y2c_ptr;	//!> pointer for the complex 1D second derivative
	double* y2a_ptr;			//!> pointer for the 2D second derivative
	complex<double>* y2ca_ptr;   //!> pointer for the complex 2D second derivative

    /* standard constructor */
    Interpolator()
	{
		y2_ptr=NULL;
		y2c_ptr=NULL;
		y2a_ptr=NULL;
		y2ca_ptr=NULL;
	}
	/* Destruktor */
	~Interpolator()
	{
	  if ( y2_ptr != NULL)
	  {
		  delete y2_ptr;
		  y2_ptr = NULL;
	  }
  	  if ( y2a_ptr != NULL)
	  {
		  delete y2a_ptr;
		  y2a_ptr = NULL;
	  }
	  if ( y2c_ptr != NULL)
	  {
		  delete y2c_ptr;
		  y2c_ptr = NULL;
	  }
	  if ( y2ca_ptr != NULL)
	  {
		  delete y2ca_ptr;
		  y2ca_ptr = NULL;
	  }
	}

	// init 2D interpolation of double data
	interpErr initInterpolation(double *x1a, double *x2a, double *ya,const unsigned int width, const unsigned int height);
	// init 1D interpolation of double data
	interpErr initInterpolation(double *x, double *y, const unsigned int width, const double yp1,  const double ypn);
	// init 2D interpolation of complex data
	interpErr initInterpolation(double *x1a, double *x2a, complex<double> *ya, const unsigned int width, const unsigned int height);
	// init 1D interpolation of complex data
	interpErr initInterpolation(double *x, complex<double> *y, const unsigned int width, const double yp1,  const double ypn);

	// do 2D interpolation of double data
	interpErr doInterpolation(double *x1a, double *x2a, const unsigned int width, const unsigned int height, double *ya, const double x1, const double x2, double *y);
	// do 1D interpolation of double data
	interpErr doInterpolation(double *xa, double *ya, const unsigned int width, const double x,  double *y);
	// do 2D interpolation of complex data
	interpErr doInterpolation(double *x1a, double *x2a, const unsigned int width, const unsigned int height, complex<double> *ya, const double x1, const double x2, complex<double> *y);
	// do 1D interpolation of complex data
	interpErr doInterpolation(double *xa, complex<double> *ya, const unsigned int width, const double x,  complex<double> *y);

	interpErr nearestNeighbour(const double& x1_0, const double& x2_0, const double& delta_x1, const double& delta_x2, const double *yin_ptr, const unsigned int dim_x1, const unsigned int dim_x2, const double &x1_out, const double &x2_out, double *yout_ptr);


};




#endif