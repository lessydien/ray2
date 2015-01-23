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

/**\file ScalarSphericalField.h
* \brief 
* 
*           
* \author Mauch
*/

#ifndef SCALARSPHERICALFIELD_H
  #define SCALARSPHERICALFIELD_H

#include <optix.h>
#include "stdlib.h"
//#include "complex.h"
#include <complex>
#include "ScalarLightField.h"
//#include "macrosim_types.h"
#include <optix_math.h>
#include "pugixml.hpp"

/* declare class */
/**
  *\class   gaussianFieldParams
  *\ingroup Field
  *\brief   
  *
  *         
  *
  *         \todo
  *         \remarks           
  *         \sa       NA
  *         \date     04.01.2011
  *         \author  Mauch
  *
  */
class sphericalFieldParams : public scalarFieldParams
{
public:
	double2 radius;
	double2 numApt;
};

/* declare class */
/**
  *\class   ScalarSphericalField
  *\ingroup Field
  *\brief   
  *
  *         
  *
  *         \todo
  *         \remarks           
  *         \sa       NA
  *         \date     04.01.2011
  *         \author  Mauch
  *
  */
class ScalarSphericalField: public ScalarLightField
{
  protected:
	sphericalFieldParams* paramsPtr;

  public:
    /* standard constructor */
    ScalarSphericalField() :
		ScalarLightField()
	{
		ScalarLightField::U = NULL;
		//paramsPtr->MTransform=make_double4x4(
		paramsPtr->nrPixels=make_long3(0,0,0);
		paramsPtr->scale=make_double3(0,0,0);
		paramsPtr->units.x=metric_au;
		paramsPtr->units.y=metric_au;
		paramsPtr->units.z=metric_au;
		paramsPtr->lambda=0;
		paramsPtr->unitLambda=metric_au;
	}
    /* Konstruktor */
    ScalarSphericalField(sphericalFieldParams paramsIn) :
		ScalarLightField(paramsIn)
	{
	  // allocate memory and initialize it to zero		
	    this->paramsPtr = new sphericalFieldParams();
	    *(this->paramsPtr)=paramsIn;
		//fftw_complex *in=(fftw_complex*) fftw_malloc(sizeof(fftw_complex) * paramsPtr->nrPixels.x*paramsPtr->nrPixels.y*paramsPtr->nrPixels.z);
		//U=reinterpret_cast<complex<double>*>(in);
		//// init to zero
		//for (unsigned long jx=0;jx<paramsPtr->nrPixels.x;jx++)
		//{
		//	for (unsigned long jy=0;jy<paramsPtr->nrPixels.y;jy++)
		//	{
		//		for (unsigned long jz=0;jz<paramsPtr->nrPixels.z;jz++)
		//		{
		//			ScalarLightField::U[jx+jy*paramsPtr->nrPixels.x+jz*paramsPtr->nrPixels.y]=0;
		//		}
		//	}
		//}
		//U = (complex<double>*) calloc(paramsPtr->nrPixels.x*paramsPtr->nrPixels.y*paramsPtr->nrPixels.z, sizeof(complex<double>));
	}
	/* Destruktor */
	~ScalarSphericalField()
	{
	  if (paramsPtr != NULL)
	  {
		delete paramsPtr;
		paramsPtr=NULL;
	  }
	}

	fieldError parseXml(pugi::xml_node &field, vector<Field*> &fieldVec, SimParams simParams);
};

#endif

