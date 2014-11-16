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

/**\file ScalarGaussianField.h
* \brief 
* 
*           
* \author Mauch
*/

#ifndef SCALARGAUSSIANFIELD_H
  #define SCALARGAUSSIANFIELD_H

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
class gaussianFieldParams : public scalarFieldParams
{
public:
	double2 focusWidth;
	double2 distToFocus;
};

/* declare class */
/**
  *\class   ScalarGaussianField
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
class ScalarGaussianField: public ScalarLightField
{
  protected:
	gaussianFieldParams* paramsPtr;


  public:
    /* standard constructor */
    ScalarGaussianField() :
		ScalarLightField()
	{
		paramsPtr->nrPixels=make_long3(0,0,0);
		paramsPtr->scale=make_double3(0,0,0);
		paramsPtr->units.x=metric_au;
		paramsPtr->units.y=metric_au;
		paramsPtr->units.z=metric_au;
		paramsPtr->lambda=0;
		paramsPtr->unitLambda=metric_au;
		paramsPtr->distToFocus=make_double2(0,0);
		paramsPtr->focusWidth=make_double2(0,0);

	}
    /* Konstruktor */
    ScalarGaussianField(gaussianFieldParams paramsIn) :
		ScalarLightField(paramsIn)
	{
	  // allocate memory and initialize it to zero		
	    this->paramsPtr = new gaussianFieldParams();
	    *(this->paramsPtr)=paramsIn;
		//double *in=(double*) malloc(sizeof(double) * paramsPtr->nrPixels.x*paramsPtr->nrPixels.y*paramsPtr->nrPixels.z);
		//U=reinterpret_cast<complex<double>*>(in);
		//// init to zero
		//for (unsigned long jx=0;jx<paramsPtr->nrPixels.x;jx++)
		//{
		//	for (unsigned long jy=0;jy<paramsPtr->nrPixels.y;jy++)
		//	{
		//		for (unsigned long jz=0;jz<paramsPtr->nrPixels.z;jz++)
		//		{
		//			U[jx+jy*paramsPtr->nrPixels.x+jz*paramsPtr->nrPixels.y]=0;
		//		}
		//	}
		//}
		//U = (complex<double>*) calloc(paramsPtr->nrPixels.x*paramsPtr->nrPixels.y*paramsPtr->nrPixels.z, sizeof(complex<double>));
	}
	/* Destruktor */
	~ScalarGaussianField()
	{
	  if (this->paramsPtr != NULL)
	  {
		delete paramsPtr;
		paramsPtr=NULL;
	  }
	}

	fieldError parseXml(pugi::xml_node &field, vector<Field*> &fieldVec, SimParams simParams);
};

#endif

