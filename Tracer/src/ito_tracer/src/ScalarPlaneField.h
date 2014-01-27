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

/**\file ScalarPlaneField.h
* \brief 
* 
*           
* \author Mauch
*/

#ifndef SCALARPLANEFIELD_H
  #define SCALARPLANEFIELD_H

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
  *\class   planeFieldParams
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
class planeFieldParams : public scalarFieldParams
{
public:
	double2 fieldWidth;
};

/* declare class */
/**
  *\class   ScalarPlaneField
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
class ScalarPlaneField: public ScalarLightField
{
  protected:
	planeFieldParams* paramsPtr;

  public:
    /* standard constructor */
    ScalarPlaneField() :
		ScalarLightField()
	{
		//ScalarLightField::U = NULL;
		//if (ScalarLightField::paramsPtr != NULL)
		//	delete ScalarLightField::paramsPtr;
		paramsPtr=new planeFieldParams();
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
    ScalarPlaneField(planeFieldParams paramsIn) :
		ScalarLightField(paramsIn)
	{
	  // delete params of base class	
//		if (ScalarLightField::paramsPtr != NULL)
//			delete ScalarLightField::paramsPtr;
	    this->paramsPtr = new planeFieldParams();
	    *(this->paramsPtr)=paramsIn;
		//fftw_complex *in=(fftw_complex*) fftw_malloc(sizeof(fftw_complex) * paramsPtr->nrPixels.x*paramsPtr->nrPixels.y*paramsPtr->nrPixels.z);
		//ScalarLightField::U=reinterpret_cast<complex<double>*>(in);
		//double x=-(paramsPtr->nrPixels.x-1)/2*paramsPtr->scale.x;
		//double y=-(paramsPtr->nrPixels.y-1)/2*paramsPtr->scale.y;
		//double z=-(paramsPtr->nrPixels.z-1)/2*paramsPtr->scale.z;
		//// init
		//for (unsigned long jx=0;jx<paramsPtr->nrPixels.x;jx++)
		//{
		//	x=x+paramsPtr->scale.x;
		//	for (unsigned long jy=0;jy<paramsPtr->nrPixels.y;jy++)
		//	{
		//		y=y+paramsPtr->scale.y;
		//		for (unsigned long jz=0;jz<paramsPtr->nrPixels.z;jz++)
		//		{
		//			if (abs(x)<=paramsPtr->fieldWidth.x)
		//				ScalarLightField::U[jx+jy*paramsPtr->nrPixels.x+jz*paramsPtr->nrPixels.y]=paramsPtr->amplMax;
		//			else
		//				ScalarLightField::U[jx+jy*paramsPtr->nrPixels.x+jz*paramsPtr->nrPixels.y]=0;
		//		}
		//	}
		//}
		//U = (complex<double>*) calloc(paramsPtr->nrPixels.x*paramsPtr->nrPixels.y*paramsPtr->nrPixels.z, sizeof(complex<double>));
	}
	/* Destruktor */
	~ScalarPlaneField()
	{
//	  if (ScalarLightField::U != NULL)
//	  {
//		delete ScalarLightField::U;
//		ScalarLightField::U = NULL;
//	  }
	 // if (paramsPtr != NULL)
	 // {
		//delete paramsPtr;
		//this->paramsPtr=NULL;
	 // }
	}

	virtual fieldError initSimulation(Group &oGroup, simAssParams &params);

	fieldError initGPUSubset(RTcontext &context);
	fieldError initCPUSubset();
	fieldParams* getParamsPtr();

	fieldError createCPUSimInstance();
	fieldError createOptixInstance(RTcontext &context, RTbuffer &output_buffer_obj, RTbuffer &seed_buffer_obj);

	fieldError parseXml(pugi::xml_node &field, vector<Field*> &fieldVec, SimParams simParams);
};

#endif

