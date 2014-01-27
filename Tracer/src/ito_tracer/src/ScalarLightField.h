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

/**\file ScalarLightField.h
* \brief 
* 
*           
* \author Mauch
*/

#ifndef SCALARLIGHTFIELD_H
  #define SCALARLIGHTFIELD_H

#include <optix.h>
#include "stdlib.h"
//#include "complex.h"
#include <complex>
#include "Field.h"
//#include "macrosim_types.h"
#include <optix_math.h>
#include "PropagationMath.h"
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
class scalarFieldParams : public fieldParams
{
public:
	double amplMax;
};

/* declare class */
/**
  *\class   ScalarLightField
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
class ScalarLightField: public Field
{
  protected:
    complex<double>* U;
	scalarFieldParams* paramsPtr;
	virtual fieldError write2TextFile(char* filename, detParams &oDetParams);
	virtual fieldError write2MatFile(char* filename, detParams &oDetParams);


  public:
    /* standard constructor */
    ScalarLightField()
	{
		U = NULL;
		this->paramsPtr=new scalarFieldParams();
		////paramsPtr->MTransform=make_double4x4(
		//paramsPtr->nrPixels=make_long3(0,0,0);
		//paramsPtr->scale=make_double3(0,0,0);
		//paramsPtr->units.x=metric_au;
		//paramsPtr->units.y=metric_au;
		//paramsPtr->units.z=metric_au;
		//paramsPtr->lambda=0;
		//paramsPtr->unitLambda=metric_au;
		//paramsPtr->amplMax=0;
	}
    /* Konstruktor */
    ScalarLightField(scalarFieldParams paramsIn)
	{
	  // allocate memory and initialize it to zero		
	    this->paramsPtr = new scalarFieldParams();
	    *(this->paramsPtr)=paramsIn;
		fftw_complex *in=(fftw_complex*) fftw_malloc(sizeof(fftw_complex) * paramsPtr->nrPixels.x*paramsPtr->nrPixels.y*paramsPtr->nrPixels.z);
		U=reinterpret_cast<complex<double>*>(in);
		// init to zero
		for (unsigned long jx=0;jx<paramsPtr->nrPixels.x;jx++)
		{
			for (unsigned long jy=0;jy<paramsPtr->nrPixels.y;jy++)
			{
				for (unsigned long jz=0;jz<paramsPtr->nrPixels.z;jz++)
				{
					U[jx+jy*paramsPtr->nrPixels.x+jz*paramsPtr->nrPixels.y]=0;
				}
			}
		}
		//U = (complex<double>*) calloc(paramsPtr->nrPixels.x*paramsPtr->nrPixels.y*paramsPtr->nrPixels.z, sizeof(complex<double>));
	}
	/* Destruktor */
	~ScalarLightField()
	{
	  if (this->U != NULL)
	  {
		fftw_free(U);
		U = NULL;
	  }
	  if (this->paramsPtr != NULL)
	  {
		delete this->paramsPtr;
		this->paramsPtr=NULL;
	  }
	}

	complex<double> getPix(ulong2 pixel);
	fieldError setPix(ulong2 pixel, complex<double> value);
	complex<double>* getFieldPtr();
	virtual fieldParams* getParamsPtr();
	fieldError parseXml(pugi::xml_node &field, vector<Field*> &fieldVec, SimParams simParams);

	fieldError convert2ScalarField(Field* imagePtr, detParams &oDetParams);
	fieldError convert2Intensity(Field* imagePtr, detParams &oDetParams);

	fieldError  doSim(Group &oGroup, simAssParams &params, bool &simDone);

	virtual fieldError initGPUSubset(RTcontext &context);
	virtual fieldError initCPUSubset();

	virtual fieldError createCPUSimInstance();
	virtual fieldError createOptixInstance(RTcontext &context, RTbuffer &output_buffer_obj, RTbuffer &seed_buffer_obj);

	virtual fieldError traceScene(Group &oGroup, bool RunOnCPU);

};

inline complex<double> ScalarLightField::getPix(ulong2 pixel)
{
	if ( (pixel.x<this->paramsPtr->nrPixels.x) && (pixel.y<=this->paramsPtr->nrPixels.y) )
	{
		return U[pixel.x+pixel.y*paramsPtr->nrPixels.x];
	}
	return U[0]; // we need some error handling here !!
}

inline fieldError ScalarLightField::setPix(ulong2 pixel, complex<double> value)
{
	if ( (pixel.x<this->paramsPtr->nrPixels.x) && (pixel.y<=this->paramsPtr->nrPixels.y) )
	{
		this->U[pixel.x+pixel.y*paramsPtr->nrPixels.x]=value;
		return FIELD_NO_ERR;
	}
	return FIELD_INDEXOUTOFRANGE_ERR;
}

#endif

