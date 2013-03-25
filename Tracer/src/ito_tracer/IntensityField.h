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

/**\file IntensityField.h
* \brief 
* 
*           
* \author Mauch
*/

#ifndef INTENSITYFIELD_H
  #define INTENSITYFIELD_H

#include <optix.h>
#include "stdlib.h"
//#include "complex.h"
#include <complex>
#include "my_vector_types.h"
#include "Field.h"
#include "pugixml.hpp"
//#include "PropagationMath.h"
using namespace std;

/* declare class */
/**
  *\class   IntensityField
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
class IntensityField : public Field
{
  protected:
	double* Iptr; //!> field for Intensity.
	complex<double> *Uptr; //!> field for complex amplitude. Here we have to temporarily save the complex amplitude if we have to coherently sum several GPU-subsets 
	fieldParams* paramsPtr;
	virtual fieldError write2TextFile(char* filename, detParams &oDetParams);
	virtual fieldError write2MatFile(char* filename, detParams &oDetParams);

  public:
    /* standard constructor */
    IntensityField()
	{
		Iptr=NULL;
		Uptr = NULL;
		this->paramsPtr=NULL;
		////paramsPtr->MTransform=make_double4x4(
		//paramsPtr->nrPixels=make_long3(0,0,0);
		//paramsPtr->scale=make_double3(0,0,0);
		//paramsPtr->units.x=metric_au;
		//paramsPtr->units.y=metric_au;
		//paramsPtr->units.z=metric_au;
		//paramsPtr->lambda=0;
		//paramsPtr->unitLambda=metric_au;

	}
	/* constructor */
	IntensityField(fieldParams paramsIn)
	{
		this->paramsPtr=new fieldParams();
		*(this->paramsPtr)=paramsIn;
		this->Iptr=(double*)calloc(paramsPtr->nrPixels.x*paramsPtr->nrPixels.y*paramsPtr->nrPixels.z,sizeof(double));
		this->Uptr=(complex<double>*)calloc(paramsPtr->nrPixels.x*paramsPtr->nrPixels.y*paramsPtr->nrPixels.z,sizeof(complex<double>));
	}
	/* Destruktor */
	~IntensityField()
	{
		if (this->Iptr != NULL)
		{
			delete this->Iptr;
			this->Iptr=NULL;
		}
		if (this->Uptr != NULL)
		{
			delete this->Uptr;
			this->Uptr=NULL;
		}
		if (this->paramsPtr != NULL)
		{
			delete this->paramsPtr;
			this->paramsPtr=NULL;
		}
	}
	double getPix(ulong2 pixel);
	fieldError setPix(ulong2 pixel, double value);
	fieldParams* getParamsPtr();
	double* getIntensityPtr();
	complex<double>* getComplexAmplPtr();
	virtual fieldError convert2ItomObject(void** dataPtrPtr, ItomFieldParams* paramsOut);
	virtual fieldError parseXml(pugi::xml_node &field, vector<Field*> &fieldVec);
};

inline double IntensityField::getPix(ulong2 pixel)
{
	if ( (pixel.x<this->paramsPtr->nrPixels.x) && (pixel.y<=this->paramsPtr->nrPixels.y) )
	{
		return Iptr[pixel.x+pixel.y*paramsPtr->nrPixels.x];
	}
	return Iptr[0]; // we need some error handling here !!
}

inline fieldError IntensityField::setPix(ulong2 pixel, double value)
{
	if ( (pixel.x<this->paramsPtr->nrPixels.x) && (pixel.y<=this->paramsPtr->nrPixels.y) )
	{
		this->Iptr[pixel.x+pixel.y*paramsPtr->nrPixels.x]=value;
		return FIELD_NO_ERR;
	}
	return FIELD_INDEXOUTOFRANGE_ERR;
}

#endif

