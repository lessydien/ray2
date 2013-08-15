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

/**\file VectorLightField.h
* \brief 
* 
*           
* \author Mauch
*/

#ifndef LIGHTFIELD_H
  #define LIGHTFIELD_H

#include <optix.h>
#include "stdlib.h"
//#include "complex.h"
//#include "my_vector_types.h"
#include "rayData.h"
#include "Field.h"
#include <optix_math.h>
#include "pugixml.hpp"

typedef struct
{
	double3 amplitude;
	double phase;
} vecLight;

/* declare class */
/**
  *\class   VectorLightField
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
class VectorLightField: Field
{
  protected:
	vecLight* E;
	fieldParams *paramsPtr;
	virtual fieldError write2TextFile(FILE* hFile, detParams &oDetParams);
	virtual fieldError write2MatFile(char* filename, detParams &oDetParams);

  public:
    /* standard constructor */
    VectorLightField()
	{
		E = NULL;
		//paramsPtr->offsetX=0;
		//paramsPtr->offsetY=0;
		//paramsPtr->scaleX=0;
		//paramsPtr->scaleY=0;
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
    VectorLightField(fieldParams* paramsPtrIn)
	{
	  // allocate memory and initialize it to zero
		E = (vecLight*) calloc(paramsPtr->nrPixels.x*paramsPtr->nrPixels.y*paramsPtr->nrPixels.z, sizeof(vecLight));
		this->paramsPtr = paramsPtrIn;
	}
	/* Destruktor */
	~VectorLightField()
	{
	  delete E;
	  E = NULL;
	}
	vecLight getPix(ulong2 pixel);
	fieldError setPix(ulong2 pixel, vecLight value);

	fieldError parseXml(pugi::xml_node &field, vector<Field*> &fieldVec);
};

inline vecLight VectorLightField::getPix(ulong2 pixel)
{
	if ( (pixel.x<this->paramsPtr->nrPixels.x) && (pixel.y<=this->paramsPtr->nrPixels.y) )
	{
		return E[pixel.x+pixel.y*paramsPtr->nrPixels.x];
	}
	return E[0]; // we need some error handling here !!
}

inline fieldError VectorLightField::setPix(ulong2 pixel, vecLight value)
{
	if ( (pixel.x<this->paramsPtr->nrPixels.x) && (pixel.y<=this->paramsPtr->nrPixels.y) )
	{
		this->E[pixel.x+pixel.y*paramsPtr->nrPixels.x]=value;
		return FIELD_NO_ERR;
	}
	return FIELD_INDEXOUTOFRANGE_ERR;
}

#endif

