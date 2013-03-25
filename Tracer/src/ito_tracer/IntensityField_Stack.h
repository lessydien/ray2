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

#ifndef INTENSITYFIELDSTACK_H
  #define INTENSITYFIELDSTACK_H

#include <optix.h>
#include "stdlib.h"
//#include "complex.h"
//#include "optix_math.h"
#include "my_vector_types.h"
#include "Field.h"
#include "Field_Stack.h"
#include "IntensityField.h"

/* declare class */
/**
  *\class   IntensityFieldStack
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
class IntensityFieldStack :  public FieldStack
{
  private:
	IntensityField** IntFieldList;
	fieldStackParams* paramsPtr;

  public:
    /* standard constructor */
    IntensityFieldStack()
	{
		this->paramsPtr=new fieldStackParams();
		this->IntFieldList=NULL;
	}
	/* constructor */
	IntensityFieldStack(fieldStackParams *paramsPtrIn)
	{
		this->paramsPtr=paramsPtrIn;
		this->IntFieldList=new IntensityField*[paramsPtrIn->runningParamLength];
	}
	/* Destructor */
	~IntensityFieldStack()
	{
		if (IntFieldList != NULL)
		{
			for (int i=0;i<paramsPtr->runningParamLength;i++)
			{
				delete IntFieldList[i];
			}
		}
		delete paramsPtr;
	}

	fieldStackError addField(IntensityField *fieldPtr, double runningParamIn, long long index);
	IntensityField* getField(long long index);
	virtual fieldError write2TextFile(FILE* hFile, detParams &oDetParams);

	//double getPix(ulong2 pixel);
	//fieldError setPix(ulong2 pixel, double value);
	//fieldParams* getParamsPtr();
	//double* getIntensityPtr();
};

//inline double IntensityField::getPix(ulong2 pixel)
//{
//	if ( (pixel.x<this->paramsPtr->nrPixels.x) && (pixel.y<=this->paramsPtr->nrPixels.y) )
//	{
//		return Iptr[pixel.x+pixel.y*paramsPtr->nrPixels.x];
//	}
//	return Iptr[0]; // we need some error handling here !!
//}
//
//inline fieldError IntensityField::setPix(ulong2 pixel, double value)
//{
//	if ( (pixel.x<this->paramsPtr->nrPixels.x) && (pixel.y<=this->paramsPtr->nrPixels.y) )
//	{
//		this->Iptr[pixel.x+pixel.y*paramsPtr->nrPixels.x]=value;
//		return FIELD_NO_ERR;
//	}
//	return FIELD_INDEXOUTOFRANGE_ERR;
//}

#endif

