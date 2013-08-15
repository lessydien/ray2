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

/**\file PhaseSpaceField.h
* \brief 
* 
*           
* \author Mauch
*/

#ifndef PHASESPACEFIELD_H
  #define PHASESPACEFIELD_H

#include <optix.h>
#include "stdlib.h"
//#include "complex.h"
#include <complex>
#include "my_vector_types.h"
#include "Field.h"
//#include "PropagationMath.h"
using namespace std;

/* declare class */
/**
  *\class   detPhaseSpaceParams
  *\ingroup Detector
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
class phaseSpaceParams: public fieldParams
{
public:
	ulong2 nrPixels_PhaseSpace;
	double2 scale_dir;
	double2 dirHalfWidth;
};

/* declare class */
/**
  *\class   PhaseSpaceField
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
class PhaseSpaceField : public Field
{
  protected:
	double* PSptr; //!> field for PhaseSpace.
//	complex<double> *Uptr; //!> field for complex amplitude. Here we have to temporarily save the complex amplitude if we have to coherently sum several GPU-subsets 
	phaseSpaceParams* paramsPtr;
	virtual fieldError write2TextFile(char* filename, detParams &oDetParams);
	virtual fieldError write2MatFile(char* filename, detParams &oDetParams);

  public:
    /* standard constructor */
    PhaseSpaceField()
	{
		PSptr=NULL;
//		Uptr = NULL;
		paramsPtr->nrPixels_PhaseSpace=make_ulong2(0,0);
		paramsPtr->nrPixels=make_long3(0,0,0);
		paramsPtr->scale=make_double3(0,0,0);
		paramsPtr->units.x=metric_au;
		paramsPtr->units.y=metric_au;
		paramsPtr->units.z=metric_au;
		paramsPtr->lambda=0;
		paramsPtr->unitLambda=metric_au;

	}
	/* constructor */
	PhaseSpaceField(phaseSpaceParams paramsIn)
	{
		this->paramsPtr=new phaseSpaceParams();
		*(this->paramsPtr)=paramsIn;
		this->PSptr=(double*)calloc(paramsPtr->nrPixels.x*paramsPtr->nrPixels.y*paramsPtr->nrPixels.z*paramsPtr->nrPixels_PhaseSpace.x*paramsPtr->nrPixels_PhaseSpace.y,sizeof(double));
		//this->Uptr=(complex<double>*)calloc(paramsPtr->nrPixels.x*paramsPtr->nrPixels.y*paramsPtr->nrPixels.z,sizeof(complex<double>));
	}
	/* Destruktor */
	~PhaseSpaceField()
	{
		if (this->PSptr != NULL)
		{
			delete this->PSptr;
			this->PSptr=NULL;
		}
//		delete this->Uptr;
//		this->Uptr=NULL;
		if (this->paramsPtr != NULL)
		{
			delete this->paramsPtr;
			this->paramsPtr=NULL;
		}
	}
	double getPix(ulong2 pixel, ulong2 pixel_PhaseSpace);
	fieldError setPix(ulong2 pixel, ulong2 pixel_PhaseSpace, double value);
	phaseSpaceParams* getParamsPtr();
	double* getPhaseSpacePtr();
	//complex<double>* getComplexAmplPtr();
};

inline double PhaseSpaceField::getPix(ulong2 pixel, ulong2 pixel_PhaseSpace)
{
	if ( (pixel.x<this->paramsPtr->nrPixels.x) && (pixel.y<=this->paramsPtr->nrPixels.y) )
	{
		return PSptr[pixel.x+pixel.y*paramsPtr->nrPixels.x+pixel_PhaseSpace.x*paramsPtr->nrPixels.x*paramsPtr->nrPixels.y+pixel_PhaseSpace.y*paramsPtr->nrPixels_PhaseSpace.x*paramsPtr->nrPixels.x*paramsPtr->nrPixels.y];
	}
	return PSptr[0]; // we need some error handling here !!
}

inline fieldError PhaseSpaceField::setPix(ulong2 pixel, ulong2 pixel_PhaseSpace, double value)
{
	if ( (pixel.x<this->paramsPtr->nrPixels.x) && (pixel.y<=this->paramsPtr->nrPixels.y) && (pixel_PhaseSpace.x<=this->paramsPtr->nrPixels_PhaseSpace.x) && (pixel_PhaseSpace.y<=this->paramsPtr->nrPixels_PhaseSpace.y) )
	{
		this->PSptr[pixel.x+pixel.y*paramsPtr->nrPixels.x+pixel_PhaseSpace.x*paramsPtr->nrPixels.x*paramsPtr->nrPixels.y+pixel_PhaseSpace.y*paramsPtr->nrPixels_PhaseSpace.x*paramsPtr->nrPixels.x*paramsPtr->nrPixels.y]=value;
		return FIELD_NO_ERR;
	}
	return FIELD_INDEXOUTOFRANGE_ERR;
}

#endif

