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

/**\file GaussBeamRayField.h
* \brief 
* 
*           
* \author Mauch
*/

#ifndef GAUSSBEAMRAYFIELD_H
  #define GAUSSBEAMRAYFIELD_H

#include <optix.h>
#include "rayData.h"
#include "stdlib.h"
#include "RayField.h"
#include "inputOutput.h"

/* declare class */
/**
  *\class   GaussBeamRayField
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
class GaussBeamRayField : public RayField
{
  protected:
	gaussBeamRayStruct* rayList;
	rayFieldParams *rayParamsPtr;

  public:
    /* standard constructor */
    GaussBeamRayField()
	{
		rayList = NULL;
		rayListLength=0;
	}
    /* Konstruktor */
    GaussBeamRayField(unsigned long long length)
	{
	  rayList = (gaussBeamRayStruct*) malloc(length*sizeof(gaussBeamRayStruct));
	  rayListLength = length;
	}
	/* Destruktor */
	~GaussBeamRayField()
	{
	  delete rayList;
	  rayList = NULL;
	}

	fieldError setRay(gaussBeamRayStruct ray, unsigned long long index);
	gaussBeamRayStruct* getRay(unsigned long long index);
	unsigned long long getRayListLength(void);
	gaussBeamRayStruct* getRayList(void);

	void createCPUSimInstance(double lambda);

    fieldError createOptixInstance(RTcontext &context, SimParams simParams, double lambda);

	fieldError traceScene(Group &oGroup);

	fieldError writeData2File(FILE *hFile_pos, rayDataOutParams outParams);

	fieldError convert2Intensity(IntensityField* imagePtr);
	fieldError convert2ScalarField(ScalarLightField* imagePtr);
	fieldError convert2VecField(VectorLightField* imagePtr);

};

#endif

