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

/**\file Pupil.h
* \brief header file that contains the detectorParams class
* 
*           
* \author Mauch
*/

/**
 *\defgroup Pupil
 */

#ifndef PUPIL_H
  #define PUPIL_H

#include <optix.h>
#include <optix_math.h>
#include "Pupil_aim.h"
#include "rayData.h"
#include "PupilParams.h"
#include "stdlib.h"
#include "sampleConfig.h"
#include "stdio.h"

typedef enum 
{
  PUP_ERR,
  PUP_NO_ERR
} PupilError;

/* declare class */
/**
  *\class   pupilParams_RayAim_FarField
  *\ingroup Pupil
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
class pupilParams
{
public:
  double2 apertureHalfWidth;
  double3 root;
  long long pupilID;
  double3 tilt;
  ApertureType apertureType;
  pupilType type;
};

/* declare class */
/**
  *\class   Material
  *\ingroup Material
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
class Pupil
{
  protected:
    char* path_to_ptx;
	pupilParams *fullParamsPtr;
	Pupil_RedParams *redParamsPtr;
	/* OptiX variables */


  public:

    Pupil()
	{
		path_to_ptx=(char*)malloc(512*sizeof(char));
	}
	virtual ~Pupil()
	{
		delete path_to_ptx;
	}
	void setPathToPtx(char* path);
	char* getPathToPtx(void);
	PupilError createPupilAimProgramPtx(RTcontext context, TraceMode mode);

//    virtual PupilError createOptiXInstance(RTcontext context, RTgeometryinstance &instance, int index, SimParams simParams, double lambda);
	virtual PupilError createCPUSimInstance(double lambda);
	virtual PupilError updateCPUSimInstance(double lambda);
//    virtual PupilError updateOptiXInstance(RTcontext context, RTgeometryinstance &instance, int index, SimParams simParams, double lambda);
	virtual void aim(rayStruct &ray, unsigned long long iX, unsigned long long iY);
	virtual void setFullParamsPtr(pupilParams *params);
	virtual pupilParams* getFullParamsPtr(void);
	PupilError reduceParams(double lambda);

	virtual PupilError processParseResults(PupilParseParamStruct &parseResults_Pupil);
};

#endif
