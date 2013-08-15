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

/**\file Pupil_RayAim_FarField.h
* \brief header file that contains the detectorParams class
* 
*           
* \author Mauch
*/

#ifndef PUP_RAYAIM_FARFIELD_H
  #define PUP_RAYAIM_FARFIELD_H

#include "Pupil.h"
#include "Pupil_RayAim_FarField_aim.h"

#define PATH_TO_AIM_RAYAIM_FARFIELD "ITO-MacroSim_generated_aimRayAimFarField"

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
class pupilParams_RayAim : public pupilParams
{
public:
	double f_eff;
};

/* declare class */
/**
  *\class   Pupil_RayAim_FarField
  *\ingroup Pupil
  *\brief   
  *
  *         
  *
  *         \todo
  *         \remarks           
  *         \sa       NA
  *         \date     07.12.2011
  *         \author  Mauch
  *
  */
class Pupil_RayAim_FarField: public Pupil
{
	protected:
		pupilParams_RayAim *fullParamsPtr;
		Pupil_RayAim_FarField_RedParams *redParamsPtr;

		PupilError reduceParams(void);

  public:
    /* standard constructor */
    Pupil_RayAim_FarField()
	{
		/* set ptx path for OptiX calculations */
		//path_to_ptx[512];
		sprintf( path_to_ptx, "%s" PATH_SEPARATOR "%s", PATH_TO_PTX, PATH_TO_AIM_RAYAIM_FARFIELD );
		//this->setPathToPtx(path_to_ptx);
	}
	/* Destruktor */
    ~Pupil_RayAim_FarField()
	{
		//delete path_to_ptx;
	}
//    PupilError createOptiXInstance(RTcontext context, RTgeometryinstance &instance, int index, simMode mode, double lambda);
	PupilError updateCPUSimInstance(double lambda);
	PupilError createCPUSimInstance(double lambda);
	void setPathToPtx(char* path);
	char* getPathToPtx(void);
	virtual void setFullParamsPtr(pupilParams *params);
	virtual pupilParams* getFullParamsPtr(void);

	void aim(rayStruct *ray, unsigned long long iX, unsigned long long iY);
	PupilError processParseResults(PupilParseParamStruct &parseResults_Pupil);
};
#endif
