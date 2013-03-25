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

/**\file MaterialReflecting_DiffRays.h
* \brief 
* 
*           
* \author Mauch
*/

#ifndef MATERIALREFLECTING_DIFFRAYS_H
#define MATERIALREFLECTING_DIFFRAYS_H

#include "Material_DiffRays.h"
#include <sampleConfig.h>
#include <stdio.h>
#include "../Group.h"
#include "../MaterialReflecting.h"
#include "MaterialReflecting_DiffRays_hit.h"

#define PATH_TO_HIT_REFLECTING_DIFFRAYS "ITO-MacroSim_generated_hitFunctionReflecting"

/* declare class */
/**
  *\class   MaterialReflecting_DiffRays 
  *\brief   reduced set of params that is calculated before the actual tracing from the full set of params. This parameter set will be loaded onto the GPU if the tracing is done there
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
class MaterialReflecting_DiffRays: public MaterialReflecting
{
	protected:

  public:
    /* standard constructor */
    MaterialReflecting_DiffRays()
	{
		/* set ptx path for OptiX calculations */
//		path_to_ptx[512];
		sprintf( path_to_ptx, "%s" PATH_SEPARATOR "%s", SAMPLES_PTX_DIR, PATH_TO_HIT_REFLECTING_DIFFRAYS );
//		this->setPathToPtx(path_to_ptx);
	}
	/* Destruktor */
    ~MaterialReflecting_DiffRays()
	{
//		delete path_to_ptx;
	}

 //   MaterialError createOptiXInstance(RTcontext context, RTgeometryinstance &instance, int index, simMode mode, double lambda);
	//MaterialError updateOptiXInstance(RTcontext context, RTgeometryinstance &instance, int index, simMode mode, double lambda);
	//MaterialError createCPUSimInstance(double lambda);
//	void setPathToPtx(char* path);
//	char* getPathToPtx(void);
	void hit(diffRayStruct &ray, Mat_DiffRays_hitParams hitParams, double t_hit, int geometryID);
};

#endif


