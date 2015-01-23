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

/**\file MaterialReflecting_CovGlass.h
* \brief 
* 
*           
* \author Mauch
*/

#ifndef MATERIALREFLECTING_COVGLASS_H
#define MATERIALREFLECTING_COVGLASS_H

#include "Material.h"
#include "DetectorParams.h"
#include <sampleConfig.h>
#include <stdio.h>
#include "Group.h"
#include "MaterialReflecting_CovGlass_hit.h"

#define PATH_TO_HIT_REFLECTING_COVGLASS "macrosim_tracer_generated_hitFunctionReflecting_CovGlass"

/* declare class */
/**
  *\class   MaterialReflecting_CovGlass 
  *\ingroup Material
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
class MaterialReflecting_CovGlass: public Material
{
	protected:
		MatReflecting_CovGlass_params params;

  public:
    /* standard constructor */
    MaterialReflecting_CovGlass()
	{
		/* set ptx path for OptiX calculations */
//		path_to_ptx[512];
		sprintf( path_to_ptx, "%s" PATH_SEPARATOR "%s", PATH_TO_PTX, PATH_TO_HIT_REFLECTING_COVGLASS );
//		this->setPathToPtx(path_to_ptx);
	}
	/* Destruktor */
    ~MaterialReflecting_CovGlass()
	{
//		delete path_to_ptx;
	}

    MaterialError createOptiXInstance(RTcontext context, RTgeometryinstance &instance, int index, SimParams simParams, double lambda);
	MaterialError updateOptiXInstance(RTcontext context, RTgeometryinstance &instance, int index, SimParams simParams, double lambda);
	void hit(rayStruct &ray, Mat_hitParams hitParams, double t_hit, int geometryID);
	void hit(gaussBeamRayStruct &ray, gaussBeam_geometricNormal normal, int geometryID);
	MaterialError processParseResults(MaterialParseParamStruct &parseResults_Mat, DetectorParseParamStruct &parseResults_Det);

};

#endif


