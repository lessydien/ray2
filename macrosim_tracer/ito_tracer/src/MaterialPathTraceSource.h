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

/**\file MaterialPathTraceSource.h
* \brief 
* 
*           
* \author Mauch
*/

#ifndef MATERIALPATHTRACESRC_H
#define MATERIALPATHTRACESRC_H

#include "Material.h"
#include <sampleConfig.h>
#include <stdio.h>
#include "Group.h"
#include "MaterialPathTraceSource_hit.h"

#define PATH_TO_HIT_PATHTRACESOURCE "macrosim_tracer_generated_hitFunctionPathTraceSource"

/* declare class */
/**
  *\class   MaterialPathTraceSource 
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
class MaterialPathTraceSource: public Material
{
	protected:
		MatPathTraceSource_params params;

  public:
    /* standard constructor */
    MaterialPathTraceSource()
	{
		/* set ptx path for OptiX calculations */
//		path_to_ptx[512];
		sprintf( path_to_ptx, "%s" PATH_SEPARATOR "%s", PATH_TO_PTX, PATH_TO_HIT_PATHTRACESOURCE );
//		this->setPathToPtx(path_to_ptx);
	}
	/* Destruktor */
    ~MaterialPathTraceSource()
	{
//		delete path_to_ptx;
	}

    MaterialError createOptiXInstance(RTcontext context, RTgeometryinstance &instance, int index, SimParams simParams, double lambda);
	MaterialError updateOptiXInstance(RTcontext context, RTgeometryinstance &instance, int index, SimParams simParams, double lambda);
	void setParams(MatPathTraceSource_params params);
	MatPathTraceSource_params getParams(void);
	void hit(rayStruct &ray, Mat_hitParams hitParams, double t_hit, int geometryID);
	MaterialError processParseResults(MaterialParseParamStruct &parseResults_Mat);

};

#endif


