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

/**\file MaterialLight_GeomRender.h
* \brief 
* 
*           
* \author Mauch
*/

#ifndef MATERIALRENDERLIGHT_H
#define MATERIALRENDERLIGHT_H

#include "..\Material.h"
#include <sampleConfig.h>
#include <stdio.h>
#include "..\Group.h"
#include "MaterialLight_GeomRender_hit.h"

#define PATH_TO_HIT_RENDERLIGHT "macrosim_tracer_generated_hitFunctionLight_GeomRender"

/* declare class */
/**
  *\class   MaterialLight_GeomRender
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
class MaterialLight_GeomRender: public Material
{
	protected:
		MatLight_GeomRender_params params; // reduced parameter set for the ray trac ( on GPU )

  public:
    /* standard constructor */
    MaterialLight_GeomRender()
	{
		params.power=1;
		/* set ptx path for OptiX calculations */
		//path_to_ptx=(char*)malloc(512*sizeof(char));
		sprintf( path_to_ptx, "%s" PATH_SEPARATOR "%s", PATH_TO_PTX, PATH_TO_HIT_RENDERLIGHT );
		//this->setPathToPtx(path_to_ptx);
	}
	/* Destruktor */
    ~MaterialLight_GeomRender()
	{
//		delete path_to_ptx;
	}

	void setParams(MatLight_GeomRender_params params);
	MatLight_GeomRender_params getParams(void);
    MaterialError createOptiXInstance(RTcontext context, RTgeometryinstance &instance, int index, SimParams simParams, double lambda);
	MaterialError updateOptiXInstance(RTcontext context, RTgeometryinstance &instance, int index, SimParams simParams, double lambda);
	MaterialError createCPUSimInstance(double lambda);
//	void setPathToPtx(char* path);
//	char* getPathToPtx(void);
	void hit(rayStruct &ray, Mat_hitParams hitParams, double t_hit, int geometryID);
//	void hit(diffRayStruct &ray,double3 normal, double3 mainDirX, double3 mainDirY, double2 mainRad, double t_hit, int geometryID);
	MaterialError parseXml(pugi::xml_node &geometry, SimParams simParams);	

};

#endif


