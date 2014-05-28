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

/**\file MaterialIdealLense_GeomRender.h
* \brief 
* 
*           
* \author Mauch
*/

#ifndef MATERIALIDEALLENSE_GEOMRENDER_H
#define MATERIALIDEALLENSE_GEOMRENDER_H

#include "Material_GeomRender.h"
#include <sampleConfig.h>
#include <stdio.h>
#include "../Group.h"
#include "../MaterialIdealLense.h"
#include "MaterialIdealLense_GeomRender_hit.h"

#define PATH_TO_HIT_IDEALLENSE_GEOMRENDER "ITO-MacroSim_generated_hitFunctionIdealLense"

/* declare class */
/**
  *\class   MaterialIdealLense_GeomRender 
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
class MaterialIdealLense_GeomRender : public MaterialIdealLense
{

  public:
    /* standard constructor */
    MaterialIdealLense_GeomRender()
	{
		params.f=0;
		dispersionParamsPtr=new MatIdealLense_DispersionParams();
		/* set ptx path for OptiX calculations */
//		path_to_ptx[512];
		sprintf( path_to_ptx, "%s" PATH_SEPARATOR "%s", PATH_TO_PTX, PATH_TO_HIT_IDEALLENSE_GEOMRENDER );
//		this->setPathToPtx(path_to_ptx);
	}
	/* Destruktor */
    ~MaterialIdealLense_GeomRender()
	{
//		if (this->dispersionParamsPtr!=NULL)
//		{
//			delete this->dispersionParamsPtr;
//			this->dispersionParamsPtr=NULL;
//		}
//		delete path_to_ptx;
	}

 //   MaterialError createOptiXInstance(RTcontext context, RTgeometryinstance &instance, int index, SimParams simParams, double lambda);
	//MaterialError updateOptiXInstance(RTcontext context, RTgeometryinstance &instance, int index, SimParams simParams, double lambda);
	//MaterialError createCPUSimInstance(double lambda);
//	void hit(geomRenderRayStruct &ray, Mat_GeomRender_hitParams hitParams, double t_hit, int geometryID);
//	MaterialError updateCPUSimInstance(double lambda);
};

#endif


