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

/**\file Scatter_TorranceSparrow2D_GeomRender.cpp
* \brief scattering in one dimension according to th Torrance Sparrow model
* 
*           
* \author Mauch
*/

#include "Scatter_TorranceSparrow2D_GeomRender.h"
#include "../Group.h"
//#include "GlobalConstants.h"
//#include "myUtil.h"
//#include "sampleConfig.h"
//#include <iostream>
//#include <string.h>
//#include "rayTracingMath.h"
//#include "TopObject.h"

void Scatter_TorranceSparrow2D_GeomRender::hit(rayStruct &ray, Mat_hitParams hitParams)
{
	extern Group *oGroup;
	//rayStruct_GeomRender* ray_interpreted;
	//// we have a path tracing ray here ... hopefully ...
	//ray_interpreted=reinterpret_cast<rayStruct_PathTracing*>(&ray);
	//if (!ray_interpreted->secondary)
	//{
	//	// create secondary ray
	//	rayStruct_PathTracing l_ray;
	//	l_ray.position=ray_interpreted->position;
	//	l_ray.currentGeometryID=ray_interpreted->currentGeometryID;
	//	l_ray.currentSeed=ray_interpreted->currentSeed;
	//	l_ray.flux=ray_interpreted->flux;
	//	l_ray.lambda=ray_interpreted->lambda;
	//	l_ray.nImmersed=ray_interpreted->nImmersed;
	//	l_ray.opl=ray_interpreted->opl;
	//	l_ray.result=0;
	//	l_ray.depth=ray_interpreted->depth+1;
	//	l_ray.secondary_nr=ray_interpreted->secondary_nr+1;
	//	l_ray.running=true;
	//	l_ray.secondary=true;
	//	// aim it towards light source
	//	aimRayTowardsImpArea(l_ray.direction, l_ray.position, this->reducedParams.srcAreaRoot, this->reducedParams.srcAreaHalfWidth, this->reducedParams.srcAreaTilt, this->reducedParams.srcAreaType, l_ray.currentSeed);
	//	// adjust flux of secondary ray according to scattering angle
	//	double l_scatAngle=dot(l_ray.direction,ray_interpreted->direction);
	//	l_ray.flux=l_ray.flux*this->reducedParams.Kdl*cos(l_scatAngle)+this->reducedParams.Ksl*exp(-l_scatAngle*l_scatAngle/(2*this->reducedParams.sigmaXsl))+this->reducedParams.Ksp*exp(-l_scatAngle*l_scatAngle/(2*this->reducedParams.sigmaXsp));
	//	// trace secondary ray only if it is not blocked by the scattering surface itself
	//	if ( dot(ray_interpreted->direction,l_ray.direction)>0 )
	//	{
	//		// trace secondary ray towards light source
	//		oGroup->trace(l_ray);
	//	}
	//	if (l_ray.result>0)
	//	{
	//		// add result of secondary ray to our initial ray
	//		ray_interpreted->result=ray_interpreted->result+l_ray.result; // do we need some kind of normalization here ???!!!!
	//		ray_interpreted->secondary_nr=l_ray.secondary_nr;
	//	}
	//}
	//// continue conventional tracing
	//if (!hitTorranceSparrow2D_PathTrace(*ray_interpreted, hitParams, this->reducedParams) )
	//{
	//	std::cout <<"error in ScatterTorranceSparrow1D.hit(): hitTorranceSparrow1D returned an error." << "...\n";
	//	// some error mechanism !!
	//}
}

ScatterError Scatter_TorranceSparrow2D_GeomRender::createCPUSimInstance()
{
	// calc the refractive indices at current wavelength
	this->reducedParams.Kdl=this->fullParamsPtr->Kdl;
	this->reducedParams.Ksl=this->fullParamsPtr->Ksl;
	this->reducedParams.Ksp=this->fullParamsPtr->Ksp;
	this->reducedParams.scatAxis=this->fullParamsPtr->scatAxis;
	this->reducedParams.sigmaXsl=this->fullParamsPtr->sigmaXsl;
	this->reducedParams.sigmaXsp=this->fullParamsPtr->sigmaXsp;
	this->reducedParams.impAreaHalfWidth=this->fullParamsPtr->impAreaHalfWidth;
	this->reducedParams.impAreaRoot=this->fullParamsPtr->impAreaRoot;
	this->reducedParams.impAreaTilt=this->fullParamsPtr->impAreaTilt;
	this->reducedParams.impAreaType=this->fullParamsPtr->impAreaType;

	return SCAT_NO_ERROR;
};

ScatterError Scatter_TorranceSparrow2D_GeomRender::createOptiXInstance(char** path_to_ptx_in)
{
    this->createCPUSimInstance();
	this->update=false;
	strcat(*path_to_ptx_in, this->path_to_ptx);
	return SCAT_NO_ERROR;
};

