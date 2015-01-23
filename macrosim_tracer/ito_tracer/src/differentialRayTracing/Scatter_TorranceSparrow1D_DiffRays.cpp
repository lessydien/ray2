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

/**\file Scatter_TorranceSparrow1D_DiffRays.cpp
* \brief scattering in one dimension according to th Torrance Sparrow model
* 
*           
* \author Mauch
*/

#include "Scatter_TorranceSparrow1D_DiffRays.h"
#include "Scatter_TorranceSparrow1D_DiffRays_hit.h"
#include "../GlobalConstants.h"
#include "../myUtil.h"
#include <sampleConfig.h>
#include <iostream>
#include <string.h>
#include "../rayTracingMath.h"

void Scatter_TorranceSparrow1D_DiffRays::hit(diffRayStruct &ray, Mat_DiffRays_hitParams hitParams)
{
//	extern Group oGroup;
	if (hitTorranceSparrow1D_DiffRays(ray, hitParams, this->reducedParams) )
	{
//		ray.currentGeometryID=geometryID;
		//if (ray.depth<MAX_DEPTH_CPU )//&& ray.flux>MIN_FLUX_CPU)
		//{			
		//	oGroup.trace(ray);
		//}

	}
	else
	{
		std::cout <<"error in ScatterTorranceSparrow1D.hit(): hitTorranceSparrow1D returned an error." << "...\n";
		// some error mechanism !!
	}

}

ScatterError Scatter_TorranceSparrow1D_DiffRays::createOptiXInstance(double lambda, char** path_to_ptx_in)
{
	// calc the refractive indices at current wavelength
	this->reducedParams.Kdl=this->fullParamsPtr->Kdl;
	this->reducedParams.Ksl=this->fullParamsPtr->Ksl;
	this->reducedParams.Ksp=this->fullParamsPtr->Ksp;
	this->reducedParams.scatAxis=this->fullParamsPtr->scatAxis;
	this->reducedParams.sigmaXsl=this->fullParamsPtr->sigmaXsl;
	this->reducedParams.sigmaXsp=this->fullParamsPtr->sigmaXsp;
	//this->reducedParams.importanceAreaHalfWidth=this->fullParamsPtr->importanceAreaHalfWidth;
	//this->reducedParams.importanceAreaRoot=this->fullParamsPtr->importanceAreaRoot;
	//this->reducedParams.importanceAreaTilt=this->fullParamsPtr->importanceAreaTilt;
	//this->reducedParams.importanceAreaApertureType=this->fullParamsPtr->importanceAreaApertureType;
	this->update=false;
	strcat(*path_to_ptx_in, this->path_to_ptx);
	return SCAT_NO_ERROR;
};

ScatterError Scatter_TorranceSparrow1D_DiffRays::createCPUSimInstance(double lambda)
{
	// calc the refractive indices at current wavelength
	this->reducedParams.Kdl=this->fullParamsPtr->Kdl;
	this->reducedParams.Ksl=this->fullParamsPtr->Ksl;
	this->reducedParams.Ksp=this->fullParamsPtr->Ksp;
	this->reducedParams.scatAxis=this->fullParamsPtr->scatAxis;
	this->reducedParams.sigmaXsl=this->fullParamsPtr->sigmaXsl;
	this->reducedParams.sigmaXsp=this->fullParamsPtr->sigmaXsp;
	//this->reducedParams.importanceAreaHalfWidth=this->fullParamsPtr->importanceAreaHalfWidth;
	//this->reducedParams.importanceAreaRoot=this->fullParamsPtr->importanceAreaRoot;
	//this->reducedParams.importanceAreaTilt=this->fullParamsPtr->importanceAreaTilt;
	//this->reducedParams.importanceAreaApertureType=this->fullParamsPtr->importanceAreaApertureType;
	this->update=false;
	return SCAT_NO_ERROR;
};

