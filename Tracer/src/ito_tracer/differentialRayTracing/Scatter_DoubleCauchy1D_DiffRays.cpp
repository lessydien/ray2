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

/**\file Scatter_DoubleCauchy1D_DiffRays.cpp
* \brief scattering in one dimension according to th Torrance Sparrow model
* 
*           
* \author Mauch
*/

#include "Scatter_DoubleCauchy1D_DiffRays.h"
#include "../GlobalConstants.h"
#include "../myUtil.h"
#include <sampleConfig.h>
#include <iostream>
#include <string.h>
#include "../rayTracingMath.h"

void Scatter_DoubleCauchy1D_DiffRays::hit(diffRayStruct &ray, Mat_DiffRays_hitParams hitParams)
{
//	extern Group oGroup;
	if (hitDoubleCauchy1D(ray, hitParams, this->reducedParams) )
	{
//		ray.currentGeometryID=geometryID;
		//if (ray.depth<MAX_DEPTH_CPU )//&& ray.flux>MIN_FLUX_CPU)
		//{			
		//	oGroup.trace(ray);
		//}

	}
	else
	{
		std::cout <<"error in ScatterDoubleCauchy1D.hit(): hitDoubleCauchy1D returned an error." << std::endl;
		// some error mechanism !!
	}

}

ScatterError Scatter_DoubleCauchy1D_DiffRays::createOptiXInstance(double lambda, char** path_to_ptx_in)
{
	// calc the refractive indices at current wavelength
	this->reducedParams.Ksl=this->fullParamsPtr->Ksl;
	this->reducedParams.Ksp=this->fullParamsPtr->Ksp;
	this->reducedParams.scatAxis=this->fullParamsPtr->scatAxis;
	this->reducedParams.gammaXsl=this->fullParamsPtr->gammaXsl;
	this->reducedParams.gammaXsp=this->fullParamsPtr->gammaXsp;
	this->update=false;
	strcat(*path_to_ptx_in, this->path_to_ptx);
	return SCAT_NO_ERROR;
};

ScatterError Scatter_DoubleCauchy1D_DiffRays::createCPUSimInstance(double lambda)
{
	// calc the refractive indices at current wavelength
	this->reducedParams.Ksl=this->fullParamsPtr->Ksl;
	this->reducedParams.Ksp=this->fullParamsPtr->Ksp;
	this->reducedParams.scatAxis=this->fullParamsPtr->scatAxis;
	this->reducedParams.gammaXsl=this->fullParamsPtr->gammaXsl;
	this->reducedParams.gammaXsp=this->fullParamsPtr->gammaXsp;
	return SCAT_NO_ERROR;
};

