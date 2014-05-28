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

/**\file Scatter_Lambert2D_GeomRender.cpp
* \brief scattering in one dimension according to th Torrance Sparrow model
* 
*           
* \author Mauch
*/

#include "Scatter_Lambert2D_GeomRender.h"
#include "../GlobalConstants.h"
#include "../myUtil.h"
#include <sampleConfig.h>
#include <iostream>
#include <string.h>
#include "../rayTracingMath.h"
#include "../Group.h"


void Scatter_Lambert2D_GeomRender::hit(rayStruct &ray, Mat_hitParams hitParams)
{
	extern Group oGroup;
    // cast to geomRenderRay
    geomRenderRayStruct *l_pRay=reinterpret_cast<geomRenderRayStruct*>(&ray);
    // create secondary ray
    geomRenderRayStruct sdRay=*l_pRay;
    sdRay.secondary=true;
    sdRay.secondary_nr++;
    if (hitLambert2D_GeomRender(sdRay, hitParams, this->reducedParams) )
    {
		if (sdRay.secondary_nr<2 && sdRay.flux>MIN_FLUX_CPU)
		{			
			oGroup.trace(ray);
		}
        l_pRay->cumFlux+=sdRay.cumFlux;
    }

    // continue primary ray
    ScatLambert2D_params primParams=this->reducedParams;
    primParams.impAreaType=AT_INFTY; // primary ray does not use the importance area
	if (hitLambert2D_GeomRender(*l_pRay, hitParams, primParams) )
	{
//		ray.currentGeometryID=geometryID;
		//if (ray.depth<MAX_DEPTH_CPU )//&& ray.flux>MIN_FLUX_CPU)
		//{			
		//	oGroup.trace(ray);
		//}

	}
	else
	{
		std::cout <<"error in ScatterDoubleCauchy1D.hit(): hitDoubleCauchy1D returned an error." << "...\n";
		// some error mechanism !!
	}

}

ScatterError Scatter_Lambert2D_GeomRender::createOptiXInstance(double lambda, char** path_to_ptx_in)
{
	// calc the refractive indices at current wavelength
	this->reducedParams.TIR=this->fullParamsPtr->TIR;
	//this->reducedParams.importanceAreaHalfWidth=this->fullParamsPtr->importanceAreaHalfWidth;
	//this->reducedParams.importanceAreaRoot=this->fullParamsPtr->importanceAreaRoot;
	//this->reducedParams.importanceAreaTilt=this->fullParamsPtr->importanceAreaTilt;
	//this->reducedParams.importanceAreaApertureType=this->fullParamsPtr->importanceAreaApertureType;
	this->update=false;
	strcat(*path_to_ptx_in, this->path_to_ptx);
	return SCAT_NO_ERROR;
};

ScatterError Scatter_Lambert2D_GeomRender::createCPUSimInstance(double lambda)
{
	// calc the refractive indices at current wavelength
	this->reducedParams.TIR=this->fullParamsPtr->TIR;
	//this->reducedParams.importanceAreaHalfWidth=this->fullParamsPtr->importanceAreaHalfWidth;
	//this->reducedParams.importanceAreaRoot=this->fullParamsPtr->importanceAreaRoot;
	//this->reducedParams.importanceAreaTilt=this->fullParamsPtr->importanceAreaTilt;
	//this->reducedParams.importanceAreaApertureType=this->fullParamsPtr->importanceAreaApertureType;

	return SCAT_NO_ERROR;
};

