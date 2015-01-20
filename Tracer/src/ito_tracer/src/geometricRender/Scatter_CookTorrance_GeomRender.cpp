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

/**\file Scatter_CookTorrance_GeomRender.cpp
* \brief scattering in one dimension according to th Torrance Sparrow model
* 
*           
* \author Mauch
*/

#include "Scatter_CookTorrance_GeomRender.h"
#include "../GlobalConstants.h"
#include "../myUtil.h"
#include <sampleConfig.h>
#include <iostream>
#include <string.h>
#include "../rayTracingMath.h"
#include "../Group.h"


void Scatter_CookTorrance_GeomRender::hit(rayStruct &ray, Mat_hitParams hitParams)
{
	extern Group *oGroup;
    // cast to geomRenderRay
    geomRenderRayStruct *l_pRay=reinterpret_cast<geomRenderRayStruct*>(&ray);
    // create secondary ray
    geomRenderRayStruct sdRay=*l_pRay;
    sdRay.secondary=true;
    sdRay.secondary_nr++;
    sdRay.cumFlux=0;
    if (hitCookTorrance_GeomRender(sdRay, hitParams, this->reducedParams) )
    {
		if (sdRay.secondary_nr<2 && sdRay.flux>MIN_FLUX_CPU)
		{
            // do the trace the secondary ray through the scene
            while (sdRay.running)
			    oGroup->trace(sdRay);
            l_pRay->cumFlux+=sdRay.cumFlux;
            l_pRay->currentSeed=sdRay.currentSeed;
		}       
    }

//    ray.running=false;

    // continue primary ray
    ScatCookTorrance_params primParams=this->reducedParams;
    primParams.impAreaType=AT_INFTY; // primary ray does not use the importance area
	if (hitCookTorrance_GeomRender(*l_pRay, hitParams, primParams) )
	{

	}
	else
	{
		std::cout <<"error in ScatterDoubleCauchy1D.hit(): hitDoubleCauchy1D returned an error." << "...\n";
		// some error mechanism !!
	}

}

ScatterError Scatter_CookTorrance_GeomRender::createOptiXInstance(double lambda, char** path_to_ptx_in)
{
	// calc the refractive indices at current wavelength
    this->reducedParams.coefLambertian=this->fullParamsPtr->coefLambertian;
    this->reducedParams.fresnelParam=this->fullParamsPtr->fresnelParam;
    this->reducedParams.roughnessFactor=this->fullParamsPtr->roughnessFactor;
	this->reducedParams.impAreaHalfWidth=this->fullParamsPtr->impAreaHalfWidth;
	this->reducedParams.impAreaRoot=this->fullParamsPtr->impAreaRoot;
	this->reducedParams.impAreaTilt=this->fullParamsPtr->impAreaTilt;
	this->reducedParams.impAreaType=this->fullParamsPtr->impAreaType;
	this->update=false;
	strcat(*path_to_ptx_in, this->path_to_ptx);
	return SCAT_NO_ERROR;
};

ScatterError Scatter_CookTorrance_GeomRender::createCPUSimInstance(double lambda)
{
	// calc the refractive indices at current wavelength
    this->reducedParams.coefLambertian=this->fullParamsPtr->coefLambertian;
    this->reducedParams.fresnelParam=this->fullParamsPtr->fresnelParam;
    this->reducedParams.roughnessFactor=this->fullParamsPtr->roughnessFactor;
	this->reducedParams.impAreaHalfWidth=this->fullParamsPtr->impAreaHalfWidth;
	this->reducedParams.impAreaRoot=this->fullParamsPtr->impAreaRoot;
	this->reducedParams.impAreaTilt=this->fullParamsPtr->impAreaTilt;
	this->reducedParams.impAreaType=this->fullParamsPtr->impAreaType;

	return SCAT_NO_ERROR;
};

