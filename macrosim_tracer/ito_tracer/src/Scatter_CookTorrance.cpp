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

/**\file Scatter_Phong.cpp
* \brief scattering in one dimension according to th Torrance Sparrow model
* 
*           
* \author Mauch
*/

#include "Scatter_CookTorrance.h"
#include "GlobalConstants.h"
#include "myUtil.h"
#include "sampleConfig.h"
#include <iostream>
#include <string.h>
#include "rayTracingMath.h"

#include "Parser_XML.h"

ScatterError Scatter_CookTorrance::setFullParams(ScatCookTorrance_scatParams* ptrIn)
{
	if (this->fullParamsPtr != NULL)
		delete this->fullParamsPtr;
	this->fullParamsPtr=ptrIn;
	return SCAT_NO_ERROR;
};

ScatCookTorrance_scatParams* Scatter_CookTorrance::getFullParams(void)
{
	return this->fullParamsPtr;
};

ScatterError Scatter_CookTorrance::setReducedParams(ScatCookTorrance_params* ptrIn)
{
	this->reducedParams=*ptrIn;
	return SCAT_NO_ERROR;
};

ScatCookTorrance_params* Scatter_CookTorrance::getReducedParams(void)
{
	return &(this->reducedParams);
};

void Scatter_CookTorrance::hit(rayStruct &ray, Mat_hitParams hitParams)
{
//	extern Group oGroup;
	if (hitCookTorrance(ray, hitParams, this->reducedParams) )
	{
//		ray.currentGeometryID=geometryID;
		//if (ray.depth<MAX_DEPTH_CPU )//&& ray.flux>MIN_FLUX_CPU)
		//{			
		//	oGroup.trace(ray);
		//}

	}
	else
	{
		std::cout <<"error in ScatterCookTorrance.hit(): CookTorrance returned an error." << "...\n";
		// some error mechanism !!
	}

}

void Scatter_CookTorrance::hit(gaussBeamRayStruct &ray, gaussBeam_geometricNormal normal)
{
//	extern Group oGroup;
	// refract all the rays making up the gaussian beam
	//ray.baseRay.direction=calcSnellsLaw(ray.baseRay.direction, normal.normal_baseRay,ray.nImmersed, n);
	//ray.waistRayX.direction=calcSnellsLaw(ray.waistRayX.direction, normal.normal_waistRayX,ray.nImmersed, n);
	//ray.waistRayY.direction=calcSnellsLaw(ray.waistRayY.direction, normal.normal_waistRayY,ray.nImmersed, n);
	//ray.divRayX.direction=calcSnellsLaw(ray.divRayX.direction, normal.normal_divRayX,ray.nImmersed, n);
	//ray.divRayY.direction=calcSnellsLaw(ray.divRayY.direction, normal.normal_divRayY,ray.nImmersed, n);
	//if (ray.baseRay.depth<MAX_DEPTH_CPU && ray.baseRay.flux>MIN_FLUX_CPU)
	//{			
	//	oGroup.trace(ray);
	//}
}

ScatterError Scatter_CookTorrance::setParams2Program( RTcontext context, RTprogram *closest_hit_programPtr, RTvariable *l_scatterParamsPtr)
{
	if ( !RT_CHECK_ERROR_NOEXIT( rtProgramDeclareVariable( *closest_hit_programPtr, "scatterParams", l_scatterParamsPtr ), context) )
		return SCAT_ERROR;
	if ( !RT_CHECK_ERROR_NOEXIT( rtVariableSetUserData(*l_scatterParamsPtr, sizeof(ScatCookTorrance_params), &(this->reducedParams)), context) )
		return SCAT_ERROR;
	
	return SCAT_NO_ERROR;
};

ScatterError Scatter_CookTorrance::createCPUSimInstance(double lambda)
{
	// calc the refractive indices at current wavelength
	this->update=false;
	this->reducedParams.coefLambertian=this->fullParamsPtr->coefLambertian;
	this->reducedParams.fresnelParam=this->fullParamsPtr->fresnelParam;
	this->reducedParams.roughnessFactor=this->fullParamsPtr->roughnessFactor;
	this->reducedParams.impAreaHalfWidth=this->fullParamsPtr->impAreaHalfWidth;
	this->reducedParams.impAreaRoot=this->fullParamsPtr->impAreaRoot;
	this->reducedParams.impAreaTilt=this->fullParamsPtr->impAreaTilt;
	this->reducedParams.impAreaType=this->fullParamsPtr->impAreaType;
	return SCAT_NO_ERROR;
};

/**
 * \detail processParseResults 
 *
 * sets the parameters of the detector according to the given parse results
 *
 * \param[in] MaterialParseParamStruct &parseResults_Geom
 * 
 * \return detError
 * \sa 
 * \remarks 
 * \author Mauch
 */
ScatterError Scatter_CookTorrance::processParseResults(MaterialParseParamStruct &parseResults_Mat)
{
	this->fullParamsPtr=new ScatCookTorrance_scatParams;
	this->fullParamsPtr->type=ST_LAMBERT2D;
//	this->fullParamsPtr->TIR=parseResults_Mat.varParams[0];
	this->fullParamsPtr->impAreaHalfWidth=parseResults_Mat.importanceAreaHalfWidth;
	this->fullParamsPtr->impAreaRoot=parseResults_Mat.importanceAreaRoot;
	this->fullParamsPtr->impAreaTilt=parseResults_Mat.importanceAreaTilt;
	this->fullParamsPtr->impAreaType=parseResults_Mat.importanceAreaApertureType;
	return SCAT_NO_ERROR;
};

/**
 * \detail parseXml 
 *
 * sets the parameters of the scatter according to the given xml node
 *
 * \param[in] xml_node &geometry
 * 
 * \return geometryError
 * \sa 
 * \remarks 
 * \author Mauch
 */
ScatterError Scatter_CookTorrance::parseXml(pugi::xml_node &scatter, SimParams simParams)
{
    if (!Scatter::parseXml(scatter, simParams))
    {
        std::cout << "error in CookTorrance.parseXml(): Scatter.parseXml() returned an error" << "...\n";
        return SCAT_ERROR;
    }
	Parser_XML l_parser;
	if (!this->checkParserError(l_parser.attrByNameToDouble(scatter, "coefLambertian", this->getFullParams()->coefLambertian)))
		return SCAT_ERROR;
	if (!this->checkParserError(l_parser.attrByNameToDouble(scatter, "fresnelParam", this->getFullParams()->fresnelParam)))
		return SCAT_ERROR;
	if (!this->checkParserError(l_parser.attrByNameToDouble(scatter, "roughnessFactor", this->getFullParams()->roughnessFactor)))
		return SCAT_ERROR;
	return SCAT_NO_ERROR;
}