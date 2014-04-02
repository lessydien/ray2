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

/**\file Scatter_Lambert2D.cpp
* \brief scattering in one dimension according to th Torrance Sparrow model
* 
*           
* \author Mauch
*/

#include "Scatter_Lambert2D.h"
#include "GlobalConstants.h"
#include "myUtil.h"
#include "sampleConfig.h"
#include <iostream>
#include <string.h>
#include "rayTracingMath.h"

#include "Parser_XML.h"

ScatterError Scatter_Lambert2D::setFullParams(ScatLambert2D_scatParams* ptrIn)
{
	if (this->fullParamsPtr != NULL)
		delete this->fullParamsPtr;
	this->fullParamsPtr=ptrIn;
	return SCAT_NO_ERROR;
};

ScatLambert2D_scatParams* Scatter_Lambert2D::getFullParams(void)
{
	return this->fullParamsPtr;
};

ScatterError Scatter_Lambert2D::setReducedParams(ScatLambert2D_params* ptrIn)
{
	this->reducedParams=*ptrIn;
	return SCAT_NO_ERROR;
};

ScatLambert2D_params* Scatter_Lambert2D::getReducedParams(void)
{
	return &(this->reducedParams);
};

void Scatter_Lambert2D::hit(rayStruct &ray, Mat_hitParams hitParams)
{
//	extern Group oGroup;
	if (hitLambert2D(ray, hitParams, this->reducedParams) )
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

void Scatter_Lambert2D::hit(gaussBeamRayStruct &ray, gaussBeam_geometricNormal normal)
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

ScatterError Scatter_Lambert2D::setParams2Program( RTcontext context, RTprogram *closest_hit_programPtr, RTvariable *l_scatterParamsPtr)
{
	if ( !RT_CHECK_ERROR_NOEXIT( rtProgramDeclareVariable( *closest_hit_programPtr, "scatterParams", l_scatterParamsPtr ), context) )
		return SCAT_ERROR;
	if ( !RT_CHECK_ERROR_NOEXIT( rtVariableSetUserData(*l_scatterParamsPtr, sizeof(ScatLambert2D_params), &(this->reducedParams)), context) )
		return SCAT_ERROR;
	
	return SCAT_NO_ERROR;
};

ScatterError Scatter_Lambert2D::createCPUSimInstance(double lambda)
{
	// calc the refractive indices at current wavelength
	this->update=false;
	this->reducedParams.TIR=this->fullParamsPtr->TIR;
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
ScatterError Scatter_Lambert2D::processParseResults(MaterialParseParamStruct &parseResults_Mat)
{
	this->fullParamsPtr=new ScatLambert2D_scatParams;
	this->fullParamsPtr->type=ST_LAMBERT2D;
	this->fullParamsPtr->TIR=parseResults_Mat.varParams[0];
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
ScatterError Scatter_Lambert2D::parseXml(pugi::xml_node &geometry, SimParams simParams)
{
	Parser_XML l_parser;
	if (!this->checkParserError(l_parser.attrByNameToDouble(geometry, "Tis", this->getFullParams()->TIR)))
		return SCAT_ERROR;
	return SCAT_NO_ERROR;
}