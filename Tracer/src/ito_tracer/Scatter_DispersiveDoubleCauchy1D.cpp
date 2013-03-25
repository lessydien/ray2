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

/**\file Scatter_DispersiveDoubleCauchy1D.cpp
* \brief scattering in one dimension according to th Torrance Sparrow model
* 
*           
* \author Mauch
*/

#include "Scatter_DispersiveDoubleCauchy1D.h"
#include "GlobalConstants.h"
#include "myUtil.h"
#include "sampleConfig.h"
#include <iostream>
#include <string.h>
#include "rayTracingMath.h"


ScatterError Scatter_DispersiveDoubleCauchy1D::setFullParams(ScatDispersiveDoubleCauchy1D_scatParams* ptrIn)
{
	this->fullParamsPtr=ptrIn;
	return SCAT_NO_ERROR;
};

ScatDispersiveDoubleCauchy1D_scatParams* Scatter_DispersiveDoubleCauchy1D::getFullParams(void)
{
	return this->fullParamsPtr;
};

ScatterError Scatter_DispersiveDoubleCauchy1D::setReducedParams(ScatDispersiveDoubleCauchy1D_params* ptrIn)
{
	this->reducedParams=*ptrIn;
	return SCAT_NO_ERROR;
};

ScatDispersiveDoubleCauchy1D_params* Scatter_DispersiveDoubleCauchy1D::getReducedParams(void)
{
	return &(this->reducedParams);
};

void Scatter_DispersiveDoubleCauchy1D::hit(rayStruct &ray, Mat_hitParams hitParams)
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

void Scatter_DispersiveDoubleCauchy1D::hit(gaussBeamRayStruct &ray, gaussBeam_geometricNormal normal)
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

ScatterError Scatter_DispersiveDoubleCauchy1D::setParams2Program( RTcontext context, RTprogram *closest_hit_programPtr, RTvariable *l_scatterParamsPtr)
{
	if ( !RT_CHECK_ERROR_NOEXIT( rtProgramDeclareVariable( *closest_hit_programPtr, "scatterParams", l_scatterParamsPtr ), context) )
		return SCAT_ERROR;
	if ( !RT_CHECK_ERROR_NOEXIT( rtVariableSetUserData(*l_scatterParamsPtr, sizeof(ScatDispersiveDoubleCauchy1D_params), &(this->reducedParams)), context) )
		return SCAT_ERROR;

	return SCAT_NO_ERROR;
};


ScatterError Scatter_DispersiveDoubleCauchy1D::createCPUSimInstance(double lambda)
{
	double l_lambda=lambda*1e3; // we need lambda in um here...
	// calc the refractive indices at current wavelength
	this->reducedParams.Ksl=this->fullParamsPtr->a_k_sl*l_lambda*l_lambda+this->fullParamsPtr->c_k_sl;
	this->reducedParams.Ksp=this->fullParamsPtr->a_k_sp*l_lambda*l_lambda+this->fullParamsPtr->c_k_sp;
	this->reducedParams.scatAxis=this->fullParamsPtr->scatAxis;
	this->reducedParams.gammaXsl=this->fullParamsPtr->a_gamma_sl*l_lambda*l_lambda+this->fullParamsPtr->c_gamma_sl;
	this->reducedParams.gammaXsp=this->fullParamsPtr->a_gamma_sp*l_lambda*l_lambda+this->fullParamsPtr->c_gamma_sp;
	this->reducedParams.impAreaHalfWidth=this->fullParamsPtr->impAreaHalfWidth;
	this->reducedParams.impAreaRoot=this->fullParamsPtr->impAreaRoot;
	this->reducedParams.impAreaTilt=this->fullParamsPtr->impAreaTilt;
	this->reducedParams.impAreaType=this->fullParamsPtr->impAreaType;
	this->update=false;
	return SCAT_NO_ERROR;
};

/**
 * \detail processParseResults 
 *
 * sets the parameters of the detector according to the given parse results
 *
 * \param[in] GeometryParseParamStruct &parseResults_Geom
 * 
 * \return detError
 * \sa 
 * \remarks 
 * \author Mauch
 */
ScatterError Scatter_DispersiveDoubleCauchy1D::processParseResults(MaterialParseParamStruct &parseResults_Mat)
{
	this->fullParamsPtr=new ScatDispersiveDoubleCauchy1D_scatParams;
	this->fullParamsPtr->type=ST_DISPDOUBLECAUCHY1D;
	this->fullParamsPtr->Ksl=parseResults_Mat.varParams[0];
	this->fullParamsPtr->Ksp=parseResults_Mat.varParams[1];
	this->fullParamsPtr->gammaXsl=parseResults_Mat.varParams[2];
	this->fullParamsPtr->gammaXsp=parseResults_Mat.varParams[3];
	this->fullParamsPtr->scatAxis=parseResults_Mat.scatteringAxis;
	this->fullParamsPtr->scatAxis=normalize(this->fullParamsPtr->scatAxis);
	this->fullParamsPtr->a_gamma_sl=parseResults_Mat.varParams[7];
	this->fullParamsPtr->c_gamma_sl=parseResults_Mat.varParams[8];
	this->fullParamsPtr->a_gamma_sp=parseResults_Mat.varParams[9];
	this->fullParamsPtr->c_gamma_sp=parseResults_Mat.varParams[10];
	this->fullParamsPtr->a_k_sp=parseResults_Mat.varParams[11];
	this->fullParamsPtr->c_k_sp=parseResults_Mat.varParams[12];
	this->fullParamsPtr->a_k_sl=parseResults_Mat.varParams[13];
	this->fullParamsPtr->c_k_sl=parseResults_Mat.varParams[14];

	return SCAT_NO_ERROR;
};

