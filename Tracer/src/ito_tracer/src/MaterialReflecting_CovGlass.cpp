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

/**\file MaterialReflecting_CovGlass.cpp
* \brief reflecting material
* 
*           
* \author Mauch
*/

#include "MaterialReflecting_CovGlass.h"
#include "GlobalConstants.h"
#include "myUtil.h"
#include "sampleConfig.h"
#include <iostream>


/**
 * \detail hit function of material for geometric rays
 *
 * Here we need to call the hit function of the coating first. If the coating transmits the ray through the material it passes without any further deflectiosn. Finally we call the hit function of the Scatter attached to the material
 *
 * \param[in] rayStruct &ray, double3 normal, double t_hit, int geometryID
 * 
 * \return MaterialError
 * \sa 
 * \remarks 
 * \author Mauch
 */
void MaterialReflecting_CovGlass::hit(rayStruct &ray, Mat_hitParams hitParams, double t_hit, int geometryID)
{
//	extern Group oGroup;
	double3 n=hitParams.normal;
	bool coat_reflected = true;
	if ( this->coatingPtr->getFullParams()->type != CT_NOCOATING )
		coat_reflected=this->coatingPtr->hit(ray, hitParams);
	// if the coating wants transmission we do not change the ray direction at all !!!
	if (coat_reflected)
		hitReflecting_CovGlass(ray, hitParams, this->params, t_hit, geometryID);
	if ( this->scatterPtr->getFullParams()->type != ST_NOSCATTER )
		this->scatterPtr->hit(ray, hitParams);

	//if (ray.depth<MAX_DEPTH_CPU && ray.flux>MIN_FLUX_CPU)
	//{			
	//	oGroup.trace(ray);
	//}
	if (ray.depth>MAX_DEPTH_CPU || ray.flux<MIN_FLUX_CPU)
		ray.running=false;//stop ray
}

/**
 * \detail hit function of the material for gaussian beam rays
 *
 * \param[in] gaussBeamRayStruct &ray, gaussBeam_geometricNormal normal, int geometryID
 * 
 * \return void
 * \sa 
 * \remarks not tested yet
 * \author Mauch
 */
void MaterialReflecting_CovGlass::hit(gaussBeamRayStruct &ray, gaussBeam_geometricNormal normal, int geometryID)
{
		extern Group oGroup;
		// reflect all the rays making up the gaussian beam
		ray.baseRay.direction=reflect(ray.baseRay.direction,normal.normal_baseRay);
		ray.waistRayX.direction=reflect(ray.waistRayX.direction,normal.normal_waistRayX);
		ray.waistRayY.direction=reflect(ray.waistRayY.direction,normal.normal_waistRayY);
		ray.divRayX.direction=reflect(ray.divRayX.direction,normal.normal_divRayX);
		ray.divRayY.direction=reflect(ray.divRayY.direction,normal.normal_divRayY);
		ray.baseRay.currentGeometryID=geometryID;
		if (ray.baseRay.depth<MAX_DEPTH_CPU && ray.baseRay.flux>MIN_FLUX_CPU)
		{			
			oGroup.trace(ray);
		}
}

/**
 * \detail createOptiXInstance
 *
 * \param[in] RTcontext context, RTgeometryinstance &instance, int index, TraceMode mode, double lambda
 * 
 * \return MaterialError
 * \sa 
 * \remarks we have a seperate .cu file for each combination of material, scatter and coating. Therefore we set the path to that ptx file that corresponds to the combination present in the current instance
 * \author Mauch
 */
MaterialError MaterialReflecting_CovGlass::createOptiXInstance(RTcontext context, RTgeometryinstance &instance, int index, SimParams simParams, double lambda)
{
	if (MAT_NO_ERR != Material::createOptiXInstance(context, instance, index, simParams, lambda) )
	{
		std::cout << "error in MaterialReflecting_CovGlass.createOptiXInstance(): Material.creatOptiXInstance() returned an error" << "...\n";
		return MAT_ERR;
	}
	/* set the variables of the geometry */
	if ( !RT_CHECK_ERROR_NOEXIT( rtProgramDeclareVariable( closest_hit_program, "params", &l_params ), context) )
		return MAT_ERR;
	if ( !RT_CHECK_ERROR_NOEXIT( rtVariableSetUserData(l_params, sizeof(MatReflecting_CovGlass_params), &(this->params)), context) )
		return MAT_ERR;

	return MAT_NO_ERR;	
};

/**
 * \detail updateOptiXInstance 
 *
 * \param[in] RTcontext context, RTgeometryinstance &instance, int index, TraceMode mode, double lambda
 * 
 * \return MaterialError
 * \sa 
 * \remarks 
 * \author Mauch
 */
MaterialError MaterialReflecting_CovGlass::updateOptiXInstance(RTcontext context, RTgeometryinstance &instance, int index, SimParams simParams, double lambda)
{
	if (this->update)
	{
		/* set the variables of the geometry */
		if ( !RT_CHECK_ERROR_NOEXIT( rtProgramQueryVariable( closest_hit_program, "params", &l_params ), context) )
			return MAT_ERR;
		if ( !RT_CHECK_ERROR_NOEXIT( rtVariableSetUserData(l_params, sizeof(MatReflecting_CovGlass_params), &(this->params)), context) )
			return MAT_ERR;

	}

	if (MAT_NO_ERR != Material::updateOptiXInstance(context, instance, index, simParams, lambda) )
	{
		std::cout << "error in MaterialReflecting_CovGlass.updateOptiXInstance(): Material.updateOptiXInstance() returned an error" << "...\n";
		return MAT_ERR;
	}

	return MAT_NO_ERR;	
};

/**
 * \detail processParseResults 
 *
 * sets the parameters of the detector according to the given parse results
 *
 * \param[in] MaterialParseParamStruct &parseResults_Mat
 * 
 * \return detError
 * \sa 
 * \remarks 
 * \author Mauch
 */
MaterialError MaterialReflecting_CovGlass::processParseResults(MaterialParseParamStruct &parseResults_Mat, DetectorParseParamStruct &parseResults_Det)
{
	this->params.geomID=parseResults_Det.geomID;
	if (parseResults_Mat.coating_r <= 0)
	{
		std::cout << "error in MaterialReflecting_CovGlass.processParseResults(): cover glass material is not allowed with an reflection coefficient equal to zero" << "...\n";
		return MAT_ERR;
	}
	else
	{
		this->params.r=parseResults_Mat.coating_r;
		this->params.t=parseResults_Mat.coating_t;
	}
	return MAT_NO_ERR;
};
