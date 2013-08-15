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

/**\file MaterialPathTraceSource.cpp
* \brief reflecting material
* 
*           
* \author Mauch
*/

#include "MaterialPathTraceSource.h"
#include "GlobalConstants.h"
#include "myUtil.h"
#include "sampleConfig.h"
#include <iostream>
#include <string.h>


/**
 * \detail hit function of material for geometric rays
 *
 * Here we need to call the hit function of the coating first. If the coating transmits the ray through the material it passes without any further deflectiosn. Finally we call the hit function of the Scatter attached to the material
 *
 * \param[in] rayStruct_PathTracing &ray, double3 normal, double t_hit, int geometryID
 * 
 * \return MaterialError
 * \sa 
 * \remarks 
 * \author Mauch
 */
void MaterialPathTraceSource::hit(rayStruct &ray, Mat_hitParams hitParams, double t_hit, int geometryID)
{
	rayStruct_PathTracing* ray_interpreted;
	// we have a path tracing ray here ... hopefully ...
	ray_interpreted=reinterpret_cast<rayStruct_PathTracing*>(&ray);

	double3 n=hitParams.normal;
	bool coat_reflected = true;
	if ( this->coatingPtr->getFullParams()->type != CT_NOCOATING )
		coat_reflected=this->coatingPtr->hit(*ray_interpreted, hitParams);
	// if the coating wants transmission we do not change the ray direction at all !!!
	if (coat_reflected)
		hitPathTraceSource(*ray_interpreted, hitParams, this->params, t_hit, geometryID);
	if ( this->scatterPtr->getFullParams()->type != ST_NOSCATTER )
		this->scatterPtr->hit(*ray_interpreted, hitParams);

	if (ray.depth>MAX_DEPTH_CPU || ray.flux<MIN_FLUX_CPU)
		ray.running=false;//stop ray
}


/**
 * \detail createOptiXInstance
 *
 * \param[in] RTcontext context, RTgeometryinstance &instance, int index, simMode mode, double lambda
 * 
 * \return MaterialError
 * \sa 
 * \remarks we have a seperate .cu file for each combination of material, scatter and coating. Therefore we set the path to that ptx file that corresponds to the combination present in the current instance
 * \author Mauch
 */
MaterialError MaterialPathTraceSource::createOptiXInstance(RTcontext context, RTgeometryinstance &instance, int index, simMode mode, double lambda)
{
	if (MAT_NO_ERR != Material::createOptiXInstance(context, instance, index, mode, lambda) )
	{
		std::cout << "error in MaterialPathTraceSource.createOptiXInstance(): Material.creatOptiXInstance() returned an error" << std::endl;
		return MAT_ERR;
	}
	/* set the variables of the geometry */
	if ( !RT_CHECK_ERROR_NOEXIT( rtProgramDeclareVariable( closest_hit_program, "params", &l_params ), context) )
		return MAT_ERR;
	if ( !RT_CHECK_ERROR_NOEXIT( rtVariableSetUserData(l_params, sizeof(MatPathTraceSource_params), &(this->params)), context) )
		return MAT_ERR;

	return MAT_NO_ERR;	
};

/**
 * \detail updateOptiXInstance 
 *
 * \param[in] RTcontext context, RTgeometryinstance &instance, int index, simMode mode, double lambda
 * 
 * \return MaterialError
 * \sa 
 * \remarks 
 * \author Mauch
 */
MaterialError MaterialPathTraceSource::updateOptiXInstance(RTcontext context, RTgeometryinstance &instance, int index, simMode mode, double lambda)
{
	if (this->update)
	{
		/* set the variables of the geometry */
		if ( !RT_CHECK_ERROR_NOEXIT( rtProgramQueryVariable( closest_hit_program, "params", &l_params ), context) )
			return MAT_ERR;
		if ( !RT_CHECK_ERROR_NOEXIT( rtVariableSetUserData(l_params, sizeof(MatPathTraceSource_params), &(this->params)), context) )
			return MAT_ERR;

		this->update=false;
	}
	if (MAT_NO_ERR != Material::updateOptiXInstance(context, instance, index, mode, lambda) )
	{
		std::cout << "error in MaterialLinearGrating1D.updateOptiXInstance(): Material.updateOptiXInstance() returned an error" << std::endl;
		return MAT_ERR;
	}
	return MAT_NO_ERR;	
};

/**
 * \detail setParams 
 *
 * \param[in] MatIdealLense_params params
 * 
 * \return void
 * \sa 
 * \remarks 
 * \author Mauch
 */

void MaterialPathTraceSource::setParams(MatPathTraceSource_params params)
{
	this->update=true;
	this->params=params;
}

/**
 * \detail getParams 
 *
 * \param[in] void
 * 
 * \return MatIdealLense_params
 * \sa 
 * \remarks 
 * \author Mauch
 */
MatPathTraceSource_params MaterialPathTraceSource::getParams(void)
{
	return this->params;
}

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
MaterialError MaterialPathTraceSource::processParseResults(MaterialParseParamStruct &parseResults_Mat)
{

	this->params.acceptanceAngleMax=parseResults_Mat.importanceConeAlphaMax;
	this->params.acceptanceAngleMin=parseResults_Mat.importanceConeAlphaMin;
	this->params.tilt=parseResults_Mat.tilt;
	this->params.flux=parseResults_Mat.flux;
	// Theres really nothing to do here...
	return MAT_NO_ERR;
};
