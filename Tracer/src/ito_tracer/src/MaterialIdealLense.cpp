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

/**\file MaterialIdealLense.cpp
* \brief material of ideal lense
* 
*           
* \author Mauch
*/

#include "MaterialIdealLense.h"
#include "GlobalConstants.h"
#include "myUtil.h"
#include "sampleConfig.h"
#include <iostream>
#include <string.h>
#include "rayTracingMath.h"


/**
 * \detail hit function of material for geometric rays
 *
 * we call hitIdealLense that describes the interaction of the ray with the material and can be called from GPU as well. Then we call the hit function of the coatin attached to the Material. Finally we call the hit function of the Scatter attached to the material
 *
 * \param[in] rayStruct &ray, double3 normal, double t_hit, int geometryID
 * 
 * \return MaterialError
 * \sa 
 * \remarks 
 * \author Mauch
 */
void MaterialIdealLense::hit(rayStruct &ray, Mat_hitParams hitParams, double t_hit, int geometryID)
{
	
//	extern Group oGroup;
	bool coat_reflected=false;
	if ( this->coatingPtr->getFullParams()->type != CT_NOCOATING )
		coat_reflected=this->coatingPtr->hit(ray, hitParams);

	if ( hitIdealLense(ray, hitParams, this->params, t_hit, geometryID, coat_reflected) )
	{
		if ( this->scatterPtr->getFullParams()->type != ST_NOSCATTER )
			this->scatterPtr->hit(ray, hitParams);
		if (ray.depth>MAX_DEPTH_CPU || ray.flux<MIN_FLUX_CPU)
			ray.running=false;//stop ray
	}

}

/**
 * \detail hit function of material for gaussian beam rays
 *
 * we repeatedly call hitAbsorbing that describes the interaction of geometric rays with the material and can be called from GPU as well. Then we call the hit function of the coatin attached to the Material. Finally we call the hit function of the Scatter attached to the material
 *
 * \param[in] gaussBeamRayStruct &ray, gaussBeam_geometricNormal normal, int geometryID
 * 
 * \return MaterialError
 * \sa 
 * \remarks not implemented yet
 * \author Mauch
 */
void MaterialIdealLense::hit(gaussBeamRayStruct &ray, gaussBeam_geometricNormal normal, int geometryID)
{
	extern Group oGroup;

	ray.baseRay.currentGeometryID=geometryID;
	if (ray.baseRay.depth<MAX_DEPTH_CPU && ray.baseRay.flux>MIN_FLUX_CPU)
	{			
		oGroup.trace(ray);
	}
}

/**
 * \detail createOptiXInstance
 *
 * we repeatedly call hitAbsorbing that describes the interaction of geometric rays with the material and can be called from GPU as well. Then we call the hit function of the coatin attached to the Material. Finally we call the hit function of the Scatter attached to the material
 *
 * \param[in] gaussBeamRayStruct &ray, gaussBeam_geometricNormal normal, int geometryID
 * 
 * \return MaterialError
 * \sa 
 * \remarks not implemented yet
 * \author Mauch
 */
MaterialError MaterialIdealLense::createOptiXInstance(RTcontext context, RTgeometryinstance &instance, int index, simMode mode, double lambda)
{
	if (MAT_NO_ERR != Material::createOptiXInstance(context, instance, index, mode, lambda) )
	{
		std::cout << "error in MaterialIdealLense.createOptiXInstance(): Material.creatOptiXInstance() returned an error" << std::endl;
		return MAT_ERR;
	}

	lambda_old=lambda; // when creating the OptiXInstance we need to do this

	// calc the refractive indices at current wavelength
	if (MAT_NO_ERR != calcFocalLength(lambda))
	{
		std::cout << "error in MaterialIdealLense.createOptixInstance(): calcFocalLength returned an error" << std::endl;
		return MAT_ERR;
	}
	this->params.orientation=this->dispersionParamsPtr->orientation;
	this->params.root=this->dispersionParamsPtr->root;
	/* set the variables of the material */
	if ( !RT_CHECK_ERROR_NOEXIT( rtProgramDeclareVariable( closest_hit_program, "params", &l_params ), context) )
		return MAT_ERR;
	if ( !RT_CHECK_ERROR_NOEXIT( rtVariableSetUserData(l_params, sizeof(MatIdealLense_params), &(this->params)), context) )
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
MaterialError MaterialIdealLense::updateOptiXInstance(RTcontext context, RTgeometryinstance &instance, int index, simMode mode, double lambda)
{

	if  ( (this->update==true)||(this->lambda_old!=lambda) )
	{
		// calc the refractive indices at current wavelength
		if (MAT_NO_ERR != calcFocalLength(lambda))
		{
			std::cout << "error in MaterialIdealLense.updateOptixInstance(): calcFocalLength returned an error" << std::endl;
			return MAT_ERR;
		}
		this->params.orientation=this->dispersionParamsPtr->orientation;
		this->params.root=this->dispersionParamsPtr->root;
		/* set the variables of the geometry */
		if ( !RT_CHECK_ERROR_NOEXIT( rtProgramQueryVariable( closest_hit_program, "params", &l_params ), context) )
			return MAT_ERR;
		if ( !RT_CHECK_ERROR_NOEXIT( rtVariableSetUserData(l_params, sizeof(MatIdealLense_params), &(this->params)), context) )
			return MAT_ERR;

		this->update=false;
	}

	if (MAT_NO_ERR != Material::updateOptiXInstance(context, instance, index, mode, lambda) )
	{
		std::cout << "error in MaterialIdealLense.updateOptiXInstance(): Material.updateOptiXInstance() returned an error" << std::endl;
		return MAT_ERR;
	}

	return MAT_NO_ERR;	
};

/**
 * \detail updateCPUSimInstance 
 *
 * \param[in] RTcontext context, RTgeometryinstance &instance, int index, simMode mode, double lambda
 * 
 * \return MaterialError
 * \sa 
 * \remarks 
 * \author Mauch
 */
MaterialError MaterialIdealLense::updateCPUSimInstance(double lambda)
{
	if ( (this->coatingPtr->update)||(this->lambda_old!=lambda) )
	{
		// create simulation instance of coating
		this->coatingPtr->createCPUSimInstance(lambda);

		this->coatingPtr->update=false;
	}
	if ( (this->scatterPtr->update==true)||(this->lambda_old!=lambda) )
	{
		this->scatterPtr->createCPUSimInstance(lambda);

		this->scatterPtr->update=false;
	}
	if  ( (this->update==true)||(this->lambda_old!=lambda) )
	{
		// calc the refractive indices at current wavelength
		if (MAT_NO_ERR != calcFocalLength(lambda))
		{
			std::cout << "error in MaterialIdealLense.updateOptixInstance(): calcFocalLength returned an error" << std::endl;
			return MAT_ERR;
		}
		this->params.orientation=this->dispersionParamsPtr->orientation;
		this->params.root=this->dispersionParamsPtr->root;

		this->update=false;
	}
	lambda_old=lambda;

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

void MaterialIdealLense::setParams(MatIdealLense_params params)
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
MatIdealLense_params MaterialIdealLense::getParams(void)
{
	return this->params;
}

/**
 * \detail setDispersionParams 
 *
 * \param[in] MatIdealLense_DispersionParams* paramsInPtr
 * 
 * \return void
 * \sa 
 * \remarks 
 * \author Mauch
 */
void MaterialIdealLense::setDispersionParams(MatIdealLense_DispersionParams* paramsInPtr)
{
	*(this->dispersionParamsPtr)=*(paramsInPtr);
}

/**
 * \detail getDispersionParams 
 *
 * \param[in] void
 * 
 * \return MatIdealLense_DispersionParams*
 * \sa 
 * \remarks 
 * \author Mauch
 */
MatIdealLense_DispersionParams* MaterialIdealLense::getDispersionParams(void)
{
	return this->dispersionParamsPtr;
}

/**
 * \detail calcFocalLength 
 *
 * \param[in] double lambda
 * 
 * \return MaterialError
 * \sa 
 * \remarks 
 * \author Mauch
 */
MaterialError MaterialIdealLense::calcFocalLength(double lambda)
{
	//this->params.f=this->dispersionParamsPtr->f0+this->dispersionParamsPtr->A*(2*PI)/(this->dispersionParamsPtr->lambda0-lambda);
	if (this->dispersionParamsPtr->A!=0)
		this->params.f=this->dispersionParamsPtr->A/(lambda*1e3);
	else
		this->params.f=this->dispersionParamsPtr->f0;
	if (this->dispersionParamsPtr->apertureHalfWidth.x >= this->dispersionParamsPtr->apertureHalfWidth.y)
		this->params.thickness=sqrt(pow(this->dispersionParamsPtr->apertureHalfWidth.x,2)+pow(this->params.f,2));
	else
		this->params.thickness=sqrt(pow(this->dispersionParamsPtr->apertureHalfWidth.y,2)+pow(this->params.f,2));
	return MAT_NO_ERR;
};

/**
 * \detail createCPUSimInstance 
 *
 * \param[in] RTcontext context, RTgeometryinstance &instance, int index, simMode mode, double lambda
 * 
 * \return MaterialError
 * \sa 
 * \remarks 
 * \author Mauch
 */
MaterialError MaterialIdealLense::createCPUSimInstance(double lambda)
{
	// calc the refractive indices at current wavelength
	if (MAT_NO_ERR!=this->calcFocalLength(lambda))
	{
		std::cout << "error in MaterialIdealLense.createCPUSimInstance(): calcFocalLength returned an error" << std::endl;
		return MAT_ERR;
	}
	this->params.orientation=this->dispersionParamsPtr->orientation;
	this->params.root=this->dispersionParamsPtr->root;

	// create simulation instance of coating
	if (MAT_NO_ERR != Material::createCPUSimInstance(lambda) )
	{
		std::cout << "error in MaterialIdealLense.createCPUSimInstance(): Material.createCPUSimInstance() returned an error." << std::endl;
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
MaterialError MaterialIdealLense::processParseResults(MaterialParseParamStruct &parseResults_Mat)
{
	this->dispersionParamsPtr=new MatIdealLense_DispersionParams;
	this->dispersionParamsPtr->orientation=parseResults_Mat.normal;
	this->dispersionParamsPtr->root=parseResults_Mat.root;
	this->dispersionParamsPtr->A=parseResults_Mat.idealLense_A;
	this->dispersionParamsPtr->f0=parseResults_Mat.idealLense_f0;
	this->dispersionParamsPtr->lambda0=parseResults_Mat.idealLense_lambda0;
	this->dispersionParamsPtr->apertureHalfWidth=parseResults_Mat.apertureHalfWidth;
	
	return MAT_NO_ERR;
}

/**
 * \detail parseXml 
 *
 * sets the parameters of the material according to the given xml node
 *
 * \param[in] xml_node &geometry
 * 
 * \return geometryError
 * \sa 
 * \remarks 
 * \author Mauch
 */
MaterialError MaterialIdealLense::parseXml(pugi::xml_node &geometry)
{
	if (!Material::parseXml(geometry))
	{
		std::cout << "error in MaterialIdealLense.parseXml(): Material.parseXml() returned an error." << std::endl;
		return MAT_ERR;
	}

	return MAT_NO_ERR;
};
