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

/**\file MaterialFilter.cpp
* \brief absorbing material
* 
*           
* \author Mauch
*/

#include "MaterialFilter.h"
#include "myUtil.h"
#include "sampleConfig.h"
#include <iostream>
#include <string.h>


//void MaterialFilter::setPathToPtx(char* path)
//{
//	memcpy(this->path_to_ptx, path, sizeof(this->path_to_ptx));	
//};
//
//char* MaterialFilter::getPathToPtx(void)
//{
//	return this->path_to_ptx;
//};

/**
 * \detail hit function of material for geometric rays
 *
 * we call hitAbsorbing that describes the interaction of the ray with the material and can be called from GPU as well. Then we call the hit function of the coatin attached to the Material. Finally we call the hit function of the Scatter attached to the material
 *
 * \param[in] rayStruct &ray, double3 normal, double t_hit, int geometryID
 * 
 * \return MaterialError
 * \sa 
 * \remarks 
 * \author Mauch
 */
void MaterialFilter::hit(rayStruct &ray, Mat_hitParams hitParams, double t_hit, int geometryID)
{
	//ray.direction=make_double3(0.0);
	if ( hitFilter(ray, hitParams, this->params, t_hit, geometryID) )
	{
		bool reflected;
		if ( this->coatingPtr->getFullParams()->type != CT_NOCOATING )
		{
			reflected=this->coatingPtr->hit(ray, hitParams);
			if (reflected)
			{
				ray.direction=reflect(ray.direction, hitParams.normal);
				ray.running=true; // keep ray alive
			}
		}

		if ( this->scatterPtr->getFullParams()->type != ST_NOSCATTER )
		{
			this->scatterPtr->hit(ray, hitParams);
			ray.running=true; // keep ray alive
		}
	}
}

/**
 * \detail setParams 
 *
 * \param[in] MatFilter_params params
 * 
 * \return void
 * \sa 
 * \remarks 
 * \author Mauch
 */

void MaterialFilter::setParams(MatFilter_params params)
{
	this->update=true;
	this->params=params;
}

/**
 * \detail getParams 
 *
 * \param[in] void
 * 
 * \return MatFilter_params
 * \sa 
 * \remarks 
 * \author Mauch
 */
MatFilter_params MaterialFilter::getParams(void)
{
	return this->params;
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
 * \remarks not tested yet
 * \author Mauch
 */
void MaterialFilter::hit(gaussBeamRayStruct &ray, gaussBeam_geometricNormal normal, int geometryID)
{
	//ray.direction=make_double3(0.0);
	ray.baseRay.running=false;
	ray.baseRay.currentGeometryID=geometryID;
}

/**
 * \detail createCPUSimInstance
 *
 * \param[in] double lambda
 * 
 * \return MaterialError
 * \sa 
 * \remarks 
 * \author Mauch
 */
MaterialError MaterialFilter::createCPUSimInstance(double lambda)
{
	// calc the refractive indices at current wavelength
	if (MAT_NO_ERR!=this->calcFilter(lambda))
	{
		std::cout << "error in MaterialFilter.createCPUSimInstance(): calcFilter returned an error" << std::endl;
		return MAT_ERR;
	}

	// create simulation instance of coating
	if (MAT_NO_ERR != Material::createCPUSimInstance(lambda) )
	{
		std::cout << "error in MaterialFilter.createCPUSimInstance(): Material.createCPUSimInstance() returned an error." << std::endl;
		return MAT_ERR;
	}

	return MAT_NO_ERR;
};

/**
 * \detail setDispersionParams 
 *
 * \param[in] MatFilter_DispersionParams* paramsInPtr
 * 
 * \return void
 * \sa 
 * \remarks 
 * \author Mauch
 */
void MaterialFilter::setDispersionParams(MatFilter_DispersionParams* paramsInPtr)
{
	*(this->dispersionParamsPtr)=*(paramsInPtr);
}

/**
 * \detail getDispersionParams 
 *
 * \param[in] void
 * 
 * \return MatFilter_DispersionParams*
 * \sa 
 * \remarks 
 * \author Mauch
 */
MatFilter_DispersionParams* MaterialFilter::getDispersionParams(void)
{
	return this->dispersionParamsPtr;
}

/**
 * \detail updateCPUSimInstance 
 *
 * \param[in] RTcontext context, RTgeometryinstance &instance, int index, TraceMode mode, double lambda
 * 
 * \return MaterialError
 * \sa 
 * \remarks 
 * \author Mauch
 */
MaterialError MaterialFilter::updateCPUSimInstance(double lambda)
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
		if (MAT_NO_ERR != calcFilter(lambda))
		{
			std::cout << "error in MaterialFilter.updateOptixInstance(): calcFilter returned an error" << std::endl;
			return MAT_ERR;
		}

		this->update=false;
	}
	lambda_old=lambda;

	return MAT_NO_ERR;	
};

/**
 * \detail calcFilter 
 *
 * \param[in] double lambda
 * 
 * \return MaterialError
 * \sa 
 * \remarks 
 * \author Mauch
 */
MaterialError MaterialFilter::calcFilter(double lambda)
{
	if ( (lambda>this->dispersionParamsPtr->lambdaMin) && (lambda<this->dispersionParamsPtr->lambdaMax) )
		this->params.absorb=false;
	else
		this->params.absorb=true;
	return MAT_NO_ERR;
};

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
MaterialError MaterialFilter::createOptiXInstance(RTcontext context, RTgeometryinstance &instance, int index, SimParams simParams, double lambda)
{
	if (MAT_NO_ERR != Material::createOptiXInstance(context, instance, index, simParams, lambda) )
	{
		std::cout << "error in MaterialFilter.createOptiXInstance(): Material.creatOptiXInstance() returned an error" << std::endl;
		return MAT_ERR;
	}

	lambda_old=lambda;

	// calc the refractive indices at current wavelength
	if (MAT_NO_ERR != calcFilter(lambda))
	{
		std::cout << "error in MaterialFilter.createOptixInstance(): calcFilter returned an error" << std::endl;
		return MAT_ERR;
	}

	/* set the variables of the geometry */
	if ( !RT_CHECK_ERROR_NOEXIT( rtProgramDeclareVariable( closest_hit_program, "params", &l_params ), context) )
		return MAT_ERR;
	if ( !RT_CHECK_ERROR_NOEXIT( rtVariableSetUserData(l_params, sizeof(MatFilter_params), &(this->params)), context) )
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
MaterialError MaterialFilter::updateOptiXInstance(RTcontext context, RTgeometryinstance &instance, int index, SimParams simParams, double lambda)
{
	if ( (this->update)||(this->lambda_old!=lambda) )
	{
		// calc the refractive indices at current wavelength
		if (MAT_NO_ERR != calcFilter(lambda))
		{
			std::cout << "error in MaterialFilter.updateOptixInstance(): calcFilter returned an error" << std::endl;
			return MAT_ERR;
		}
		/* set the variables of the geometry */
		if ( !RT_CHECK_ERROR_NOEXIT( rtProgramQueryVariable( closest_hit_program, "params", &l_params ), context) )
			return MAT_ERR;
		if ( !RT_CHECK_ERROR_NOEXIT( rtVariableSetUserData(l_params, sizeof(MatFilter_params), &(this->params)), context) )
			return MAT_ERR;

		this->update=false;
	}

	if (MAT_NO_ERR != Material::updateOptiXInstance(context, instance, index, simParams, lambda) )
	{
		std::cout << "error in MaterialFilter.updateOptiXInstance(): Material.updateOptiXInstance() returned an error" << std::endl;
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
MaterialError MaterialFilter::processParseResults(MaterialParseParamStruct &parseResults_Mat)
{
	this->dispersionParamsPtr=new MatFilter_DispersionParams;
	this->dispersionParamsPtr->lambdaMax=parseResults_Mat.filterMax*1e-3;
	this->dispersionParamsPtr->lambdaMin=parseResults_Mat.filterMin*1e-3;
	// Theres really nothing to do here...
	return MAT_NO_ERR;
};