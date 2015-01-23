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

/**\file MaterialLight_GeomRender.cpp
* \brief refracting material
* 
*           
* \author Mauch
*/

#include "MaterialLight_GeomRender.h"
#include "..\GlobalConstants.h"
#include "..\myUtil.h"
#include "sampleConfig.h"
#include <iostream>
#include <string.h>
#include "..\rayTracingMath.h"

#include "..\Parser_XML.h"


/**
 * \detail hit function of material for geometric rays
 *
 * we call the hit function of the coating first. Then we call the hit function of the material. Finally we call the hit function of the Scatter attached to the material
 *
 * \param[in] rayStruct &ray, double3 normal, double t_hit, int geometryID
 * 
 * \return MaterialError
 * \sa 
 * \remarks 
 * \author Mauch
 */
void MaterialLight_GeomRender::hit(rayStruct &ray, Mat_hitParams hitParams, double t_hit, int geometryID)
{
	
//	extern Group oGroup;
    geomRenderRayStruct *l_pRay=reinterpret_cast<geomRenderRayStruct*>(&ray);
	if ( !hitRenderLight(*l_pRay, hitParams, this->params, t_hit, geometryID) )
        cout << "error in MaterialLight_GeomRender.hit(): a ray is immersed in " << ray.nImmersed << ". It has hit the geometry " << geometryID << " which is not immersed in that medium!! ray will be stopped." << endl;		

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
MaterialError MaterialLight_GeomRender::createCPUSimInstance(double lambda)
{
	// create simulation instance of coating
	if (MAT_NO_ERR != Material::createCPUSimInstance(lambda) )
	{
		std::cout << "error in MaterialLight_GeomRender.createCPUSimInstance(): Material.createCPUSimInstance() returned an error." << "...\n";
		return MAT_ERR;
	}

	return MAT_NO_ERR;
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
MaterialError MaterialLight_GeomRender::createOptiXInstance(RTcontext context, RTgeometryinstance &instance, int index, SimParams simParams, double lambda)
{
	if (MAT_NO_ERR != Material::createOptiXInstance(context, instance, index, simParams, lambda) )
	{
		std::cout << "error in MaterialLight_GeomRender.createOptiXInstance(): Material.creatOptiXInstance() returned an error" << "...\n";
		return MAT_ERR;
	}

	lambda_old=lambda; // when creating the OptiXInstance we need to do this

	/* set the variables of the geometry */
	if ( !RT_CHECK_ERROR_NOEXIT( rtProgramDeclareVariable( closest_hit_program, "params", &l_params ), context) )
		return MAT_ERR;
	if ( !RT_CHECK_ERROR_NOEXIT( rtVariableSetUserData(l_params, sizeof(MatLight_GeomRender_params), &(this->params)), context) )
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
MaterialError MaterialLight_GeomRender::updateOptiXInstance(RTcontext context, RTgeometryinstance &instance, int index, SimParams simParams, double lambda)
{
	if ( (this->update)||(this->lambda_old!=lambda) )
	{
		/* set the variables of the geometry */
		if ( !RT_CHECK_ERROR_NOEXIT( rtProgramQueryVariable( closest_hit_program, "params", &l_params ), context) )
			return MAT_ERR;
		if ( !RT_CHECK_ERROR_NOEXIT( rtVariableSetUserData(l_params, sizeof(MatRefracting_params), &(this->params)), context) )
			return MAT_ERR;

		this->update=false;
	}

	if (MAT_NO_ERR != Material::updateOptiXInstance(context, instance, index, simParams, lambda) )
	{
		std::cout << "error in MaterialLight_GeomRender.updateOptiXInstance(): Material.updateOptiXInstance() returned an error" << "...\n";
		return MAT_ERR;
	}

	return MAT_NO_ERR;	
};


/**
 * \detail setParams 
 *
 * \param[in] MatRefracting_params params
 * 
 * \return void
 * \sa 
 * \remarks 
 * \author Mauch
 */
void MaterialLight_GeomRender::setParams(MatLight_GeomRender_params params)
{
	this->update=true;
	this->params=params;
}

/**
 * \detail getParams 
 *
 * \param[in] void
 * 
 * \return MatRefractingParams
 * \sa 
 * \remarks 
 * \author Mauch
 */
MatLight_GeomRender_params MaterialLight_GeomRender::getParams(void)
{
	return this->params;
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
MaterialError MaterialLight_GeomRender::parseXml(pugi::xml_node &material, SimParams simParams)
{
	if (!Material::parseXml(material, simParams))
	{
		std::cout << "error in MaterialLight_GeomRender.parseXml(): Material.parseXml() returned an error." << "...\n";
		return MAT_ERR;
	}

    Parser_XML l_parser;
	if (!this->checkParserError(l_parser.attrByNameToDouble(material, "power", this->params.power)))
		return MAT_ERR;
    if (!this->checkParserError(l_parser.attrByNameToDouble(material, "pupilRoot.x", this->params.pupilRoot.x)))
		return MAT_ERR;
    if (!this->checkParserError(l_parser.attrByNameToDouble(material, "pupilRoot.y", this->params.pupilRoot.y)))
		return MAT_ERR;
    if (!this->checkParserError(l_parser.attrByNameToDouble(material, "pupilRoot.z", this->params.pupilRoot.z)))
		return MAT_ERR;

    if (!this->checkParserError(l_parser.attrByNameToDouble(material, "pupilTilt.x", this->params.pupilTilt.x)))
		return MAT_ERR;
    if (!this->checkParserError(l_parser.attrByNameToDouble(material, "pupilTilt.y", this->params.pupilTilt.y)))
		return MAT_ERR;
    if (!this->checkParserError(l_parser.attrByNameToDouble(material, "pupilTilt.z", this->params.pupilTilt.z)))
		return MAT_ERR;

    if (!this->checkParserError(l_parser.attrByNameToDouble(material, "pupilAptRad.x", this->params.pupilAptRad.x)))
		return MAT_ERR;
    if (!this->checkParserError(l_parser.attrByNameToDouble(material, "pupilAptRad.y", this->params.pupilAptRad.y)))
		return MAT_ERR;



	return MAT_NO_ERR;
}