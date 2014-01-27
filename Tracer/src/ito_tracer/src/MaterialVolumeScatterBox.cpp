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

/**\file MaterialVolumeScatterBox.cpp
* \brief refracting material
* 
*           
* \author Mauch
*/

#include "MaterialVolumeScatterBox.h"
#include "GlobalConstants.h"
#include "myUtil.h"
#include "sampleConfig.h"
#include <iostream>
#include <string.h>
#include "rayTracingMath.h"

#include "Parser_XML.h"
#include "Parser.h"


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
void MaterialVolumeScatterBox::hit(rayStruct &ray, Mat_hitParams hitParams, double t_hit, int geometryID)
{
	
//	extern Group oGroup;
	bool coat_reflected=false;
	if ( this->coatingPtr->getFullParams()->type != CT_NOCOATING )
		coat_reflected=this->coatingPtr->hit(ray, hitParams);

	if ( hitVolumeScatterBox(ray, hitParams, this->params, t_hit, geometryID, coat_reflected) )
	{
		if ( this->scatterPtr->getFullParams()->type != ST_NOSCATTER )
			this->scatterPtr->hit(ray, hitParams);
		//if (ray.depth<MAX_DEPTH_CPU && ray.flux>MIN_FLUX_CPU)
		//{			
		//	oGroup.trace(ray);
		//}
		if (ray.depth>MAX_DEPTH_CPU || ray.flux<MIN_FLUX_CPU)
			ray.running=false;//stop ray

	}

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
MaterialError MaterialVolumeScatterBox::createCPUSimInstance(double lambda)
{

	// create simulation instance of coating
	if (MAT_NO_ERR != Material::createCPUSimInstance(lambda) )
	{
		std::cout << "error in MaterialVolumeScatterBox.createCPUSimInstance(): Material.createCPUSimInstance() returned an error." << std::endl;
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
MaterialError MaterialVolumeScatterBox::createOptiXInstance(RTcontext context, RTgeometryinstance &instance, int index, SimParams simParams, double lambda)
{
	if (MAT_NO_ERR != Material::createOptiXInstance(context, instance, index, simParams, lambda) )
	{
		std::cout << "error in MaterialVolumeScatterBox.createOptiXInstance(): Material.creatOptiXInstance() returned an error" << std::endl;
		return MAT_ERR;
	}

	lambda_old=lambda; // when creating the OptiXInstance we need to do this

	/* set the variables of the geometry */
	if ( !RT_CHECK_ERROR_NOEXIT( rtProgramDeclareVariable( closest_hit_program, "params", &l_params ), context) )
		return MAT_ERR;
	if ( !RT_CHECK_ERROR_NOEXIT( rtVariableSetUserData(l_params, sizeof(MatRefracting_params), &(this->params)), context) )
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
MaterialError MaterialVolumeScatterBox::updateOptiXInstance(RTcontext context, RTgeometryinstance &instance, int index, SimParams simParams, double lambda)
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
		std::cout << "error in MaterialVolumeScatterBox.updateOptiXInstance(): Material.updateOptiXInstance() returned an error" << std::endl;
		return MAT_ERR;
	}

	return MAT_NO_ERR;	
};


/**
 * \detail setParams 
 *
 * \param[in] MatVolumeScatterParams params
 * 
 * \return void
 * \sa 
 * \remarks 
 * \author Mauch
 */
void MaterialVolumeScatterBox::setParams(MatVolumeScatterBox_params params)
{
	this->update=true;
	this->params=params;
}

/**
 * \detail getParams 
 *
 * \param[in] void
 * 
 * \return MatVolumeScatterParams
 * \sa 
 * \remarks 
 * \author Mauch
 */
MatVolumeScatterBox_params MaterialVolumeScatterBox::getParams(void)
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
MaterialError MaterialVolumeScatterBox::parseXml(pugi::xml_node &material, SimParams simParams)
{
	if (!Material::parseXml(material, simParams))
	{
		std::cout << "error in MaterialVolumeScatterBox.parseXml(): Material.parseXml() returned an error." << std::endl;
		return MAT_ERR;
	}

	Parser_XML l_parser;

	if (!this->checkParserError(l_parser.attrByNameToDouble(material, "n1", this->params.n1)))
		return MAT_ERR;
	if (!this->checkParserError(l_parser.attrByNameToDouble(material, "n2", this->params.n2)))
		return MAT_ERR;
	if (!this->checkParserError(l_parser.attrByNameToDouble(material, "meanFreePath", this->params.meanFreePath)))
		return MAT_ERR;
	double l_g;
	if (!this->checkParserError(l_parser.attrByNameToDouble(material, "anisotropyFac", l_g)))
		return MAT_ERR;
	this->params.g=l_g/360*2*M_PI;
	if (!this->checkParserError(l_parser.attrByNameToDouble(material, "absorptionCoeff", this->params.absorptionCoeff)))
		return MAT_ERR;
	if (!this->checkParserError(l_parser.attrByNameToInt(material, "maxNrBounces", this->params.maxNrBounces)))
		return MAT_ERR;
	if (!this->checkParserError(l_parser.attrByNameToDouble(material, "apertureRadius.x", this->params.aprtRadius.x)))
		return MAT_ERR;
	if (!this->checkParserError(l_parser.attrByNameToDouble(material, "apertureRadius.y", this->params.aprtRadius.y)))
		return MAT_ERR;
	if (!this->checkParserError(l_parser.attrByNameToDouble(material, "thickness", this->params.thickness)))
		return MAT_ERR;
	if (!this->checkParserError(l_parser.attrByNameToDouble(material, "root.x", this->params.root.x)))
		return MAT_ERR;
	if (!this->checkParserError(l_parser.attrByNameToDouble(material, "root.y", this->params.root.y)))
		return MAT_ERR;
	if (!this->checkParserError(l_parser.attrByNameToDouble(material, "root.z", this->params.root.z)))
		return MAT_ERR;
	if (!this->checkParserError(l_parser.attrByNameToDouble(material, "tilt.x", this->params.tilt.x)))
		return MAT_ERR;
	if (!this->checkParserError(l_parser.attrByNameToDouble(material, "tilt.y", this->params.tilt.y)))
		return MAT_ERR;
	if (!this->checkParserError(l_parser.attrByNameToDouble(material, "tilt.z", this->params.tilt.z)))
		return MAT_ERR;


	return MAT_NO_ERR;
}