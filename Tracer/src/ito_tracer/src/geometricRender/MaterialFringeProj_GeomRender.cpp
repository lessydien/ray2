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

/**\file MaterialFringeProj_GeomRender.cpp
* \brief refracting material
* 
*           
* \author Mauch
*/

#include "MaterialFringeProj_GeomRender.h"
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
void MaterialFringeProj_GeomRender::hit(rayStruct &ray, Mat_hitParams hitParams, double t_hit, int geometryID)
{
	
//	extern Group oGroup;
    geomRenderRayStruct *l_pRay=reinterpret_cast<geomRenderRayStruct*>(&ray);
	if ( !hitRenderFringeProj(*l_pRay, hitParams, this->params, t_hit, geometryID) )
        cout << "error in MaterialFringeProj_GeomRender.hit(): a ray is immersed in " << ray.nImmersed << ". It has hit the geometry " << geometryID << " which is not immersed in that medium!! ray will be stopped." << endl;		

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
MaterialError MaterialFringeProj_GeomRender::createCPUSimInstance(double lambda)
{
	// create simulation instance of coating
	if (MAT_NO_ERR != Material::createCPUSimInstance(lambda) )
	{
		std::cout << "error in MaterialFringeProj_GeomRender.createCPUSimInstance(): Material.createCPUSimInstance() returned an error." << "...\n";
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
MaterialError MaterialFringeProj_GeomRender::createOptiXInstance(RTcontext context, RTgeometryinstance &instance, int index, SimParams simParams, double lambda)
{
	if (MAT_NO_ERR != Material::createOptiXInstance(context, instance, index, simParams, lambda) )
	{
		std::cout << "error in MaterialFringeProj_GeomRender.createOptiXInstance(): Material.creatOptiXInstance() returned an error" << "...\n";
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
MaterialError MaterialFringeProj_GeomRender::updateOptiXInstance(RTcontext context, RTgeometryinstance &instance, int index, SimParams simParams, double lambda)
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
		std::cout << "error in MaterialFringeProj_GeomRender.updateOptiXInstance(): Material.updateOptiXInstance() returned an error" << "...\n";
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
void MaterialFringeProj_GeomRender::setParams(MatFringeProj_GeomRender_params params)
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
MatFringeProj_GeomRender_params MaterialFringeProj_GeomRender::getParams(void)
{
	return this->params;
}

FringeType MaterialFringeProj_GeomRender::asciiToFringeType(const char* ascii) const
{
	if (!strcmp(ascii, "GRAYCODE"))
		return FT_GRAYCODE;
	if (!strcmp(ascii, "SINUS"))
		return FT_SINUS;

    return FT_SINUS;
}

Orientation MaterialFringeProj_GeomRender::asciiToOrientation(const char* ascii) const
{
	if (!strcmp(ascii, "X"))
		return O_X;
	if (!strcmp(ascii, "Y"))
		return O_Y;
	if (!strcmp(ascii, "Z"))
		return O_Z;
    return O_Z;
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
MaterialError MaterialFringeProj_GeomRender::parseXml(pugi::xml_node &material, SimParams simParams)
{
	if (!Material::parseXml(material, simParams))
	{
		std::cout << "error in MaterialFringeProj_GeomRender.parseXml(): Material.parseXml() returned an error." << "...\n";
		return MAT_ERR;
	}

    Parser_XML l_parser;
	if (!this->checkParserError(l_parser.attrByNameToDouble(material, "power", this->params.power)))
		return MAT_ERR;
  //  if (!this->checkParserError(l_parser.attrByNameToDouble(material, "pupilRoot.x", this->params.pupilRoot.x)))
		//return MAT_ERR;
  //  if (!this->checkParserError(l_parser.attrByNameToDouble(material, "pupilRoot.y", this->params.pupilRoot.y)))
		//return MAT_ERR;
  //  if (!this->checkParserError(l_parser.attrByNameToDouble(material, "pupilRoot.z", this->params.pupilRoot.z)))
		//return MAT_ERR;

  //  if (!this->checkParserError(l_parser.attrByNameToDouble(material, "pupilTilt.x", this->params.pupilTilt.x)))
		//return MAT_ERR;
  //  if (!this->checkParserError(l_parser.attrByNameToDouble(material, "pupilTilt.y", this->params.pupilTilt.y)))
		//return MAT_ERR;
  //  if (!this->checkParserError(l_parser.attrByNameToDouble(material, "pupilTilt.z", this->params.pupilTilt.z)))
		//return MAT_ERR;

  //  if (!this->checkParserError(l_parser.attrByNameToDouble(material, "pupilAptRad.x", this->params.pupilAptRad.x)))
		//return MAT_ERR;
  //  if (!this->checkParserError(l_parser.attrByNameToDouble(material, "pupilAptRad.y", this->params.pupilAptRad.y)))
		//return MAT_ERR;

    const char* ascii_type=l_parser.attrValByName(material, "fringeType");
    this->params.fringeType=this->asciiToFringeType(ascii_type);
    const char* ascii_orientation=l_parser.attrValByName(material, "fringeOrientation");
    this->params.fringeOrientation=this->asciiToOrientation(ascii_orientation);

    if (!this->checkParserError(l_parser.attrByNameToDouble(material, "fringePeriod", this->params.fringePeriod)))
		return MAT_ERR;
    if (!this->checkParserError(l_parser.attrByNameToInt(material, "nrBits", this->params.nrBits)))
		return MAT_ERR;
    if (!this->checkParserError(l_parser.attrByNameToInt(material, "codeNr", this->params.codeNr)))
		return MAT_ERR;
    if (!this->checkParserError(l_parser.attrByNameToDouble(material, "fringePhase", this->params.fringePhase)))
		return MAT_ERR;
    this->params.fringePhase=this->params.fringePhase/360*2*M_PI;

    if (!this->checkParserError(l_parser.attrByNameToDouble(material, "geomRoot.x", this->params.geomRoot.x)))
		return MAT_ERR;
    if (!this->checkParserError(l_parser.attrByNameToDouble(material, "geomRoot.y", this->params.geomRoot.y)))
		return MAT_ERR;
    if (!this->checkParserError(l_parser.attrByNameToDouble(material, "geomRoot.z", this->params.geomRoot.z)))
		return MAT_ERR;

    if (!this->checkParserError(l_parser.attrByNameToDouble(material, "geomTilt.x", this->params.geomTilt.x)))
		return MAT_ERR;
    if (!this->checkParserError(l_parser.attrByNameToDouble(material, "geomTilt.y", this->params.geomTilt.y)))
		return MAT_ERR;
    if (!this->checkParserError(l_parser.attrByNameToDouble(material, "geomTilt.z", this->params.geomTilt.z)))
		return MAT_ERR;
    this->params.geomTilt.z=this->params.geomTilt.z/360*2*M_PI;
    this->params.geomTilt.y=this->params.geomTilt.y/360*2*M_PI;
    this->params.geomTilt.x=this->params.geomTilt.x/360*2*M_PI;

    
    return MAT_NO_ERR;
}