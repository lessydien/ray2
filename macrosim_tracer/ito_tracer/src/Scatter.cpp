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

/**\file Scatter.cpp
* \brief base class to all scattering properties that can be applied to surfaces
* 
*           
* \author Mauch
*/

#include "Scatter.h"
#include "myUtil.h"
#include "sampleConfig.h"
#include <iostream>

#include "Parser_XML.h"
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
ScatterError Scatter::processParseResults(MaterialParseParamStruct &parseResults_Mat)
{
	std::cout << "error in Scatter.processParseResults(): not defined for the given Scatter" << "...\n";
	return SCAT_ERROR;
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
ScatterError Scatter::parseXml(pugi::xml_node &geometry, SimParams simParams)
{
	Parser_XML l_parser;
    if (!this->checkParserError(l_parser.attrByNameToDouble(geometry, "pupRoot.x", this->getFullParams()->impAreaRoot.x)))
		return SCAT_ERROR;
    if (!this->checkParserError(l_parser.attrByNameToDouble(geometry, "pupRoot.y", this->getFullParams()->impAreaRoot.y)))
		return SCAT_ERROR;
    if (!this->checkParserError(l_parser.attrByNameToDouble(geometry, "pupRoot.z", this->getFullParams()->impAreaRoot.z)))
		return SCAT_ERROR;

    if (!this->checkParserError(l_parser.attrByNameToDouble(geometry, "pupTilt.x", this->getFullParams()->impAreaTilt.x)))
		return SCAT_ERROR;
    if (!this->checkParserError(l_parser.attrByNameToDouble(geometry, "pupTilt.y", this->getFullParams()->impAreaTilt.y)))
		return SCAT_ERROR;
    if (!this->checkParserError(l_parser.attrByNameToDouble(geometry, "pupTilt.z", this->getFullParams()->impAreaTilt.z)))
		return SCAT_ERROR;

    if (!this->checkParserError(l_parser.attrByNameToDouble(geometry, "pupAptRad.x", this->getFullParams()->impAreaHalfWidth.x)))
		return SCAT_ERROR;
    if (!this->checkParserError(l_parser.attrByNameToDouble(geometry, "pupAptRad.y", this->getFullParams()->impAreaHalfWidth.y)))
		return SCAT_ERROR;

    if (!this->checkParserError(l_parser.attrByNameToApertureType(geometry, "pupAptType", this->getFullParams()->impAreaType)))
		return SCAT_ERROR;

}

void Scatter::setPathToPtx(char* path)
{
	memcpy(this->path_to_ptx, path, sizeof(this->path_to_ptx));	
};

char* Scatter::getPathToPtx(void)
{
	return this->path_to_ptx;
};

ScatterError Scatter::createCPUSimInstance(double lambda)
{
	return SCAT_NO_ERROR;	
};

ScatterError Scatter::createOptiXInstance(double lambda, char** path_to_ptx_in)
{
	this->createCPUSimInstance(lambda);
	strcat(*path_to_ptx_in, this->getPathToPtx());
	this->update=false;
	return SCAT_NO_ERROR;	
};

ScatterError Scatter::setParams2Program( RTcontext context, RTprogram *closest_hit_programPtr, RTvariable *l_scatterParamsPtr)
{
	this->update=false;
	return SCAT_NO_ERROR;
};

ScatterError Scatter::setFullParams(Scatter_Params* ptrIn)
{
//	this->fullParamsPtr=ptrIn;
    std::cout << "error in Scatter.setFullParams(): not defined for the given Scatter" << "...\n";
	return SCAT_ERROR;
};

Scatter_Params* Scatter::getFullParams(void)
{
//	return this->fullParamsPtr;
    std::cout << "error in Scatter.getFullParams(): not defined for the given Scatter" << "...\n";
	return NULL;
};

ScatterError Scatter::setReducedParams(Scatter_ReducedParams* ptrIn)
{
//	this->reducedParams=*ptrIn;
    std::cout << "error in Scatter.setReducedParams(): not defined for the given Scatter" << "...\n";
	return SCAT_ERROR;
};

Scatter_ReducedParams* Scatter::getReducedParams(void)
{
//	return &(this->reducedParams);
    std::cout << "error in Scatter.getReducedParams(): not defined for the given Scatter" << "...\n";
	return NULL;
};

void Scatter::hit(rayStruct &ray, Mat_hitParams hitParams)
{
	std::cout << "error in Scatter.hit(): hit is not yet implemented for geometric rays for the given scatter. no scatter assumed." << "...\n";
	// dummy function to be overwritten by child class
}

void Scatter::hit(diffRayStruct &ray, Mat_hitParams hitParams)
{
	std::cout << "error in Scatter.hit(): hit is not yet implemented for differential rays for the given scatter. no scatter assumed." << "...\n";
	// dummy function to be overwritten by child class
}

void Scatter::hit(gaussBeamRayStruct &ray, gaussBeam_geometricNormal normal)
{
	std::cout << "error in Scatter.hit(): hit is not yet implemented for gaussian beamlets for the given scatter. no scatter assumed." << "...\n";
	// dummy function to be overwritten by child class
}

/**
 * \detail checks wether parseing was succesfull and assembles the error message if it was not
 *
 * returns the coordinates of the minimum corner of the bounding box of the surface
 *
 * \param[in] char *msg
 * 
 * \return bool
 * \sa 
 * \remarks 
 * \author Mauch
 */
bool Scatter::checkParserError(char *msg)
{
	if (msg==NULL)
		return true;
	else
	{
		cout << "error in Scatter.parseXML(): " << msg << endl;
		delete msg;
		msg=NULL;
		return false;
	}
};