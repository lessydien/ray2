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

/**\file ScalarSphericalField.cpp
* \brief scalar representation of light field
* 
*           
* \author Mauch
*/

#include "ScalarSphericalField.h"
#include "myUtil.h"
#include "sampleConfig.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "math.h"
#ifdef _MATSUPPORT
	#include "mat.h"
#endif
#include "Parser_XML.h"

/**
 * \detail parseXml
 *
 * \param[in] pugi::xml_node &field, vector<Field*> &fieldVec
 * 
 * \return void
 * \sa 
 * \remarks 
 * \author Mauch
 */
fieldError  ScalarSphericalField::parseXml(pugi::xml_node &field, vector<Field*> &fieldVec, SimParams simParams)
{
	// call base class function
	if (FIELD_NO_ERR != ScalarLightField::parseXml(field, fieldVec, simParams))
	{
		std::cout << "error in ScalarSphericalField.parseXml(): ScalarLightField.parseXml()  returned an error." << "...\n";
		return FIELD_ERR;
	}
	Parser_XML l_parser;
	if (!l_parser.attrByNameToDouble(field, "amplMax", this->paramsPtr->amplMax))
	{
		std::cout << "error in ScalarLightField.parseXml(): amplMax is not defined" << "...\n";
		return FIELD_ERR;
	}
	if (!l_parser.attrByNameToDouble(field, "radius.x", this->paramsPtr->radius.x))
	{
		std::cout << "error in ScalarGaussianField.parseXml(): radius.x is not defined" << "...\n";
		return FIELD_ERR;
	}
	if (!l_parser.attrByNameToDouble(field, "radius.y", this->paramsPtr->radius.y))
	{
		std::cout << "error in ScalarGaussianField.parseXml(): radius.y is not defined" << "...\n";
		return FIELD_ERR;
	}
	if (!l_parser.attrByNameToDouble(field, "numApt.x", this->paramsPtr->numApt.x))
	{
		std::cout << "error in ScalarGaussianField.parseXml(): numApt.x is not defined" << "...\n";
		return FIELD_ERR;
	}
	if (!l_parser.attrByNameToDouble(field, "numApt.y", this->paramsPtr->numApt.y))
	{
		std::cout << "error in ScalarGaussianField.parseXml(): numApt.y is not defined" << "...\n";
		return FIELD_ERR;
	}
	return FIELD_NO_ERR;
};