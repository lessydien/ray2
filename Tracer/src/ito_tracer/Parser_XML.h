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

/**\file Parser_XML.h
* \brief 
* 
*           
* \author Mauch
*/

#ifndef PARSER_XML_H
#define PARSER_XML_H

#include "Group.h"
#include "Detector.h"
#include "RayField.h"

#include "pugixml.hpp"

#include <stdio.h>
#include <string.h>
#include <vector>

using namespace std;
using namespace pugi;

class Parser_XML
{
public:
	Parser_XML()
	{

	}
	~Parser_XML()
	{

	}

	int countChildsByTagName(xml_node &root, char* tagname) const;
	vector<xml_node>* childsByTagName(xml_node &root, char* tagname) const;
	xml_attribute* attrByName(xml_node &root, char* attrname) const;
	const char* attrValByName(xml_node &root, char* attrname) const;
	ApertureType asciiToApertureType(const char* ascii) const;
	geometry_type asciiToGeometryType(const char* ascii) const;
	fieldType asciiToFieldType(const char* ascii) const;
	ImpAreaType asciiToImpAreaType(const char* ascii) const;
	detType asciiToDetectorType(const char* ascii) const;
	rayPosDistrType asciiToRayPosDistrType(const char* ascii) const;
	rayDirDistrType asciiToRayDirDistrType(const char* ascii) const;
	detOutFormat asciiToDetOutFormat(const char* ascii) const;
	CoatingType asciiToCoatType(const char* ascii) const;
	ScatterType asciiToScatType(const char* ascii) const;
	MaterialType asciiToMaterialType(const char* ascii) const;

	bool attrByNameToDouble(xml_node &root, char* name, double &param) const;
	bool attrByNameToApertureType(xml_node &root, char* name, ApertureType &param) const;
	bool attrByNameToInt(xml_node &root, char* name, int &param) const;
	bool attrByNameToShort(xml_node &root, char* name, short &param) const;
	bool attrByNameToLong(xml_node &root, char* name, unsigned long &param) const;
	bool attrByNameToSLong(xml_node &root, char* name, long &param) const;
	bool attrByNameToRayPosDistrType(xml_node &root, char* name, rayPosDistrType &param) const;
	bool attrByNameToRayDirDistrType(xml_node &root, char* name, rayDirDistrType &param) const;
	bool attrByNameToDetOutFormat(xml_node &root, char* name, detOutFormat &param) const;
	bool attrByNameToBool(xml_node &root, char* name, bool &param) const;
};

#endif