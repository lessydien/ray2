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

/**\file CoatingLib.cpp
* \brief 
* 
*           
* \author Mauch
*/

#include "CoatingLib.h"

bool CoatingFab::createCoatInstFromXML(xml_node &node, Coating* &pCoat, SimParams simParams) const
{	
	Parser_XML l_parser;
	// get object type from XML element
	//const char* oType=(l_parser.attrByName(geometry, "ObjectType"))->value();
	// if we don't have an geometry, return NULL
	if (strcmp((l_parser.attrByName(node, "objectType"))->value(),"COATING"))
		return false;

	// get geometry type from XML element
	const char* coatTypeAscii = (l_parser.attrByName(node,"coatingType"))->value();
	if (coatTypeAscii == "")
	{
		std::cout << "error in CoatingLib.createCoatInstFromXML(): coatingType is not defined" << "...\n";
		return false;
	}

	CoatingType coatType=l_parser.asciiToCoatType(coatTypeAscii);
	// create instance of geometry according to geomType
	switch (coatType)
	{
	case CT_NUMCOEFFS:
		pCoat=new Coating_NumCoeffs();
		break;
	case CT_NOCOATING:
		pCoat=new Coating_NoCoating();
		break;
	case CT_FRESNELCOEFFS:
		pCoat=new Coating_FresnelCoeffs();
		break;
	case CT_DISPNUMCOEFFS:
		pCoat=new Coating_DispersiveNumCoeffs();
		break;
	default:
		pCoat=new Coating_NoCoating(); // default is no caoting
		break;
	}

	pCoat->parseXml(node, simParams);

	return true;
}