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

/**\file ScatterLib_GeomRender.cpp
* \brief 
* 
*           
* \author Mauch
*/

#include "ScatterLib_GeomRender.h"

bool ScatterFab_GeomRender::createScatInstFromXML(xml_node &node, Scatter* &pScat, SimParams simParams) const
{	
	Parser_XML l_parser;
	// get object type from XML element
	//const char* oType=(l_parser.attrByName(geometry, "ObjectType"))->value();
	// if we don't have an geometry, return NULL
	if (strcmp((l_parser.attrByName(node, "objectType"))->value(),"SCATTER"))
		return false;

	// get geometry type from XML element
	const char* scatTypeAscii = (l_parser.attrByName(node,"scatterType"))->value();
	if (scatTypeAscii == "")
	{
		std::cout << "error in ScatterLib.createScatInstFromXML(): scatterType is not defined" << "...\n";
		return false;
	}

	ScatterType scatType=l_parser.asciiToScatType(scatTypeAscii);
	// create instance of geometry according to geomType
	switch (scatType)
	{
	case ST_NOSCATTER:
		pScat=new Scatter_NoScatter();
		break;
	//case ST_TORRSPARR1D:
	//	pScat=new Scatter_TorranceSparrow1D_DiffRays();
	//	break;
	//case ST_DOUBLECAUCHY1D:
	//	pScat=new Scatter_DoubleCauchy1D_DiffRays;
	//	break;
	case ST_LAMBERT2D:
		pScat=new Scatter_Lambert2D_GeomRender();
		break;
	case ST_PHONG:
		pScat=new Scatter_Phong_GeomRender();
		break;
	case ST_COOKTORRANCE:
		pScat=new Scatter_CookTorrance_GeomRender();
		break;
	default:
        std::cout << "error in ScatterLib_GeomRender.createScatInstFromXML(): scatterType " << scatTypeAscii << " is not known...\n";
		pScat=new Scatter_NoScatter();
		break;
	}

	pScat->parseXml(node, simParams);

	return true;
}