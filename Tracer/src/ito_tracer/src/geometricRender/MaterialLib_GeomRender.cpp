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

/**\file MaterialLib_GeomRender.cpp
* \brief 
* 
*           
* \author Mauch
*/

#include "MaterialLib_GeomRender.h"
#include "string.h"
#include "Scatter_GeomRender.h"
#include "CoatingLib_GeomRender.h"
#include "ScatterLib_GeomRender.h"

bool MaterialFab_GeomRender::createMatInstFromXML(xml_node &node, Material* &pMat, SimParams simParams) const
{	
	Parser_XML l_parser;
	// get object type from XML element
	//const char* oType=(l_parser.attrByName(geometry, "ObjectType"))->value();
	// if we don't have an geometry, return NULL
	const char* objType = (l_parser.attrByName(node,"objectType"))->value();
	if (strcmp((l_parser.attrByName(node, "objectType"))->value(),"MATERIAL"))
		return false;

	// get material type from XML element
	const char* matTypeAscii = (l_parser.attrByName(node,"materialType"))->value();

	MaterialType matType=l_parser.asciiToMaterialType(matTypeAscii);
	// create instance of material according to matType
	switch (matType)
	{
	case MT_MIRROR:
		pMat=new MaterialReflecting_GeomRender();
		break;
	case MT_ABSORB:
		pMat=new MaterialAbsorbing_GeomRender();
		break;
	case MT_REFRMATERIAL:
		pMat=new MaterialRefracting_GeomRender();
		break;
	case MT_IDEALLENSE:
		pMat=new MaterialIdealLense_GeomRender();
		break;
    case MT_RENDERLIGHT:
        pMat=new MaterialLight_GeomRender();
        break;
    case MT_RENDERFRINGEPROJ:
        pMat=new MaterialFringeProj_GeomRender();
        break;

	default:
		std::cout <<"error in MaterialFab_GeomRender.createMatInstFromXML(): unknown Material type" << "...\n";
        return false;
		break;
	}

	if (MAT_NO_ERR != pMat->parseXml(node, simParams) )
	{
		std::cout <<"error in MaterialFab_GeomRender.createMatInstFromXML(): mat.parseXML() returned an error" << "...\n";
		return false;
	}

	return true;
}