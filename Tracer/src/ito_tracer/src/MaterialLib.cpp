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

/**\file MaterialLib.cpp
* \brief 
* 
*           
* \author Mauch
*/

#include "MaterialLib.h"
#include "string.h"
#include "Scatter.h"
#include "CoatingLib.h"
#include "ScatterLib.h"

bool MaterialFab::createMatInstFromXML(xml_node &node, Material* &pMat, SimParams simParams) const
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
		pMat=new MaterialReflecting();
		break;
	case MT_ABSORB:
		pMat=new MaterialAbsorbing();
		break;
	case MT_COVGLASS:
		pMat=new MaterialReflecting_CovGlass();
		break;
	case MT_LINGRAT1D:
		pMat=new MaterialLinearGrating1D();
		break;
	case MT_REFRMATERIAL:
		pMat=new MaterialRefracting();
		break;
	case MT_IDEALLENSE:
		pMat=new MaterialIdealLense();
		break;
	case MT_DIFFRACT:
		pMat=new MaterialDiffracting();
	case MT_FILTER:
		pMat=new MaterialFilter();
		break;
	case MT_PATHTRACESRC:
		pMat=new MaterialPathTraceSource();
		break;
	case MT_DOE:
		pMat=new MaterialDOE();
		break;
	case MT_VOLUMESCATTER:
		pMat=new MaterialVolumeScatter();
		break;
	case MT_VOLUMESCATTERBOX:
		pMat=new MaterialVolumeScatterBox();
		break;
	case MT_VOLUMEABSORB:
		pMat=new MaterialVolumeAbsorbing();
		break;
	default:
		pMat=new Material();
		break;
	}

	if (MAT_NO_ERR != pMat->parseXml(node, simParams) )
	{
		std::cout <<"error in MaterialFab.createMatInstFromXML(): mat.parseXML() returned an error" << "...\n";
		return false;
	}

	return true;
}