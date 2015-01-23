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

/**\file DetectorLib.cpp
* \brief 
* 
*           
* \author Mauch
*/

#include "DetectorLib.h"
#include "MaterialLib.h"
#include "string.h"

using namespace std;

bool DetectorFab::createDetInstFromXML(xml_node &detNode, vector<Detector*> &detVec) const
{	
	Detector* l_pDetector=NULL;
	Parser_XML l_parser;
	// get object type from XML element
	//const char* oType=(l_parser.attrByName(geomNode, "objectType"))->value();
	// if we don't have an geometry, return NULL
	if (strcmp((l_parser.attrByName(detNode, "objectType"))->value(),"DETECTOR"))
	{
		std::cout << "error in DetectorFab.createDetInstanceFromXML(): objectType is not defined for given node." << "...\n";
		return false; // return empty vector
	}

	// get geometry type from XML element
	const char* l_detTypeAscii = (l_parser.attrByName(detNode,"detType"))->value();
	if (l_detTypeAscii==NULL)
	{
		std::cout << "error in DetectorFab.createInstanceFromXML(): geomType is not defined for given node." << "...\n";
		return false;
	}
	detType l_detType=l_parser.asciiToDetectorType(l_detTypeAscii);

	switch (l_detType)
	{
	case DET_RAYDATA:
	case DET_RAYDATA_RED:
	case DET_RAYDATA_GLOBAL:
	case DET_RAYDATA_RED_GLOBAL:
		l_pDetector=new DetectorRaydata();
		break;
	case DET_INTENSITY:
		l_pDetector=new DetectorIntensity();
		break;
	case DET_VOLUMEINTENSITY:
		l_pDetector=new DetectorVolumeIntensity();
		break;
	case DET_PHASESPACE:
		l_pDetector=new DetectorPhaseSpace();
		break;
	case DET_FIELD:
		l_pDetector=new DetectorField();
		break;
	default:
		cout << "error in DetectorFab.createDetInstanceFromXML(): unknown detector type" << endl;
		return false;
	}

	// call parsing routine of detector
	if (DET_NO_ERROR != l_pDetector->parseXml(detNode, detVec))
	{
		cout << "error om DetectorFab.createInstanceFromXML(): detector.parseXml() returned an error" << endl;
		return false;
	}

	detVec.push_back(l_pDetector);

	//// create material
	//MaterialFab l_matFab;
	//xml_node l_matNode = geomNode.child("material");
	//Material* l_pMaterial=l_matFab.createMatInstFromXML(l_matNode);

	//// add material to geometry
	//l_pGeometry->setMaterial(l_pMaterial,0);
	return true;
}