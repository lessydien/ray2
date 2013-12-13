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

/**\file FieldLib.cpp
* \brief 
* 
*           
* \author Mauch
*/

#include "FieldLib.h"
#include "MaterialLib.h"
#include "FieldParams.h"
#include "string.h"

using namespace std;

bool FieldFab::createFieldInstFromXML(xml_node &fieldNode, vector<Field*> &fieldVec) const
{	
	Parser_XML l_parser;
	// get object type from XML element
	//const char* oType=(l_parser.attrByName(geomNode, "objectType"))->value();
	// if we don't have an geometry, return NULL
	if (strcmp((l_parser.attrByName(fieldNode, "objectType"))->value(),"FIELD"))
	{
		std::cout << "error in FieldFab.createFieldInstanceFromXML(): objectType is not defined for given node." << std::endl;
		return false; 
	}

	// get geometry type from XML element
	const char* l_fieldTypeAscii = (l_parser.attrByName(fieldNode,"fieldType"))->value();
	if (l_fieldTypeAscii==NULL)
	{
		std::cout << "error in FieldFab.createInstanceFromXML(): fieldType is not defined for given node." << std::endl;
		return false;
	}

	fieldType l_fieldType = l_parser.asciiToFieldType(l_fieldTypeAscii);

	Field* l_pField;
	// create instance of geometry according to geomType
	switch (l_fieldType)
	{
	case GEOMRAYFIELD:
		l_pField=new GeometricRayField();
		break;
	case GEOMRAYFIELD_PSEUDOBANDWIDTH:
		l_pField=new GeometricRayField_PseudoBandwidth();
		break;
	case PATHINTTISSUERAYFIELD:
		l_pField=new PathIntTissueRayField();
		break;
	case SCALARSPHERICALWAVE:
		l_pField=new ScalarSphericalField();
		break;
	case SCALARPLANEWAVE:
		l_pField=new ScalarPlaneField();
		break;
	case SCALARGAUSSIANWAVE:
		l_pField=new ScalarGaussianField();
		break;
	case SCALARUSERWAVE:
		l_pField=new ScalarUserField();
		break;

	default:
		cout << "error om GeometryFab.createInstanceFromXML(): unknown geometryType." << endl;
		return false;
		break;
	}

	// call parsing routine of geometry
	if (FIELD_NO_ERR != l_pField->parseXml(fieldNode, fieldVec))
	{
		cout << "error om FieldFab.createInstanceFromXML(): field.parseXml() returned an error" << endl;
		return false;
	}

	fieldVec.push_back(l_pField);

	return true;
}