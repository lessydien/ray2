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

/**\file GeometryLib.cpp
* \brief 
* 
*           
* \author Mauch
*/

#include "GeometryLib.h"
#include "MaterialLib.h"
#include "string.h"

using namespace std;

bool GeometryFab::createGeomInstFromXML(xml_node &geomNode, SimParams simParams, vector<Geometry*> &geomVec) const
{	
	Geometry* l_pGeometry=NULL;
	Parser_XML l_parser;
	// get object type from XML element
	//const char* oType=(l_parser.attrByName(geomNode, "objectType"))->value();
	// if we don't have an geometry, return NULL
	if (strcmp((l_parser.attrByName(geomNode, "objectType"))->value(),"GEOMETRY"))
	{
		std::cout << "error in GeometryFab.createGeomInstanceFromXML(): objectType is not defined for given node." << "...\n";
		return false; // return empty vector
	}

	// get geometry type from XML element
	const char* l_geomTypeAscii = (l_parser.attrByName(geomNode,"geomType"))->value();
	if (l_geomTypeAscii==NULL)
	{
		std::cout << "error in GeometryFab.createInstanceFromXML(): geomType is not defined for given node." << "...\n";
		return false;
	}

	geometry_type l_geomType = l_parser.asciiToGeometryType(l_geomTypeAscii);

	// create instance of geometry according to geomType
	switch (l_geomType)
	{
	case GEOM_SPHERICALLENSE:
		l_pGeometry=new SphericalLense(1);
		break;
	case GEOM_SPHERICALSURF:
		l_pGeometry=new SphericalSurface(1);
		break;
	case GEOM_PARABOLICSURF:
		l_pGeometry=new ParabolicSurface(1);
		break;
	case GEOM_MICROLENSARRAY:
		l_pGeometry=new MicroLensArray(1);
		break;
	case GEOM_MICROLENSARRAYSURF:
		l_pGeometry=new MicroLensArraySurface(1);
		break;
	case GEOM_STOPARRAYSURF:
		l_pGeometry=new StopArraySurface(1);
		break;
	case GEOM_APERTUREARRAYSURF:
		l_pGeometry=new ApertureArraySurface(1);
		break;
	case GEOM_PLANESURF:
		l_pGeometry=new PlaneSurface(1);
		break;
	case GEOM_ASPHERICALSURF:
		l_pGeometry=new AsphericalSurface(1);
		break;
	case GEOM_CYLLENSESURF:
		l_pGeometry=new CylLenseSurface(1);
		break;
	case GEOM_CYLPIPE:
		l_pGeometry=new CylPipe(1);
		break;
	case GEOM_CONEPIPE:
		l_pGeometry=new ConePipe(1);
		break;
	case GEOM_IDEALLENSE:
		l_pGeometry=new IdealLense(1);
		break;
	case GEOM_APERTURESTOP:
		l_pGeometry=new ApertureStop(1);
		break;
	case GEOM_COSINENORMAL:
		l_pGeometry=new SinusNormalSurface(1);
		break;
	case GEOM_CADOBJECT:
		l_pGeometry=new CadObject(1);
		break;
	case GEOM_SUBSTRATE:
		l_pGeometry=new Substrate(1);
		break;
	case GEOM_VOLUMESCATTERERBOX:
		l_pGeometry=new VolumeScattererBox(1);
		break;

	default:
		cout << "error om GeometryFab.createInstanceFromXML(): unknown geometryType for geometric ray tracing." << endl;
		return false;
		break;
	}

	
	// call parsing routine of geometry
	if (GEOM_NO_ERR != l_pGeometry->parseXml(geomNode, simParams, geomVec))
	{
		cout << "error om GeomFab.createInstanceFromXML(): geometry.parseXml() returned an error" << endl;
		return false;
	}

	//geomVec.push_back(l_pGeometry);

	return true;
}