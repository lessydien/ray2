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

/**\file Parser_XML.cpp
* \brief collection of functions that create the internal scene graph from the result of parsing the various input files
* 
*           
* \author Mauch
*/

#include <iostream>
#include "stdio.h"
#include <stdlib.h>
#include <stdio.h>
#include "RayField.h"
#include "Group.h"
#include "Detector.h"
#include "pugixml.hpp"
#include "string.h"
#include "Parser_XML.h"
#include "Geometry_Intersect.h"

using namespace std;
using namespace pugi;


int Parser_XML::countChildsByTagName(xml_node &root, char* tagname) const
{
	int count=0;
	for (xml_node child = root.first_child(); child; child=child.next_sibling())
	{
		if ( strcmp(child.name(),tagname) == 0)
		{
			count++;
		}
	}	
	return count;
};

vector<xml_node>* Parser_XML::childsByTagName(xml_node &root, char* tagname) const
{
	// create list
	vector<xml_node>* l_pElementList = new vector<xml_node>;
	// fill list with elements
	for (xml_node child = root.first_child(); child; child=child.next_sibling())
	{
		if ( strcmp(child.name(),tagname) == 0)
		{
			l_pElementList->push_back(child);
		}
	}	
	return l_pElementList;
};

xml_attribute* Parser_XML::attrByName(xml_node &root, char* attrname) const
{
	xml_attribute* l_pAttr=new xml_attribute;
	for (xml_attribute attr=root.first_attribute(); attr; attr=attr.next_attribute())
	{
		if (strcmp(attr.name(),attrname) == 0)
		{
			*l_pAttr=attr;
			break;
		}
	}
	return l_pAttr;
};

const char* Parser_XML::attrValByName(xml_node &root, char* attrname) const
{
	xml_attribute l_attr;
	for (xml_attribute attr=root.first_attribute(); attr; attr=attr.next_attribute())
	{
		if (strcmp(attr.name(),attrname) == 0)
		{
			return attr.value();
		}
	}
	return NULL;
};

detOutFormat Parser_XML::asciiToDetOutFormat(const char* ascii) const
{
	if (!strcmp(ascii, "MAT"))
		return DET_OUT_MAT;
	if (!strcmp(ascii, "TEXT"))
		return DET_OUT_TEXT;
	if (!strcmp(ascii, "X3P"))
		return DET_OUT_X3P;

	cout << "error in Parser_XML.asciiToDetOutFormat: unknown detector type: "  << ascii << endl;

	return DET_OUT_TEXT;
};

rayPosDistrType Parser_XML::asciiToRayPosDistrType(const char* ascii) const
{
	if (!strcmp(ascii, "RAYPOS_RAND_RECT"))
		return RAYPOS_RAND_RECT;
	if (!strcmp(ascii, "RAYPOS_RAND_RECT_NORM"))
		return RAYPOS_RAND_RECT_NORM;
	if (!strcmp(ascii, "RAYPOS_GRID_RECT"))
		return RAYPOS_GRID_RECT;
	if (!strcmp(ascii, "RAYPOS_RAND_RAD"))
		return RAYPOS_RAND_RAD;
	if (!strcmp(ascii, "RAYPOS_RAND_RAD_NORM"))
		return RAYPOS_RAND_RAD_NORM;
	if (!strcmp(ascii, "RAYPOS_GRID_RAD"))
		return RAYPOS_GRID_RAD;

	cout << "error in Parser_XML.asciiToRayPosDistrType: unknown RayPosDistr type: "  << ascii << endl;

	return RAYPOS_UNKNOWN;
};

rayDirDistrType Parser_XML::asciiToRayDirDistrType(const char* ascii) const
{
	if (!strcmp(ascii, "RAYDIR_RAND_RECT"))
		return RAYDIR_RAND_RECT;
	if (!strcmp(ascii, "RAYDIR_RAND_RAD"))
		return RAYDIR_RAND_RAD;
	if (!strcmp(ascii, "RAYDIR_RANDNORM_RECT"))
		return RAYDIR_RANDNORM_RECT;
	if (!strcmp(ascii, "RAYDIR_RANDIMPAREA"))
		return RAYDIR_RANDIMPAREA;
	if (!strcmp(ascii, "RAYDIR_UNIFORM"))
		return RAYDIR_UNIFORM;
	if (!strcmp(ascii, "RAYDIR_GRID_RECT"))
		return RAYDIR_GRID_RECT;
	if (!strcmp(ascii, "RAYDIR_GRID_RECT_FARFIELD"))
		return RAYDIR_GRID_RECT_FARFIELD;
	if (!strcmp(ascii, "RAYDIR_GRID_RAD"))
		return RAYDIR_GRID_RAD;

	cout << "error in Parser_XML.asciiToRayDirDistrType: unknown RayDirDistr type: "  << ascii << endl;

	return RAYDIR_UNKNOWN;
}

ApertureType Parser_XML::asciiToApertureType(const char* ascii) const
{
	if (!strcmp(ascii, "RECTANGULAR"))
		return AT_RECT;
	if (!strcmp(ascii, "ELLIPTICAL"))
		return AT_ELLIPT;
	if (!strcmp(ascii, "RECTOBSC"))
		return AT_RECTOBSC;
	if (!strcmp(ascii, "ELLIPTOBSC"))
		return AT_ELLIPTOBSC;
	if (!strcmp(ascii, "INFTY"))
		return AT_INFTY;

	cout << "error in Parser_XML.asciiToApertureType: unknown aperture type: "  << ascii << endl;

	return AT_UNKNOWNATYPE;
}

ImpAreaType Parser_XML::asciiToImpAreaType(const char* ascii) const
{
	if (!strcmp(ascii, "FARFIELD"))
		return IMP_FARFIELD;
	if (!strcmp(ascii, "CONV"))
		return IMP_CONV;

	cout << "error in Parser_XML.asciiToImpAreaType: unknown ImpArea type: "  << ascii << endl;

	return IMP_UNKNOWN;
};

fieldType Parser_XML::asciiToFieldType(const char* ascii) const
{
	if (!strcmp(ascii, "GEOMRAYFIELD"))
		return GEOMRAYFIELD;

	if (!strcmp(ascii, "DIFFRAYFIELD"))
		return DIFFRAYFIELD;

	if (!strcmp(ascii, "DIFFRAYFIELDRAYAIM"))
		return DIFFRAYFIELDRAYAIM;

	if (!strcmp(ascii, "PATHTRACERAYFIELD"))
		return PATHTRACERAYFIELD;

	if (!strcmp(ascii, "SCALARFIELD"))
		return SCALARFIELD;

	if (!strcmp(ascii, "SCALARSPHERICALWAVE"))
		return SCALARSPHERICALWAVE;

	if (!strcmp(ascii, "SCALARPLANEWAVE"))
		return SCALARPLANEWAVE;

	if (!strcmp(ascii, "SCALARGAUSSIANWAVE"))
		return SCALARGAUSSIANWAVE;

	if (!strcmp(ascii, "SCALARUSERWAVE"))
		return SCALARUSERWAVE;

	if (!strcmp(ascii, "VECFIELD"))
		return VECFIELD;

	if (!strcmp(ascii, "INTFIELD"))
		return INTFIELD;

	if (!strcmp(ascii, "PATHINTTISSUERAYFIELD"))
		return PATHINTTISSUERAYFIELD;

	cout << "error in Parser_XML.asciiToFieldType: unknown field type: "  << ascii << endl;

	return FIELDUNKNOWN;
};

geometry_type Parser_XML::asciiToGeometryType(const char* ascii) const
{
	if (!strcmp(ascii, "PLANESURFACE"))
		return GEOM_PLANESURF;

	if (!strcmp(ascii, "SPHERICALSURFACE"))
		return GEOM_SPHERICALSURF;

	if (!strcmp(ascii, "PARABOLICSURFACE"))
		return GEOM_PARABOLICSURF;

	if (!strcmp(ascii, "SPHERICALLENSE"))
		return GEOM_SPHERICALLENSE;

	if (!strcmp(ascii, "MICROLENSARRAY"))
		return GEOM_MICROLENSARRAY;

	if (!strcmp(ascii, "MICROLENSARRAYSURF"))
		return GEOM_MICROLENSARRAYSURF;

	if (!strcmp(ascii, "APERTUREARRAY"))
		return GEOM_APERTUREARRAYSURF;

	if (!strcmp(ascii, "STOPARRAY"))
		return GEOM_STOPARRAYSURF;

	if (!strcmp(ascii, "ASPHERICALSURF"))
		return GEOM_ASPHERICALSURF;

	if (!strcmp(ascii, "CYLLENSESURF"))
		return GEOM_CYLLENSESURF;

	if (!strcmp(ascii, "GRATING"))
		return GEOM_GRATING;

	if (!strcmp(ascii, "CYLPIPE"))
		return GEOM_CYLPIPE;

	if (!strcmp(ascii, "CYLLENSE"))
		return GEOM_CYLLENSE;

	if (!strcmp(ascii, "CONEPIPE"))
		return GEOM_CONEPIPE;

	if (!strcmp(ascii, "IDEALLENSE"))
		return GEOM_IDEALLENSE;

	if (!strcmp(ascii, "APERTURESTOP"))
		return GEOM_APERTURESTOP;

	if (!strcmp(ascii, "COSEINNORMAL"))
		return GEOM_COSINENORMAL;

	if (!strcmp(ascii, "CADOBJECT"))
		return GEOM_CADOBJECT;

	if (!strcmp(ascii, "SUBSTRATE"))
		return GEOM_SUBSTRATE;

	if (!strcmp(ascii, "VOLUMESCATTERERBOX"))
		return GEOM_VOLUMESCATTERERBOX;


	cout << "error in Parser_XML.asciiToGeometryType: unknown geometry type: "  << ascii << endl;

	return GEOM_UNKNOWN;
};

CoatingType Parser_XML::asciiToCoatType(const char* ascii) const
{
	if (!strcmp(ascii,"NUMCOEFFS"))
		return CT_NUMCOEFFS;
	if (!strcmp(ascii,"NOCOATING"))
		return CT_NOCOATING;
	if (!strcmp(ascii,"FRESNELCOEFFS"))
		return CT_FRESNELCOEFFS;
	if (!strcmp(ascii,"DISPNUMCOEFFS"))
		return CT_DISPNUMCOEFFS;

	cout << "error in Parser_XML.asciiToCoatType: unknown coat type: "  << ascii << endl;
	return CT_NOCOATING;
}

ScatterType Parser_XML::asciiToScatType(const char* ascii) const
{
	if (!strcmp(ascii,"LAMBERT2D"))
		return ST_LAMBERT2D;
	if (!strcmp(ascii,"NOSCATTER"))
		return ST_NOSCATTER;
	if (!strcmp(ascii,"TORRSPARR1D"))
		return ST_TORRSPARR1D;
	if (!strcmp(ascii,"TORRSPARR2D"))
		return ST_TORRSPARR2D;
	if (!strcmp(ascii,"TORRSPARR2DPATHTRACE"))
		return ST_TORRSPARR2DPATHTRACE;
	if (!strcmp(ascii,"DISPDOUBLECAUCHY1D"))
		return ST_DISPDOUBLECAUCHY1D;
	if (!strcmp(ascii,"ST_DOUBLECAUCHY1D"))
		return ST_TORRSPARR2D;

	cout << "error in Parser_XML.asciiToScatType: unknown scatter type: "  << ascii << endl;
	return ST_NOSCATTER;
}

MaterialType Parser_XML::asciiToMaterialType(const char* ascii) const
{
	if (!strcmp(ascii,"REFRACTING"))
		return MT_REFRMATERIAL;
	if (!strcmp(ascii,"ABSORBING"))
		return MT_ABSORB;
	if (!strcmp(ascii,"DIFFRACTING"))
		return MT_DIFFRACT;
	if (!strcmp(ascii,"FILTER"))
		return MT_FILTER;
	if (!strcmp(ascii,"LINGRAT1D"))
		return MT_LINGRAT1D;
	if (!strcmp(ascii,"MATIDEALLENSE"))
		return MT_IDEALLENSE;
	if (!strcmp(ascii,"REFLECTING"))
		return MT_MIRROR;
	if (!strcmp(ascii,"REFLECTINGCOVGLASS"))
		return MT_COVGLASS;
	if (!strcmp(ascii,"PATHTRACESOURCE"))
		return MT_PATHTRACESRC;
	if (!strcmp(ascii,"DOE"))
		return MT_DOE;
	if (!strcmp(ascii,"VOLUMESCATTER"))
		return MT_VOLUMESCATTER;
	if (!strcmp(ascii,"VOLUMESCATTERBOX"))
		return MT_VOLUMESCATTERBOX;

	cout << "error in Parser_XML.asciiToMaterialType: unknown material type: "  << ascii << endl;
	return MT_UNKNOWNMATERIAL;
}

detType Parser_XML::asciiToDetectorType(const char* ascii) const
{
	if (!strcmp(ascii, "RAYDATA"))
		return DET_RAYDATA;

	if (!strcmp(ascii, "RAYDATA_RED"))
		return DET_RAYDATA_RED;

	if (!strcmp(ascii, "RAYDATA_GLOBAL"))
		return DET_RAYDATA_GLOBAL;

	if (!strcmp(ascii, "RAYDATA_RED_GLOBAL"))
		return DET_RAYDATA_RED_GLOBAL;

	if (!strcmp(ascii, "INTENSITY"))
		return DET_INTENSITY;

	if (!strcmp(ascii, "PHASESPACE"))
		return DET_PHASESPACE;

	if (!strcmp(ascii, "FIELD"))
		return DET_FIELD;

	return DET_UNKNOWN;
};

accelType Parser_XML::asciiToAccelType(const char* ascii) const
{
	if (!strcmp(ascii, "NOACCEL"))
		return ACCEL_NOACCEL;
	if (!strcmp(ascii, "BVH"))
		return ACCEL_BVH;
	if (!strcmp(ascii, "SBVH"))
		return ACCEL_SBVH;
	if (!strcmp(ascii, "MBVH"))
		return ACCEL_MBVH;
	if (!strcmp(ascii, "LBVH"))
		return ACCEL_LBVH;
	if (!strcmp(ascii, "TRIANGLEKDTREE"))
		return ACCEL_TRIANGLEKDTREE;

	cout << "warning in Parser_XML.asciiToAccelType: unknown acceleration type: "  << ascii << ". no acceleration will be set." << endl;

	return ACCEL_NOACCEL;
}

bool Parser_XML::attrByNameToDouble(xml_node &root, char* name, double &param) const
{
	const char* str=attrValByName(root, name);
	if (str==NULL)
		return false;
	param=atof(str);
	return true;
};

bool Parser_XML::attrByNameToApertureType(xml_node &root, char* name, ApertureType &param) const
{
	const char* str=attrValByName(root, name);
	if (str==NULL)
		return false;
	param=asciiToApertureType(str);	
	return true;
};

bool Parser_XML::attrByNameToInt(xml_node &root, char* name, int &param) const
{
	const char* str=attrValByName(root, name);
	if (str==NULL)
		return false;
	param=atoi(str);	
	return true;
};

bool Parser_XML::attrByNameToShort(xml_node &root, char* name, short &param) const
{
	const char* str=attrValByName(root, name);
	if (str==NULL)
		return false;
	int tmp=atoi(str);
	param=(short)(tmp);	
	return true;
};

bool Parser_XML::attrByNameToLong(xml_node &root, char* name, unsigned long &param) const
{
	const char* str=attrValByName(root, name);
	if (str==NULL)
		return false;
	param=atol(str);	
	return true;
};

bool Parser_XML::attrByNameToSLong(xml_node &root, char* name, long &param) const
{
	const char* str=attrValByName(root, name);
	if (str==NULL)
		return false;
	param=atol(str);	
	return true;
};

bool Parser_XML::attrByNameToRayPosDistrType(xml_node &root, char* name, rayPosDistrType &param) const
{
	const char* str=attrValByName(root, name);
	if (str==NULL)
		return false;
	param=asciiToRayPosDistrType(str);	
	return true;	
};

bool Parser_XML::attrByNameToRayDirDistrType(xml_node &root, char* name, rayDirDistrType &param) const
{
	const char* str=attrValByName(root, name);
	if (str==NULL)
		return false;
	param=asciiToRayDirDistrType(str);	
	return true;			
};

bool Parser_XML::attrByNameToDetOutFormat(xml_node &root, char* name, detOutFormat &param) const
{
	const char* str=attrValByName(root, name);
	if (str==NULL)
		return false;
	param=asciiToDetOutFormat(str);	
	return true;			
};

bool Parser_XML::attrByNameToBool(xml_node &root, char* name, bool &param) const
{
	const char* str=attrValByName(root, name);
	if (str==NULL)
		return false;
	if (!strcmp(str, "true"))
	{
		param=true;
		return true;
	}
	else
	{
		param=false;
		return true;
	}
	return false;			
};

bool Parser_XML::attrByNameToAccelType(xml_node &root, char* name, accelType &param) const
{
	const char* str=attrValByName(root, name);
	if (str==NULL)
		return false;
	param=asciiToAccelType(str);	
	return true;			
};