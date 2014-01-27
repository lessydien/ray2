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

/**\file MicroLensArray.cpp
* \brief microLensArray
* 
*           
* \author Mauch
*/

#include "MicroLensArray.h"
#include "GeometryLib.h"

#include <iostream>

#include "Parser_XML.h"


/**
 * \detail parseXml 
 *
 * sets the parameters of the detector according to the given xml node
 *
 * \param[in] xml_node &geometry
 * 
 * \return geometryError
 * \sa 
 * \remarks 
 * \author Mauch
 */
geometryError MicroLensArray::parseXml(pugi::xml_node &geometry, SimParams simParams, vector<Geometry*> &geomVec)
{
	Parser_XML l_parser;

	// get child geometries
	vector<pugi::xml_node> *l_pXmlGeomList;
	l_pXmlGeomList=l_parser.childsByTagName(geometry, "surface");

	if (! ((l_pXmlGeomList->size() == 3) || (l_pXmlGeomList->size() == 6)) )
	{
		std::cout << "error in MicroLensArray.parseXML(): xml_node contains not enough subnodes" << std::endl;
		return GEOM_ERR;
	}

	GeometryFab l_GeomFab;
	for (int i=0; i<l_pXmlGeomList->size(); i++)
	{
		vector<Geometry*> t_tmpGeomVec;
		const char* str;
		str=l_parser.attrValByName(l_pXmlGeomList->at(i), "faceType");
		if (str==NULL)
		{
			std::cout << "error in MicroLensArray.parseXML(): faceType is not defined for face " << i << std::endl;
			return GEOM_ERR;
		}
		// if we are in sequential mode, we ignore the side face
        if ( ((simParams.traceMode==TRACE_SEQ) ) && (strcmp(str,"SIDEFACE")==0) )
			;
		else
			if (!l_GeomFab.createGeomInstFromXML(l_pXmlGeomList->at(i), simParams, geomVec))
			{
				std::cout << "error in MicroLensArray.parseXML(): geomFab.createGeomInstFromXml() returned an error." << std::endl;
				return GEOM_ERR;
			}
	}

	delete l_pXmlGeomList;

	return GEOM_NO_ERR;
};
