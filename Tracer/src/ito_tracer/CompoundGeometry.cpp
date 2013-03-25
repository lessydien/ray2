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

/**\file CompoundGeometry.cpp
* \brief base class for all geometries
* 
*           
* \author Mauch
*/

#include "CompoundGeometry.h"
#include "myUtil.h"
#include "sampleConfig.h"
#include <stdio.h>
#include <string.h>
#include <iostream>
#include "Material.h"
#include "MaterialLib.h"

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
geometryError CompoundGeometry::parseXml(pugi::xml_node &geometry, vector<Geometry*> &geomVec)
{
	std::cout << "error in CompoundGeometry.parseXml(): not defined" << std::endl;
	return GEOM_ERR;
}

