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

/**\file GeometryLib.h
* \brief header file that includes all the geometries defined in the application. If you define a new geometry, that's where you need to include the header to make it visible to the complete application
* 
*           
* \author Mauch
*/

#ifndef GEOMETRYLIB_GEOMRENDER_H
  #define GEOMETRYLIB_GEOMRENDER_H

// include base class
//#include "Geometry.h"
// include the individual geometries
#include "..\GeometryLib.h"
#include "PlaneSurface_GeomRender.h"
#include <vector>

class GeometryFab_GeomRender : public GeometryFab
{
protected:

public:

	GeometryFab_GeomRender()
	{

	}
	~GeometryFab_GeomRender()
	{

	}

	virtual bool createGeomInstFromXML(xml_node &node, SimParams simParams, vector<Geometry*> &geomVec) const;
};

#endif
