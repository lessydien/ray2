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

/**\file GeometryLib_GeomRender.h
* \brief header file that includes all the geometries defined in the application. If you define a new geometry, that's where you need to include the header to make it visible to the complete application
* 
*           
* \author Mauch
*/

#ifndef GEOMETRYLIB_GEOMRENDER_H
  #define GEOMETRYLIB_GEOMRENDER_H

// include base class
#include "..\GeometryLib.h"
// include the individual geometries
#include "PlaneSurface_GeomRender.h"
#include "SphericalSurface_GeomRender.h"
#include "SphericalLense_GeomRender.h"
#include "AsphericalSurface_GeomRender.h"
#include "CylPipe_GeomRender.h"
#include "ConePipe_GeomRender.h"
#include "IdealLense_GeomRender.h"
#include "ApertureStop_GeomRender.h"
#include "cadObject_GeomRender.h"

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

	bool createGeomInstFromXML(xml_node &node, SimParams simParams, vector<Geometry*> &geomVec) const;
};


#endif
