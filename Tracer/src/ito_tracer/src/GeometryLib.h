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

#ifndef GEOMETRYLIB_H
  #define GEOMETRYLIB_H

// include base class
//#include "Geometry.h"
// include the individual geometries
#include "PlaneSurface.h"
#include "differentialRayTracing/PlaneSurface_DiffRays.h"
#include "CylLenseSurface.h"
#include "SphericalSurface.h"
#include "parabolicSurface.h"
#include "AsphericalSurface.h"
#include "CylPipe.h"
#include "ConePipe.h"
#include "IdealLense.h"
#include "ApertureStop.h"
#include "SinusNormalSurface.h"
#include "Parser_XML.h"
#include "SphericalLense.h"
#include "microLensArraySurface.h"
#include "microLensArray.h"
#include "apertureArraySurface.h"
#include "stopArraySurface.h"
#include "cadObject.h"
#include "substrate.h"
#include "VolumeScattererBox.h"
#include <vector>

class GeometryFab
{
protected:

public:

	GeometryFab()
	{

	}
	~GeometryFab()
	{

	}

	bool createGeomInstFromXML(xml_node &node, simMode l_mode, vector<Geometry*> &geomVec) const;
};

#endif
