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

#ifndef GEOMETRYITEMLIB
#define GEOMETRYITEMLIB

#include "geomSphericalLenseItem.h"
#include "geomApertureStopItem.h"
#include "geomAsphericalSurfaceItem.h"
#include "geomConePipeItem.h"
#include "geomCylLenseSurfaceItem.h"
#include "geomCylPipeItem.h"
#include "geomIdealLenseItem.h"
#include "geomPlaneSurfaceItem.h"
#include "geomParabolicSurfaceItem.h"
#include "geomSphericalSurfaceItem.h"
#include "geomMicroLensArrayItem.h"

using namespace macrosim;

namespace macrosim 
{

/** @class GeometryItemLib
*   @brief class for visualizing objects from MacroSim
*   
*   The AddInModel supplies a widget showing the available objects from MacroSim with their name, filename, version and so on.
*/
class GeometryItemLib 
{

public:

	GeometryItemLib(void)
	{
	};
	~GeometryItemLib(void)
	{
	};

	GeometryItem* createGeometry(GeometryItem::GeomType type);

	//QString apertureTypeToString(const GeometryItem::ApertureType type) const;
	//ApertureType stringToApertureType(const QString str) const;
	QString geomTypeToString(const GeometryItem::GeomType type) const;
	GeometryItem::GeomType stringToGeomType(const QString str) const;
	GeometryItem::Abstract_MaterialType stringToGeomMatType(const QString str) const;
	QString geomMatTypeToString(const GeometryItem::Abstract_MaterialType type) const;
	QList<AbstractItem*> fillLibrary() const;


private:

};

}; //namespace macrosim

#endif