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

#ifndef FIELDITEMLIB
#define FIELDITEMLIB

#include "geomRayFieldItem.h"
#include "intensityFieldItem.h"
#include "scalarFieldItem.h"
#include "scalarGaussianFieldItem.h"
#include "scalarPlaneFieldItem.h"
#include "scalarSphericalFieldItem.h"
#include "scalarUserFieldItem.h"
#include "pathIntTissueFieldItem.h"

using namespace macrosim;

namespace macrosim 
{

/** @class FieldItemLib
*   @brief class for visualizing objects from MacroSim
*   
*   The AddInModel supplies a widget showing the available objects from MacroSim with their name, filename, version and so on.
*/
class FieldItemLib 
{

public:

	FieldItemLib(void)
	{
	};
	~FieldItemLib(void)
	{
	};

	FieldItem* createField(FieldItem::FieldType type);

	//QString apertureTypeToString(const FieldItem::ApertureType type) const;
	//ApertureType stringToApertureType(const QString str) const;
	QString fieldTypeToString(const FieldItem::FieldType type) const;
	FieldItem::FieldType stringToFieldType(const QString str) const;
	QList<AbstractItem*> fillLibrary() const;

private:

};

}; //namespace macrosim

#endif