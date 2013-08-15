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

#ifndef DETECTORITEMLIB
#define DETECTORITEMLIB

#include "detectorIntensityItem.h"
#include "detectorFieldItem.h"
#include "detectorRayDataItem.h"

using namespace macrosim;

namespace macrosim 
{

/** @class DetectorItemLib
*   @brief class for visualizing objects from MacroSim
*   
*   The AddInModel supplies a widget showing the available objects from MacroSim with their name, filename, version and so on.
*/
class DetectorItemLib 
{

public:

	DetectorItemLib(void)
	{
	};
	~DetectorItemLib(void)
	{
	};

	DetectorItem* createDetector(DetectorItem::DetType type);
	QList<AbstractItem*> fillLibrary() const;

	QString detTypeToString(const DetectorItem::DetType type) const;
	DetectorItem::DetType stringToDetType(const QString str) const;
	QString detOutFormatToString(const DetectorItem::DetOutFormat format) const;
	DetectorItem::DetOutFormat stringToDetOutFormat(const QString str) const;



private:

};

}; //namespace macrosim

#endif