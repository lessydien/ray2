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

#ifndef COATINGITEMLIB
#define COATINGITEMLIB

#include "coatingNoItem.h"
#include "coatingNumCoeffsItem.h"
#include "materialItem.h"

namespace macrosim 
{

/** @class CoatingItemLib
*   @brief class for visualizing objects from MacroSim
*   
*   The AddInModel supplies a widget showing the available objects from MacroSim with their name, filename, version and so on.
*/
class CoatingItemLib 
{

public:
	CoatingItemLib(void) {};
	~CoatingItemLib(void) {};
	CoatingItem* createCoating(CoatingItem::CoatingType type);

	QString coatingTypeToString(const CoatingItem::CoatingType type) const;
	CoatingItem::CoatingType stringToCoatingType(const QString str) const;

	CoatingItem::CoatingType matCoatTypeToCoatType(const MaterialItem::Mat_CoatingType type) const;
	MaterialItem::Mat_CoatingType coatTypeToMatCoatType(const CoatingItem::CoatingType type) const;

private:

};

}; //namespace macrosim

#endif