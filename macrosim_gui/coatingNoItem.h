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

#ifndef COATINGNOITEM
#define COATINGNOITEM

#include <qicon.h>

#include "QPropertyEditor/CustomTypes.h"
#include "CoatingItem.h"


namespace macrosim 
{

/** @class CoatingItem
*   @brief class for visualizing objects from MacroSim
*   
*   The AddInModel supplies a widget showing the available objects from MacroSim with their name, filename, version and so on.
*/
class CoatingNoItem :
	public CoatingItem
{
	Q_OBJECT


public:

	CoatingNoItem(QString name="noCoating", QObject *parent=0);
	~CoatingNoItem(void);

	bool writeToXML(QDomDocument &document, QDomElement &root) const;
	bool readFromXML(const QDomElement &node);

	// functions for property editor

private:

};

}; //namespace macrosim

#endif