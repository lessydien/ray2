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

#ifndef MISCITEM_H
#define MISCITEM_H

#include <qicon.h>

#include "QPropertyEditor/CustomTypes.h"
#include "AbstractItem.h"

using namespace macrosim;

namespace macrosim 
{

/** @class FieldItem
*   @brief class for visualizing objects from MacroSim
*   
*   The AddInModel supplies a widget showing the available objects from MacroSim with their name, filename, version and so on.
*/
class MiscItem :
	public AbstractItem
{
	Q_OBJECT

	Q_PROPERTY(MiscType miscType READ getMiscType DESIGNABLE true USER true);

	Q_ENUMS(MiscType);

public:

	enum MiscType{GEOMETRYGROUP, UNDEFINED};

	MiscItem(QString name="name", MiscType type=UNDEFINED, QObject *parent=0);
	~MiscItem(void);

	// functions for property editor
	const MiscType getMiscType()  {return m_miscType;};
	void setMiscType(const MiscType in) {m_miscType=in;};

	virtual bool writeToXML(QDomDocument &document, QDomElement &root) const;
	virtual bool readFromXML(const QDomElement &node);

private:
	MiscType m_miscType;

};

}; //namespace macrosim

#endif