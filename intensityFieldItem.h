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

#ifndef INTFIELDITEM
#define INTFIELDITEM

#include <qicon.h>

#include "QPropertyEditor/CustomTypes.h"
#include "fieldItem.h"

using namespace macrosim;

namespace macrosim 
{

/** @class FieldItem
*   @brief class for visualizing objects from MacroSim
*   
*   The AddInModel supplies a widget showing the available objects from MacroSim with their name, filename, version and so on.
*/
class IntensityFieldItem :
	public FieldItem
{
	Q_OBJECT

	Q_PROPERTY(QString filename READ getFileName WRITE setFileName DESIGNABLE true USER true);

public:

	IntensityFieldItem(QString name="IntensityField", QObject *parent=0);
	~IntensityFieldItem(void);

	// functions for property editor
	bool signalDataChanged();

	bool writeToXML(QDomDocument &document, QDomElement &root) const;
	bool readFromXML(const QDomElement &node);
	QString getFileName() const {return m_fieldDataFileName;};
	void setFileName(const QString in) {m_fieldDataFileName=in;};

private:
	ito::DataObject m_fieldData;
	QString m_fieldDataFileName;
	
}; // class RayFieldItem

}; //namespace macrosim

#endif