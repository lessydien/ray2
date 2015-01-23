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

#ifndef SCALARFIELDITEM
#define SCALARFIELDITEM

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
class ScalarFieldItem :
	public FieldItem
{
	Q_OBJECT

	Q_PROPERTY(double amplMax READ getAmplMax WRITE setAmplMax DESIGNABLE true USER true);
	Q_PROPERTY(Vec2i numberOfPixels READ getNumberOfPixels WRITE setNumberOfPixels DESIGNABLE true USER true);

public:

	ScalarFieldItem(QString name="ScalarField", FieldItem::FieldType type=FieldItem::UNDEFINED, QObject *parent=0);
	~ScalarFieldItem(void);

	// functions for property editor
	double getAmplMax() const {return m_amplMax;};
	void setAmplMax(double in) {m_amplMax=in;};
	Vec2i getNumberOfPixels() const {return m_numberOfPixels;};
	void setNumberOfPixels(Vec2i in) {m_numberOfPixels=in;};

	bool signalDataChanged();

	bool writeToXML(QDomDocument &document, QDomElement &root) const;
	bool readFromXML(const QDomElement &node);

	virtual void render(QMatrix4x4 &m, RenderOptions &options) {};

private:
	ito::DataObject m_fieldData;
	double m_amplMax;
	Vec2i m_numberOfPixels;
}; // class RayFieldItem

}; //namespace macrosim

#endif