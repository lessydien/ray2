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

#ifndef SCALARSPHERICALFIELDITEM
#define SCALARSPHERICALFIELDITEM

#include <qicon.h>

#include "QPropertyEditor/CustomTypes.h"
#include "scalarFieldItem.h"

using namespace macrosim;

namespace macrosim 
{

/** @class FieldItem
*   @brief class for visualizing objects from MacroSim
*   
*   The AddInModel supplies a widget showing the available objects from MacroSim with their name, filename, version and so on.
*/
class ScalarSphericalFieldItem :
	public ScalarFieldItem
{
	Q_OBJECT

	Q_PROPERTY(Vec2d radius READ getRadius WRITE setRadius DESIGNABLE true USER true);
	Q_PROPERTY(Vec2d numApt READ getNumApt WRITE setNumApt DESIGNABLE true USER true);	

public:

	ScalarSphericalFieldItem(QString name="ScalarSphericalField", QObject *parent=0);
	~ScalarSphericalFieldItem(void);

	// functions for property editor
	Vec2d getRadius() const {return m_radius;};
	void setRadius(Vec2d in) {m_radius=in;};
	Vec2d getNumApt() const {return m_numApt;};
	void setNumApt(Vec2d in) {m_numApt=in;};

	bool signalDataChanged();

	bool writeToXML(QDomDocument &document, QDomElement &root) const;
	bool readFromXML(const QDomElement &node);

	void render(QMatrix4x4 &m, RenderOptions &options);

private:
	Vec2d m_radius;
	Vec2d m_numApt;

}; // class RayFieldItem

}; //namespace macrosim

#endif