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

#ifndef PARABOLICSURFACEITEM
#define PARABOLICSURFACEITEM

#include <qicon.h>

#include "QPropertyEditor/CustomTypes.h"
#include "GeometryItem.h"
#include <QtOpenGL\qglfunctions.h>
#include "glut.h"

using namespace macrosim;

namespace macrosim 
{

/** @class SphericalLenseItem
*   @brief class for visualizing objects from MacroSim
*   
*   The AddInModel supplies a widget showing the available objects from MacroSim with their name, filename, version and so on.
*/
class ParabolicSurfaceItem :
	public GeometryItem
{
	Q_OBJECT

	Q_PROPERTY(double radius READ getRadius WRITE setRadius DESIGNABLE true USER true);


public:

	ParabolicSurfaceItem(QString name="ParabolicSurface", QObject *parent=0);
	~ParabolicSurfaceItem(void);

	// functions for property editor
	double getRadius() const {return m_radius;};
	void setRadius(const double in) {m_radius=in; emit itemChanged(m_index, m_index);};

	bool writeToXML(QDomDocument &document, QDomElement &root) const;
	bool readFromXML(const QDomElement &node);

	void render(QMatrix4x4 &m, RenderOptions &options);

	Vec3f calcNormal(Vec3f vertex, Vec3f* neighbours, int nr);


private:

//	MaterialItem::MaterialType m_materialType;
	double m_radius;
};

}; //namespace macrosim

#endif