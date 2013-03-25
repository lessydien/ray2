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

#ifndef CYLPIPEITEM
#define CYLPIPEITEM

#include <qicon.h>

#include "QPropertyEditor/CustomTypes.h"
#include "GeometryItem.h"

using namespace macrosim;

namespace macrosim 
{

/** @class SphericalLenseItem
*   @brief class for visualizing objects from MacroSim
*   
*   The AddInModel supplies a widget showing the available objects from MacroSim with their name, filename, version and so on.
*/
class CylPipeItem :
	public GeometryItem
{
	Q_OBJECT

	Q_PROPERTY(double radius READ getRadius WRITE setRadius DESIGNABLE true USER true);
	Q_PROPERTY(double thickness READ getThickness WRITE setThickness DESIGNABLE true USER true);
	//Q_PROPERTY(MaterialItem::MaterialType Material READ getMaterial WRITE setMaterial DESIGNABLE true USER true);


public:

	CylPipeItem(QString name="CylPipe", QObject *parent=0);
	~CylPipeItem(void);

	// functions for property editor
	double getRadius() const {return m_radius;};
	void setRadius(const double in) {m_radius=in; emit itemChanged(m_index, m_index);};
	double getThickness() const {return m_thickness;};
	void setThickness(const double in) {m_thickness=in; emit itemChanged(m_index, m_index);};

	bool writeToXML(QDomDocument &document, QDomElement &root) const;
	bool readFromXML(const QDomElement &node);
	void render(QMatrix4x4 &m, RenderOptions &options);

	Vec3f calcNormal(Vec3f vertex, Vec3f* neighbours, int nr);

//	MaterialItem::MaterialType getMaterial() const {return m_materialType;};
//	void setMaterial(const MaterialItem::MaterialType type) {m_materialType=type;};

private:

//	MaterialItem::MaterialType m_materialType;
	double m_radius;
	double m_thickness;
};

}; //namespace macrosim

#endif