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

#ifndef SPHERICALLENSEITEM
#define SPHERICALLENSEITEM

#include <qicon.h>

#include "QPropertyEditor/CustomTypes.h"
#include "GeometryItem.h"
#include <QtOpenGL\qglfunctions.h>
//#include "glut.h"

using namespace macrosim;

namespace macrosim 
{

/** @class SphericalLenseItem
*   @brief class for visualizing objects from MacroSim
*   
*   The AddInModel supplies a widget showing the available objects from MacroSim with their name, filename, version and so on.
*/
class SphericalLenseItem :
	public GeometryItem
{
	Q_OBJECT

	Q_PROPERTY(double radius1 READ getRadius1 WRITE setRadius1 DESIGNABLE true USER true);
	Q_PROPERTY(double radius2 READ getRadius2 WRITE setRadius2 DESIGNABLE true USER true);
	Q_PROPERTY(Vec2d apertureRadius2 READ getApertureRadius2 WRITE setApertureRadius2 DESIGNABLE true USER true);
	Q_PROPERTY(double thickness READ getThickness WRITE setThickness DESIGNABLE true USER true);
	//Q_PROPERTY(MaterialItem::MaterialType Material READ getMaterial WRITE setMaterial DESIGNABLE true USER true);


public:

	SphericalLenseItem(QString name="SphericalLense", QObject *parent=0);
	~SphericalLenseItem(void);

	// functions for property editor
	double getRadius1() const {return m_radius1;};
	void setRadius1(const double radius) {m_radius1=radius; emit itemChanged(m_index, m_index);};
	double getRadius2() const {return m_radius2;};
	void setRadius2(const double radius) {m_radius2=radius; emit itemChanged(m_index, m_index);};
	Vec2d getApertureRadius2() const {return m_apertureRadius2;};
	void setApertureRadius2(Vec2d in) {m_apertureRadius2=in; emit itemChanged(m_index, m_index);};
	double getThickness() const {return m_thickness;};
	void setThickness(const double thickness) {m_thickness=thickness; emit itemChanged(m_index, m_index);};

	bool writeToXML(QDomDocument &document, QDomElement &root) const;
	bool readFromXML(const QDomElement &node);

	void render(QMatrix4x4 &m, RenderOptions &options);
	Vec3f calcNormal(Vec3f vertex, Vec3f* neighbours, int nr);

//	MaterialItem::MaterialType getMaterial() const {return m_materialType;};
//	void setMaterial(const MaterialItem::MaterialType type) {m_materialType=type;};

private:
	Vec3f calcNormalSide(Vec3f vertex);
	Vec3f calcNormalBack(Vec3f vertex);

//	MaterialItem::MaterialType m_materialType;
	double m_radius1;
	double m_radius2;
	Vec2d m_apertureRadius2;
	double m_thickness;
};

}; //namespace macrosim

#endif