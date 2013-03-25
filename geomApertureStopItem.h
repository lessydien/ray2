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

#ifndef APERTURESTOPITEM
#define APERTURESTOPITEM

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
class ApertureStopItem :
	public GeometryItem
{
	Q_OBJECT

	Q_PROPERTY(Vec2d apertureStopRadius READ getApertureStopRadius WRITE setApertureStopRadius DESIGNABLE true USER true);
	//Q_PROPERTY(MaterialItem::MaterialType Material READ getMaterial WRITE setMaterial DESIGNABLE true USER true);


public:

	ApertureStopItem(QString name="ApertureStop", QObject *parent=0);
	~ApertureStopItem(void);

	// functions for property editor
	Vec2d getApertureStopRadius() const {return m_apertureStopRadius;};
	void setApertureStopRadius(const Vec2d in) {m_apertureStopRadius=in; emit itemChanged(m_index, m_index);};

	bool writeToXML(QDomDocument &document, QDomElement &root) const;
	bool readFromXML(const QDomElement &node);
	void render(QMatrix4x4 &m, RenderOptions &options);

	Vec3f calcNormal(Vec3f vertex, Vec3f* neighbours, int nr);

//	MaterialItem::MaterialType getMaterial() const {return m_materialType;};
//	void setMaterial(const MaterialItem::MaterialType type) {m_materialType=type;};

private:

//	MaterialItem::MaterialType m_materialType;
	Vec2d m_apertureStopRadius;
};

}; //namespace macrosim

#endif