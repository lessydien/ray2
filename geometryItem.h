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

#ifndef GEOMETRYITEM
#define GEOMETRYITEM

#include <qicon.h>

#include "QPropertyEditor/CustomTypes.h"
#include "AbstractItem.h"
#include "materialItem.h"
#include <QtOpenGL\qglfunctions.h>
#include "glut.h"

using namespace macrosim;

namespace macrosim 
{

#ifndef PI
	#define PI 3.14159265358979
#endif

/** @class GeometryItem
*   @brief class for visualizing objects from MacroSim
*   
*   The AddInModel supplies a widget showing the available objects from MacroSim with their name, filename, version and so on.
*/
class GeometryItem :
	public AbstractItem
{
	Q_OBJECT

	Q_ENUMS(GeomType);
	Q_ENUMS(ApertureType);

	Q_PROPERTY(GeomType geomType READ getGeomType DESIGNABLE true USER true);
	Q_PROPERTY(Vec3d root READ getRoot WRITE setRoot DESIGNABLE true USER true);
	Q_PROPERTY(Vec3d tilt READ getTilt WRITE setTilt DESIGNABLE true USER true);
	Q_PROPERTY(Vec2d apertureRadius READ getApertureRadius WRITE setApertureRadius DESIGNABLE true USER true);
//	Q_PROPERTY(int geometryID READ getGeometryID WRITE setGeometryID DESIGNABLE true USER true);
	Q_PROPERTY(ApertureType apertureType READ getApertureType WRITE setApertureType DESIGNABLE true USER true);	
	
//	Q_PROPERTY(MaterialItem::Test test READ getTest DESIGNABLE true USER true);

public:
	enum GeomType {UNDEFINED, SPHERICALLENSE, CYLLENSESURF, SPHERICALSURFACE, PARABOLICSURFACE, PLANESURFACE, IDEALLENSE, APERTURESTOP, ASPHERICALSURF, CYLPIPE, CONEPIPE, DETECTOR, MICROLENSARRAY};
	enum ApertureType {RECTANGULAR, ELLIPTICAL, UNKNOWN};

	GeometryItem(QString name="name", GeomType type=UNDEFINED, QObject *parent=0);
	~GeometryItem(void);

	// functions for property editor
	Vec3d getRoot() const {return m_root;};
	void setRoot(const Vec3d root) {m_root=root; emit itemChanged(m_index, m_index);};
	Vec3d getTilt() const {return m_tilt;};
	void setTilt(const Vec3d tilt) {m_tilt=tilt; emit itemChanged(m_index, m_index);};
	Vec2d getApertureRadius() const {return m_apertureRadius; };
	void setApertureRadius(const Vec2d ApertureRadius) {m_apertureRadius=ApertureRadius; emit itemChanged(m_index, m_index);};
	GeomType getGeomType() const {return m_geomType;};
	void setGeomType(const GeomType type) {m_geomType=type; emit itemChanged(m_index, m_index);};
	ApertureType getApertureType() const {return m_apertureType;};
	void setApertureType(const ApertureType type) {m_apertureType=type; emit itemChanged(m_index, m_index);};
	int getGeometryID() const {return m_geometryID;};
	void setGeometryID(const int ID) {m_geometryID=ID; emit itemChanged(m_index, m_index);};
	MaterialItem::Test getTest() const {return m_test;};

	bool signalDataChanged();
	//MaterialItem::MaterialType getMaterialType() const {return m_materialType;};
	//void setMaterialType(const MaterialItem::MaterialType type) {m_materialType = type;};

	QString apertureTypeToString(const ApertureType type) const;
	ApertureType stringToApertureType(const QString str) const;
	//QString geomTypeToString(const GeomType type) const;
	//GeomType stringToGeomType(const QString str) const;

	virtual bool writeToXML(QDomDocument &document, QDomElement &root) const;
	virtual bool readFromXML(const QDomElement &node);
	virtual void render(QMatrix4x4 &m, RenderOptions &options);

	//AbstractItem* getChild() const {return m_childs[0];};
	MaterialItem* getChild() const 
	{ 
		if (m_childs.empty())
			return NULL;
		else
		{
			if (m_childs[0]->getObjectType() != MATERIAL)
				return NULL;
			else
				return reinterpret_cast<MaterialItem*>(m_childs[0]);
		}
	};

	virtual void setChild(AbstractItem* child) 
	{
		if (child->getObjectType() == MATERIAL)
		{
			// so far we only allow one material per geometry. Therefore we clear the list if there are already entries present
			if (!m_childs.empty())
				m_childs.clear();

			m_childs.append(child);

			connect(child, SIGNAL(itemChanged(const QModelIndex &, const QModelIndex &)), this, SLOT(changeItem(const QModelIndex &, const QModelIndex &)));
		}
	}

	virtual Vec3f calcNormal(Vec3f vertex, Vec3f* neighbours, int nr);


private:
	
	ApertureType m_apertureType;
	GeomType m_geomType;
	Vec3d m_root;
	Vec3d m_tilt;
	Vec2d m_apertureRadius;
	int m_geometryID;
	//MaterialItem::MaterialType m_materialType;
	MaterialItem::Test m_test;

signals:
//	void itemChanged(const QModelIndex &topLeft, const QModelIndex &bottomRight);

public slots:
//	void changeItem(const QModelIndex &topLeft, const QModelIndex &bottomRight);
};

}; //namespace macrosim

#endif