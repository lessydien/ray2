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

#include "geometryItem.h"
#include "geometryItemLib.h"
#include "materialItemLib.h"
#include <iostream>
using namespace std;

using namespace macrosim;

GeometryItem::GeometryItem(QString name, GeomType type, QObject *parent) :
	AbstractItem(GEOMETRY, name, parent),
	m_geometryID(0),
	m_geomGroupID(0),
	m_geomType(type),
	m_apertureType(ELLIPTICAL),
	m_apertureRadius(2.0,2.0),
	m_root(0.0,0.0,0.0),
	m_tilt(0.0,0.0,0.0)
{
	// default is an absorbing material
	AbstractItem::setMaterialType(AbstractItem::ABSORBING);
	this->setChild(new MaterialAbsorbingItem);
}

GeometryItem::~GeometryItem()
{
	m_childs.clear();
}

QString GeometryItem::apertureTypeToString(const ApertureType type) const
{
	QString str;
	switch (this->m_apertureType)
	{
	case RECTANGULAR:
		str="RECTANGULAR";
		break;
	case ELLIPTICAL:
		str="ELLIPTICAL";
		break;
	default:
		str="UNKNOWN";
		break;
	}
	return str;
};


//GeometryItem::ApertureType GeometryItem::stringToApertureType(const QString str) const
//{
//	if (!str.compare("RECTANGULAR"))
//		return RECTANGULAR;
//	if (!str.compare("ELLIPTICAL"))
//		return ELLIPTICAL;
//	return UNKNOWN;
//};

GeometryItem::ApertureType GeometryItem::stringToApertureType(const QString str) const
{
	if (!str.compare("RECTANGULAR"))
		return RECTANGULAR;
	if (!str.compare("ELLIPTICAL"))
		return ELLIPTICAL;
	return UNKNOWN;
};

//QString GeometryItem::geomTypeToString(const GeomType type) const
//{
//	QString str;
//	switch (this->m_geomType)
//	{
//	case SPHERICALLENSE:
//		str="SPHERICALLENSE";
//		break;
//	case CYLLENSESURF:
//		str="CYLLENSESURF";
//		break;
//	case SPHERICALSURFACE:
//		str="SPHERICALSURFACE";
//		break;
//	case PLANESURFACE:
//		str="PLANESURFACE";
//		break;
//	case IDEALLENSE:
//		str="IDEALLENSE";
//		break;
//	case APERTURESTOP:
//		str="APERTURESTOP";
//		break;
//	case ASPHERICALSURF:
//		str="ASPHERICALSURF";
//		break;
//	case CYLPIPE:
//		str="CYLPIPE";
//		break;
//	case CONEPIPE:
//		str="CONEPIPE";
//		break;
//	case DETECTOR:
//		str="DETECTOR";
//		break;
//	default:
//		str="UNKNOWN";
//		break;
//	}
//	return str;
//};
//
//GeometryItem::GeomType GeometryItem::stringToGeomType(const QString str) const
//{
//	if (!str.compare("SPHERICALLENSE") )
//		return SPHERICALLENSE;
//	if (!str.compare("CYLLENSESURF") == 0)
//		return CYLLENSESURF;
//	if (!str.compare("SPHERICALSURFACE") == 0)
//		return SPHERICALSURFACE;
//	if (!str.compare("PLANESURFACE") == 0)
//		return PLANESURFACE;
//	if (!str.compare("APERTURESTOP") == 0)
//		return APERTURESTOP;
//	if (!str.compare("ASPHERICALSURF") == 0)
//		return ASPHERICALSURF;
//	if (!str.compare("CYLPIPE") == 0)
//		return CYLPIPE;
//	if (!str.compare("CONEPIPE") == 0)
//		return CONEPIPE;
//	if (!str.compare("DETECTOR") == 0)
//		return DETECTOR;
//	return UNDEFINED;
//};

bool GeometryItem::writeToXML(QDomDocument &document, QDomElement &node) const 
{
	if (!AbstractItem::writeToXML(document, node))
		return false;
	node.setAttribute("root.x", QString::number(m_root.X));
	node.setAttribute("root.y", QString::number(m_root.Y));
	node.setAttribute("root.z", QString::number(m_root.Z));
	node.setAttribute("tilt.x", QString::number(m_tilt.X));
	node.setAttribute("tilt.y", QString::number(m_tilt.Y));
	node.setAttribute("tilt.z", QString::number(m_tilt.Z));
	node.setAttribute("apertureRadius.x", QString::number(m_apertureRadius.X));
	node.setAttribute("apertureRadius.y", QString::number(m_apertureRadius.Y));
	node.setAttribute("apertureType", apertureTypeToString(m_apertureType));
	//node.setAttribute("geometryID", QString::number(m_geometryID));
	node.setAttribute("geometryID", QString::number(m_index.row()));
	if (m_render)
		node.setAttribute("render", "true");
	else
		node.setAttribute("render", "false");
	
	// add material
	// geometries must have exactly one child
	if (m_childs.count() != 1)
		return false;
	if (!this->getChild()->writeToXML(document, node))
		return false;
	return true;
}

bool GeometryItem::readFromXML(const QDomElement &node)
{
	// read from base class
	if (!AbstractItem::readFromXML(node))
		return false;
	m_root.X=node.attribute("root.x").toDouble();
	m_root.Y=node.attribute("root.y").toDouble();
	m_root.Z=node.attribute("root.z").toDouble();
	m_tilt.X=node.attribute("tilt.x").toDouble();
	m_tilt.Y=node.attribute("tilt.y").toDouble();
	m_tilt.Z=node.attribute("tilt.z").toDouble();
	m_apertureRadius.X=node.attribute("apertureRadius.x").toDouble();
	m_apertureRadius.Y=node.attribute("apertureRadius.y").toDouble();
	m_apertureType=stringToApertureType(node.attribute("apertureType"));
	m_geometryID=node.attribute("geometryID").toInt();
	if (!node.attribute("render").compare("true"))
		m_render=true;
	else
		m_render=false;

	// look for material
	QDomNodeList l_matNodeList=node.elementsByTagName("material");
	if (l_matNodeList.count()==0)
		return false;
	QDomElement l_matElementXML=l_matNodeList.at(0).toElement();
	MaterialItemLib l_materialLib;
	MaterialItem l_materialItem;
	QString l_matTypeStr=l_matElementXML.attribute("materialType");
	MaterialItem* l_pMaterialItem = l_materialLib.createMaterial(l_materialLib.stringToMaterialType(l_matTypeStr));
	if (!l_pMaterialItem->readFromXML(l_matElementXML))
		return false;

	GeometryItemLib l_geomItemLib;
	m_materialType=l_geomItemLib.stringToGeomMatType(l_matTypeStr);

	this->setChild(l_pMaterialItem);

	return true;
}

bool GeometryItem::signalDataChanged()
{
	// get materialitem 
	MaterialItem* l_pMaterialItem=reinterpret_cast<MaterialItem*>(this->getChild());
	// if material changed, we append the new material to our geometry
	if (this->getMaterialType() != l_pMaterialItem->getMaterialType())
	{
		// create new material according to materialType of geometryItem
		MaterialItemLib l_materialLib;
		GeometryItemLib l_geomLib;
		// create materialType from Abstract_MaterialType (this is ugly...)
		QString str=l_geomLib.geomMatTypeToString(this->getMaterialType());
		MaterialItem::MaterialType l_matType=l_materialLib.stringToMaterialType(str);
		this->setChild(l_materialLib.createMaterial(l_matType));
	}
	// if not, we pass the signal to the material
	else
	{
		if (!this->getChild()->signalDataChanged())
		{
			cout << "error in GeometryItem.signalDataChanged(): material.signalDataChanged() returned an error." << endl;
			return false;
		}
	}

	MaterialItemLib l_matItemLib;
	MaterialItem::MaterialType l_matType=this->getChild()->getMaterialType();
	QString str = l_matItemLib.materialTypeToString(l_matType);
	GeometryItemLib l_geomItemLib;
	m_materialType=l_geomItemLib.stringToGeomMatType(str);
	return true;
}

//void GeometryItem::changeItem(const QModelIndex &topLeft, const QModelIndex &bottomRight)
//{
//	// check wether scatterType or CoatingType 
//	emit itemChanged(m_index, m_index);
//}

void GeometryItem::render(QMatrix4x4 &m, RenderOptions &options)
{

}

Vec3f GeometryItem::calcNormal(Vec3f vertex, Vec3f* neighbours, int nr)
{
	// calc individual normals from cross products
	Vec3f normal=cross(neighbours[nr-1]-vertex, neighbours[0]-vertex);
	for (int i=1; i<nr; i++)
	{
		// summ normals
		normal=normal+cross(neighbours[i-1]-vertex, neighbours[i]-vertex);
	}
	// average
	normal=normal/nr;
	// normalize
	normal=normal/(sqrt(normal*normal));
	return normal;
}