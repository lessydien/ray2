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

#include "rayFieldItem.h"
#include "materialItemLib.h"
#include "geometryItemLib.h"

using namespace macrosim;

RayFieldItem::RayFieldItem(QString name, FieldType type, QObject *parent) :
	FieldItem(name, type, parent),
//		m_root(Vec3d(0.0,0.0,0.0)),
		m_tilt(Vec3d(0.0,0.0,0.0)),
		m_rayDirection(Vec3d(0.0,0.0,1.0)),
		m_alphaMax(Vec2d(0.0,0.0)),
		m_alphaMin(Vec2d(0.0,0.0)),
		m_power(1.0),
		m_coherence(0.0),
		m_width(10),
		m_height(10),
		m_widthLayout(10),
		m_heightLayout(10),
		m_rayDirDistrType(RAYDIR_UNIFORM),
		m_rayPosDistrType(RAYPOS_GRID_RECT)
{
}

RayFieldItem::~RayFieldItem()
{
	m_childs.clear();
}

QString RayFieldItem::rayDirDistrTypeToString(const RayDirDistrType type) const
{
	QString str;
	switch (this->m_rayDirDistrType)
	{
	case RAYDIR_RAND_RECT:
		str="RAYDIR_RAND_RECT";
		break;
	case RAYDIR_RAND_RAD:
		str="RAYDIR_RAND_RAD";
		break;
	case RAYDIR_RANDNORM_RECT:
		str="RAYDIR_RANDNORM_RECT";
		break;
	case RAYDIR_RANDIMPAREA:
		str="RAYDIR_RANDIMPAREA";
		break;
	case RAYDIR_UNIFORM:
		str="RAYDIR_UNIFORM";
		break;
	case RAYDIR_GRID_RECT:
		str="RAYDIR_GRID_RECT";
		break;
	case RAYDIR_GRID_RECT_FARFIELD:
		str="RAYDIR_GRID_RECT_FARFIELD";
		break;
	case RAYDIR_GRID_RAD:
		str="RAYDIR_GRID_RAD";
		break;
	default:
		str="RAYDIR_UNKNOWN";
		break;
	}
	return str;
};

RayFieldItem::RayDirDistrType RayFieldItem::stringToRayDirDistrType(const QString str) const
{
	if (!str.compare("RAYDIR_RAND_RECT"))
		return RAYDIR_RAND_RECT;
	if (!str.compare("RAYDIR_RAND_RAD"))
		return RAYDIR_RAND_RAD;
	if (!str.compare("RAYDIR_RANDNORM_RECT"))
		return RAYDIR_RANDNORM_RECT;
	if (!str.compare("RAYDIR_RANDIMPAREA"))
		return RAYDIR_RANDIMPAREA;
	if (!str.compare("RAYDIR_UNIFORM"))
		return RAYDIR_UNIFORM;
	if (!str.compare("RAYDIR_GRID_RECT"))
		return RAYDIR_GRID_RECT;
	if (!str.compare("RAYDIR_GRID_RECT_FARFIELD"))
		return RAYDIR_GRID_RECT_FARFIELD;
	if (!str.compare("RAYDIR_GRID_RAD"))
		return RAYDIR_GRID_RAD;
	return RAYDIR_UNKNOWN;
};

QString RayFieldItem::rayPosDistrTypeToString(const RayPosDistrType type) const
{
	QString str;
	switch (this->m_rayPosDistrType)
	{
	case RAYPOS_RAND_RECT:
		str="RAYPOS_RAND_RECT";
		break;
	case RAYPOS_RAND_RECT_NORM:
		str="RAYPOS_RAND_RECT_NORM";
		break;
	case RAYPOS_GRID_RECT:
		str="RAYPOS_GRID_RECT";
		break;
	case RAYPOS_RAND_RAD:
		str="RAYPOS_RAND_RAD";
		break;
	case RAYPOS_RAND_RAD_NORM:
		str="RAYPOS_RAND_RAD_NORM";
		break;
	case RAYPOS_GRID_RAD:
		str="RAYPOS_GRID_RAD";
		break;
	default:
		str="RAYPOS_UNKNOWN";
		break;
	}
	return str;
};

RayFieldItem::RayPosDistrType RayFieldItem::stringToRayPosDistrType(const QString str) const
{
	if (!str.compare("RAYPOS_RAND_RECT"))
		return RAYPOS_RAND_RECT;
	if (!str.compare("RAYPOS_RAND_RECT_NORM"))
		return RAYPOS_RAND_RECT_NORM;
	if (!str.compare("RAYPOS_GRID_RECT"))
		return RAYPOS_GRID_RECT;
	if (!str.compare("RAYPOS_RAND_RAD"))
		return RAYPOS_RAND_RAD;
	if (!str.compare("RAYPOS_RAND_RAD_NORM"))
		return RAYPOS_RAND_RAD_NORM;
	if (!str.compare("RAYPOS_GRID_RAD"))
		return RAYPOS_GRID_RAD;
	return RAYPOS_UNKNOWN;
};

bool RayFieldItem::writeToXML(QDomDocument &document, QDomElement &node) const 
{
	// write base class
	if (!FieldItem::writeToXML(document, node))
		return false;

//	node.setAttribute("root.x", QString::number(m_root.X));
//	node.setAttribute("root.y", QString::number(m_root.Y));
//	node.setAttribute("root.z", QString::number(m_root.Z));
	node.setAttribute("tilt.x", QString::number(m_tilt.X));
	node.setAttribute("tilt.y", QString::number(m_tilt.Y));
	node.setAttribute("tilt.z", QString::number(m_tilt.Z));
	node.setAttribute("rayDirection.x", QString::number(m_rayDirection.X));
	node.setAttribute("rayDirection.y", QString::number(m_rayDirection.Y));
	node.setAttribute("rayDirection.z", QString::number(m_rayDirection.Z));
	node.setAttribute("alphaMax.x", QString::number(m_alphaMax.X));
	node.setAttribute("alphaMax.y", QString::number(m_alphaMax.Y));
	node.setAttribute("alphaMin.x", QString::number(m_alphaMin.X));
	node.setAttribute("alphaMin.y", QString::number(m_alphaMin.Y));
	node.setAttribute("coherence", QString::number(m_coherence));
	node.setAttribute("power", QString::number(m_power));
	node.setAttribute("width", QString::number(m_width));
	node.setAttribute("widthLayout", QString::number(m_widthLayout));
	node.setAttribute("heightLayout", QString::number(m_heightLayout));
	node.setAttribute("height", QString::number(m_height));
	node.setAttribute("rayDirDistrType", rayDirDistrTypeToString(m_rayDirDistrType));
	node.setAttribute("rayPosDistrType", rayPosDistrTypeToString(m_rayPosDistrType));

	// write material
	if (!this->getChild()->writeToXML(document,node))
		return false;

	return true;
}

bool RayFieldItem::readFromXML(const QDomElement &node)
{
	// read base class
	if (!FieldItem::readFromXML(node))
		return false;
	m_tilt.X=node.attribute("tilt.x").toDouble();
	m_tilt.Y=node.attribute("tilt.y").toDouble();
	m_tilt.Z=node.attribute("tilt.z").toDouble();
	m_rayDirection.X=node.attribute("rayDirection.x").toDouble();
	m_rayDirection.Y=node.attribute("rayDirection.y").toDouble();
	m_rayDirection.Z=node.attribute("rayDirection.z").toDouble();
	m_alphaMax.X=node.attribute("alphaMax.x").toDouble();
	m_alphaMax.Y=node.attribute("alphaMax.y").toDouble();
	m_alphaMin.X=node.attribute("alphaMin.x").toDouble();
	m_alphaMin.Y=node.attribute("alphaMin.y").toDouble();
	m_coherence=node.attribute("coherence").toDouble();
	m_power=node.attribute("power").toDouble();
	m_width=node.attribute("width").toDouble();
	m_height=node.attribute("height").toDouble();
	m_widthLayout=node.attribute("widthLayout").toDouble();
	m_heightLayout=node.attribute("heightLayout").toDouble();
	m_rayDirDistrType=stringToRayDirDistrType(node.attribute("rayDirDistrType"));
	m_rayPosDistrType=stringToRayPosDistrType(node.attribute("rayPosDistrType"));

	// read material
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