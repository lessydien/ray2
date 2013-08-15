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

#include "fieldItem.h"
#include "materialItemLib.h"
#include "fieldItemLib.h"

using namespace macrosim;

FieldItem::FieldItem(QString name, FieldType type, QObject *parent) :
	AbstractItem(FIELD, name, parent),
	m_fieldType(type),
	m_apertureHalfWidth(Vec2d(0.0,0.0)),
	m_lambda(630.0),
	m_root(Vec3d(0,0,0)),
	m_tilt(Vec3d(0,0,0))
{
	AbstractItem::setMaterialType(AbstractItem::REFRACTING);
	this->setChild(new MaterialRefractingItem);
}

FieldItem::~FieldItem()
{
	m_childs.clear();
}

//QString FieldItem::fieldTypeToString(const FieldType type) const
//{
//	QString str;
//	switch (this->m_fieldType)
//	{
//	case RAYFIELD:
//		str="RAYFIELD";
//		break;
//	case GEOMRAYFIELD:
//		str="GEOMRAYFIELD";
//		break;
//	case DIFFRAYFIELD:
//		str="DIFFRAYFIELD";
//		break;
//	case DIFFRAYFIELDRAYAIM:
//		str="DIFFRAYFIELDRAYAIM";
//		break;
//	case PATHTRACERAYFIELD:
//		str="PATHTRACERAYFIELD";
//		break;
//	case SCALARFIELD:
//		str="SCALARFIELD";
//		break;
//	case SCALARPLANEWAVE:
//		str="SCALARPLANEWAVE";
//		break;
//	case SCALARSPHERICALWAVE:
//		str="SCALARSPHERICALWAVE";
//		break;
//	case SCALARGAUSSIANWAVE:
//		str="SCALARGAUSSIANWAVE";
//		break;
//	case SCALARUSERWAVE:
//		str="SCALARUSERWAVE";
//		break;
//	case VECFIELD:
//		str="VECFIELD";
//		break;
//	default:
//		str="UNKNOWN";
//		break;
//	}
//	return str;
//};
//
//FieldItem::FieldType FieldItem::stringToFieldType(const QString str) const
//{
//	if (!str.compare("RAYFIELD"))
//		return RAYFIELD;
//	if (!str.compare("GEOMRAYFIELD"))
//		return GEOMRAYFIELD;
//	if (!str.compare("DIFFRAYFIELD"))
//		return DIFFRAYFIELD;
//	if (!str.compare("DIFFRAYFIELDRAYAIM"))
//		return DIFFRAYFIELDRAYAIM;
//	if (!str.compare("PATHTRACERAYFIELD"))
//		return PATHTRACERAYFIELD;
//	if (!str.compare("SCALARFIELD"))
//		return SCALARFIELD;
//	if (!str.compare("VECFIELD"))
//		return VECFIELD;
//	return UNDEFINED;
//};

bool FieldItem::writeToXML(QDomDocument &document, QDomElement &node) const 
{
	if (!AbstractItem::writeToXML(document, node))
		return false;
	FieldItemLib l_fieldLib;
	node.setAttribute("fieldType", l_fieldLib.fieldTypeToString(m_fieldType));
	node.setAttribute("lambda", QString::number(m_lambda));
	node.setAttribute("apertureHalfWidth.x", QString::number(m_apertureHalfWidth.X));
	node.setAttribute("apertureHalfWidth.y", QString::number(m_apertureHalfWidth.Y));
	node.setAttribute("tilt.x", QString::number(m_tilt.X));
	node.setAttribute("tilt.y", QString::number(m_tilt.Y));
	node.setAttribute("tilt.z", QString::number(m_tilt.Z));
	node.setAttribute("root.x", m_root.X);
	node.setAttribute("root.y", m_root.Y);
	node.setAttribute("root.z", m_root.Z);


	return true;
}

bool FieldItem::readFromXML(const QDomElement &node)
{
	if (!AbstractItem::readFromXML(node))
		return false;FieldItemLib l_fieldLib;
	m_fieldType=l_fieldLib.stringToFieldType(node.attribute("fieldType"));
	m_lambda=node.attribute("lambda").toDouble();
	m_apertureHalfWidth.X=node.attribute("apertureHalfWidth.x").toDouble();
	m_apertureHalfWidth.Y=node.attribute("apertureHalfWidth.y").toDouble();
	m_tilt.X=node.attribute("tilt.x").toDouble();
	m_tilt.Y=node.attribute("tilt.y").toDouble();
	m_tilt.Z=node.attribute("tilt.z").toDouble();
	m_root.X=node.attribute("root.x").toDouble();
	m_root.Y=node.attribute("root.y").toDouble();
	m_root.Z=node.attribute("root.z").toDouble();


	return true;
}