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

#include "detectorItem.h"
#include "detectorItemLib.h"

using namespace macrosim;

DetectorItem::DetectorItem(QString name, DetType type, QObject *parent) :
	AbstractItem(DETECTOR, name, parent),
	m_detType(type),
	m_root(Vec3d(0,0,0)),
	m_tilt(Vec3d(0,0,0))
{
}

DetectorItem::~DetectorItem()
{
	m_childs.clear();
}

void DetectorItem::simulationFinished(ito::DataObject field)
{
	this->m_resultField=field;
}


bool DetectorItem::writeToXML(QDomDocument &document, QDomElement &node) const 
{
	if (!AbstractItem::writeToXML(document, node))
		return false;
	node.setAttribute("root.x", m_root.X);
	node.setAttribute("root.y", m_root.Y);
	node.setAttribute("root.z", m_root.Z);
	node.setAttribute("tilt.x", QString::number(m_tilt.X));
	node.setAttribute("tilt.y", QString::number(m_tilt.Y));
	node.setAttribute("tilt.z", QString::number(m_tilt.Z));
	node.setAttribute("apertureHalfWidth.x", QString::number(m_apertureHalfWidth.X));
	node.setAttribute("apertureHalfWidth.y", QString::number(m_apertureHalfWidth.Y));
	DetectorItemLib l_detItemLib;
	node.setAttribute("detType", l_detItemLib.detTypeToString(m_detType));
	node.setAttribute("detOutFormat", l_detItemLib.detOutFormatToString(m_detOutFormat));
	node.setAttribute("fileName", m_fileName);
	return true;
}

bool DetectorItem::readFromXML(const QDomElement &node)
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
	m_apertureHalfWidth.X=node.attribute("apertureHalfWidth.x").toDouble();
	m_apertureHalfWidth.Y=node.attribute("apertureHalfWidth.y").toDouble();
	QString detTypeStr=node.attribute("detType");
	DetectorItemLib l_detItemLib;
	m_detType=l_detItemLib.stringToDetType(detTypeStr);
	QString detOutFormatStr=node.attribute("detOutFormat");
	m_detOutFormat=l_detItemLib.stringToDetOutFormat(detOutFormatStr);
	m_fileName=node.attribute("fileName");
	
	return true;
}