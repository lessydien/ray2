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

#include "scalarPlaneFieldItem.h"

using namespace macrosim;

ScalarPlaneFieldItem::ScalarPlaneFieldItem(QString name, QObject *parent) :
ScalarFieldItem(name, FieldType::SCALARPLANEWAVE, parent)
{
}

ScalarPlaneFieldItem::~ScalarPlaneFieldItem()
{
	m_childs.clear();
}

bool ScalarPlaneFieldItem::signalDataChanged() 
{

	return true;
};

bool ScalarPlaneFieldItem::writeToXML(QDomDocument &document, QDomElement &root) const 
{
	QDomElement node = document.createElement("field");

	if (!ScalarFieldItem::writeToXML(document, node))
		return false;

	node.setAttribute("fieldWidth.x", QString::number(m_fieldWidth.X));
	node.setAttribute("fieldWidth.y", QString::number(m_fieldWidth.Y));

	root.appendChild(node);
	return true;
}

bool ScalarPlaneFieldItem::readFromXML(const QDomElement &node)
{
	if (!ScalarFieldItem::readFromXML(node))
		return false;

	m_fieldWidth.X=node.attribute("fieldWidth.x").toDouble();
	m_fieldWidth.Y=node.attribute("fieldWidth.y").toDouble();

	return true;
}