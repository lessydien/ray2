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

#include "scalarFieldItem.h"
#include "fieldItemLib.h"

using namespace macrosim;

ScalarFieldItem::ScalarFieldItem(QString name, FieldItem::FieldType type, QObject *parent) :
	FieldItem(name, type, parent),
		m_amplMax(1),
		m_numberOfPixels(Vec2i(0,0))
{
}

ScalarFieldItem::~ScalarFieldItem()
{
	m_childs.clear();
}

bool ScalarFieldItem::signalDataChanged() 
{

	return true;
};

bool ScalarFieldItem::writeToXML(QDomDocument &document, QDomElement &node) const 
{
	if (!FieldItem::writeToXML(document, node))
		return false;

	node.setAttribute("amplMax", QString::number(m_amplMax));
	node.setAttribute("numberOfPixels.x", QString::number(m_numberOfPixels.X));
	node.setAttribute("numberOfPixels.y", QString::number(m_numberOfPixels.Y));

	return true;
}

bool ScalarFieldItem::readFromXML(const QDomElement &node)
{
	if (!FieldItem::readFromXML(node))
		return false;

	m_amplMax=node.attribute("amplMax").toDouble();
	m_numberOfPixels.X=node.attribute("numberOfPixels.x").toUInt();
	m_numberOfPixels.Y=node.attribute("numberOfPixels.y").toUInt();

	return true;
}