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

#include "detectorFieldItem.h"
#include "detectorItemLib.h"

using namespace macrosim;

DetectorFieldItem::DetectorFieldItem(QString name, QObject *parent) :
	DetectorItem(name, FIELD, parent)
{
}

DetectorFieldItem::~DetectorFieldItem()
{
	m_childs.clear();
}


bool DetectorFieldItem::writeToXML(QDomDocument &document, QDomElement &root) const 
{
	QDomElement node = document.createElement("detector");

	if (!DetectorItem::writeToXML(document, node))
		return false;

	node.setAttribute("detPixel.x", QString::number(m_detPixel.X));
	node.setAttribute("detPixel.y", QString::number(m_detPixel.Y));

	root.appendChild(node);
	return true;
}

bool DetectorFieldItem::readFromXML(const QDomElement &node)
{
	// read base class
	if (!DetectorItem::readFromXML(node))
		return false;

	m_detPixel.X=node.attribute("detPixel.x").toDouble();
	m_detPixel.Y=node.attribute("detPixel.y").toDouble();

	return true;
}