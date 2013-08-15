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

#include "intensityFieldItem.h"

using namespace macrosim;

IntensityFieldItem::IntensityFieldItem(QString name, QObject *parent) :
	FieldItem(name, INTENSITYFIELD, parent)
{
}

IntensityFieldItem::~IntensityFieldItem()
{
	m_childs.clear();
}

bool IntensityFieldItem::signalDataChanged() 
{

	return true;
};

bool IntensityFieldItem::writeToXML(QDomDocument &document, QDomElement &root) const 
{
	QDomElement node = document.createElement("field");

	if (!FieldItem::writeToXML(document, node))
		return false;

	node.setAttribute("fieldDataFileName", m_fieldDataFileName);

	root.appendChild(node);
	return true;
}

bool IntensityFieldItem::readFromXML(const QDomElement &node)
{
	if (!FieldItem::readFromXML(node))
		return false;

	m_fieldDataFileName=node.attribute("fieldDataFileName");

	return true;
}