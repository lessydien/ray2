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

#include "MaterialDiffractingItem.h"

using namespace macrosim;

MaterialDiffractingItem::MaterialDiffractingItem(double n1, double n2, QString name, QObject *parent) :
	MaterialItem(DIFFRACTING, name, parent),
		m_n1(n1),
		m_n2(n2)
{
}


MaterialDiffractingItem::~MaterialDiffractingItem()
{
}

bool MaterialDiffractingItem::writeToXML(QDomDocument &document, QDomElement &root) const
{
	QDomElement material = document.createElement("material");
	material.setAttribute("n1", QString::number(m_n1));
	material.setAttribute("n2", QString::number(m_n2));
	material.setAttribute("materialType", "DIFFRACTING");
	// write parameters inherited from base class
	if (!MaterialItem::writeToXML(document, material))
		return false;

	root.appendChild(material);

	return true;
}

bool MaterialDiffractingItem::readFromXML(const QDomElement &node)
{
	if (!MaterialItem::readFromXML(node))
		return false;

	m_n1=node.attribute("n1").toDouble();
	m_n2=node.attribute("n2").toDouble();

	return true;
}


