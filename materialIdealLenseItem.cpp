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

#include "materialIdealLenseItem.h"

using namespace macrosim;

MaterialIdealLenseItem::MaterialIdealLenseItem(double f0, double lambda0, double dispConst, QString name, QObject *parent) :
	MaterialItem(MATIDEALLENSE, name, parent),
		m_f0(f0),
		m_lambda0(lambda0),
		m_dispConst(dispConst)
{
}


MaterialIdealLenseItem::~MaterialIdealLenseItem()
{
}

bool MaterialIdealLenseItem::writeToXML(QDomDocument &document, QDomElement &root) const
{
	QDomElement material = document.createElement("material");
	material.setAttribute("f0", QString::number(m_f0));
	material.setAttribute("lambda0", QString::number(m_lambda0));
	material.setAttribute("dispConst", QString::number(m_dispConst));
	material.setAttribute("materialType", "MATIDEALLENSE");
	// write parameters inherited from base class
	if (!MaterialItem::writeToXML(document, material))
		return false;

	root.appendChild(material);

	return true;
}

bool MaterialIdealLenseItem::readFromXML(const QDomElement &node)
{
	if (!MaterialItem::readFromXML(node))
		return false;

	m_dispConst=node.attribute("dispConst").toDouble();
	m_f0=node.attribute("f0").toDouble();
	m_lambda0=node.attribute("lambda0").toDouble();

	return true;
}
