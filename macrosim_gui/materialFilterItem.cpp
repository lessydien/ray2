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

#include "materialFilterItem.h"

using namespace macrosim;

MaterialFilterItem::MaterialFilterItem(double lambdaMax, double lambdaMin, QString name, QObject *parent) :
	MaterialItem(FILTER, name, parent),
		m_lambdaMax(lambdaMax),
		m_lambdaMin(lambdaMin)
{
}


MaterialFilterItem::~MaterialFilterItem()
{
}

bool MaterialFilterItem::writeToXML(QDomDocument &document, QDomElement &root) const
{
	QDomElement material = document.createElement("material");
	material.setAttribute("lambdaMax", QString::number(m_lambdaMax));
	material.setAttribute("lambdaMin", QString::number(m_lambdaMin));
	material.setAttribute("materialType", "FILTER");
	// write parameters inherited from base class
	if (!MaterialItem::writeToXML(document, material))
		return false;

	root.appendChild(material);

	return true;
}

bool MaterialFilterItem::readFromXML(const QDomElement &node)
{
	if (!MaterialItem::readFromXML(node))
		return false;

	m_lambdaMax=node.attribute("lambdaMax").toDouble();
	m_lambdaMin=node.attribute("lambdaMin").toDouble();

	return true;
}

