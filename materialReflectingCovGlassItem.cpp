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

#include "materialReflectingCovGlassItem.h"

using namespace macrosim;

MaterialReflectingCovGlassItem::MaterialReflectingCovGlassItem(double rA, double tA, int geometryID, QString name, QObject *parent) :
	MaterialItem(REFLECTINGCOVGLASS, name, parent),
		m_rA(rA),
		m_tA(tA),
		m_geometryID(geometryID)
{
}


MaterialReflectingCovGlassItem::~MaterialReflectingCovGlassItem()
{
}

bool MaterialReflectingCovGlassItem::writeToXML(QDomDocument &document, QDomElement &root) const
{
	QDomElement material = document.createElement("material");
	material.setAttribute("rA", QString::number(m_rA));
	material.setAttribute("tA", QString::number(m_tA));
	material.setAttribute("geometryID", QString::number(m_geometryID));
	material.setAttribute("materialType", "REFLECTINGCOVGLASS");
	// write parameters inherited from base class
	if (!MaterialItem::writeToXML(document, material))
		return false;

	root.appendChild(material);

	return true;
}

bool MaterialReflectingCovGlassItem::readFromXML(const QDomElement &node)
{
	if (!MaterialItem::readFromXML(node))
		return false;
	m_tA=node.attribute("tA").toDouble();
	m_rA=node.attribute("rA").toDouble();
	m_geometryID=node.attribute("geometryID").toInt();
	return true;
}


