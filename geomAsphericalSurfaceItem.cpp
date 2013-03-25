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

#include "geomAsphericalSurfaceItem.h"

using namespace macrosim;

AsphericalSurfaceItem::AsphericalSurfaceItem(QString name, QObject *parent) :
	GeometryItem(name, ASPHERICALSURF, parent),
	m_k(0), m_c(0), m_c2(0), m_c4(0), m_c6(0), m_c8(0), m_c10(0), m_c12(0), m_c14(0), m_c16(0)
{

}

AsphericalSurfaceItem::~AsphericalSurfaceItem()
{
	m_childs.clear();
}

bool AsphericalSurfaceItem::writeToXML(QDomDocument &document, QDomElement &root) const
{
	QDomElement node = document.createElement("geometry");

	// call base class method
	if (!GeometryItem::writeToXML(document, node))
		return false;

	node.setAttribute("geomType", "ASPHERICALSURF");
	node.setAttribute("nrSurfacesSeq", "1");
	node.setAttribute("nrSurfacesNonSeq", "1");
	node.setAttribute("k", QString::number(m_k));
	node.setAttribute("c", QString::number(m_c));
	node.setAttribute("c2", QString::number(m_c2));
	node.setAttribute("c4", QString::number(m_c4));
	node.setAttribute("c6", QString::number(m_c6));
	node.setAttribute("c8", QString::number(m_c8));
	node.setAttribute("c10", QString::number(m_c10));
	node.setAttribute("c12", QString::number(m_c12));
	node.setAttribute("c14", QString::number(m_c14));
	node.setAttribute("c16", QString::number(m_c16));
	root.appendChild(node);
	return true;
}

bool AsphericalSurfaceItem::readFromXML(const QDomElement &node)
{
	// read base class from XML
	if (!GeometryItem::readFromXML(node))
		return false;
	m_k=node.attribute("k").toDouble();
	m_c=node.attribute("c").toDouble();
	m_c2=node.attribute("c2").toDouble();
	m_c4=node.attribute("c4").toDouble();
	m_c6=node.attribute("c6").toDouble();
	m_c8=node.attribute("c8").toDouble();
	m_c10=node.attribute("c10").toDouble();
	m_c12=node.attribute("c12").toDouble();
	m_c14=node.attribute("c14").toDouble();
	m_c16=node.attribute("c16").toDouble();
	return true;
}

void AsphericalSurfaceItem::render(QMatrix4x4 &m, RenderOptions &options)
{
	if (this->getRender())
	{

	}

}