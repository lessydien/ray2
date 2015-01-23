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

#include "geomCylLenseSurfaceItem.h"
//#include "glut.h"

using namespace macrosim;

CylLensSurfaceItem::CylLensSurfaceItem(QString name, QObject *parent) :
	GeometryItem(name, CYLLENSESURF, parent),
	m_radius(0)
{

}

CylLensSurfaceItem::~CylLensSurfaceItem()
{
	m_childs.clear();
}

bool CylLensSurfaceItem::writeToXML(QDomDocument &document, QDomElement &root) const
{
	QDomElement node = document.createElement("geometry");

	// call base class method
	if (!GeometryItem::writeToXML(document, node))
		return false;

	node.setAttribute("geomType", "CYLLENSESURF");
	node.setAttribute("nrSurfacesSeq", "1");
	node.setAttribute("nrSurfacesNonSeq", "1");
	node.setAttribute("radius", QString::number(m_radius));

	root.appendChild(node);
	return true;
}

bool CylLensSurfaceItem::readFromXML(const QDomElement &node)
{
		// read base class from XML
	if (!GeometryItem::readFromXML(node))
		return false;
	m_radius=node.attribute("radius").toDouble();
	return true;
}

void CylLensSurfaceItem::render(QMatrix4x4 &m, RenderOptions &options)
{
	if (this->getRender())
	{

	}
}