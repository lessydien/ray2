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

#include "materialPathTraceSourceItem.h"

using namespace macrosim;

MaterialPathTraceSourceItem::MaterialPathTraceSourceItem(Vec2d acceptanceAngleMax, Vec2d acceptanceAngleMin, double flux, QString name, QObject *parent) :
	MaterialItem(PATHTRACESOURCE, name, parent),
		m_acceptanceAngleMax(acceptanceAngleMax),
		m_acceptanceAngleMin(acceptanceAngleMin),
		m_flux(flux)
{
}


MaterialPathTraceSourceItem::~MaterialPathTraceSourceItem()
{
}

bool MaterialPathTraceSourceItem::writeToXML(QDomDocument &document, QDomElement &root) const
{
	QDomElement material = document.createElement("material");
	material.setAttribute("acceptanceAngleMax.x", QString::number(m_acceptanceAngleMax.X));
	material.setAttribute("acceptanceAngleMax.y", QString::number(m_acceptanceAngleMax.Y));
	material.setAttribute("acceptanceAngleMin.x", QString::number(m_acceptanceAngleMin.X));
	material.setAttribute("acceptanceAngleMin.y", QString::number(m_acceptanceAngleMin.Y));
	material.setAttribute("double", QString::number(m_flux));
	material.setAttribute("materialType", "PATHTRACESOURCE");
	// write parameters inherited from base class
	if (!MaterialItem::writeToXML(document, material))
		return false;

	root.appendChild(material);

	return true;
}

bool MaterialPathTraceSourceItem::readFromXML(const QDomElement &node)
{
	if (!MaterialItem::readFromXML(node))
		return false;

	m_acceptanceAngleMax.X=node.attribute("acceptanceAngleMax.x").toDouble();
	m_acceptanceAngleMax.Y=node.attribute("acceptanceAngleMax.y").toDouble();
	m_acceptanceAngleMin.X=node.attribute("acceptanceAngleMin.x").toDouble();
	m_acceptanceAngleMin.Y=node.attribute("acceptanceAngleMin.y").toDouble();
	m_flux=node.attribute("flux").toDouble();

	return true;
}

