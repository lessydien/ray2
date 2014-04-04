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

#include "MaterialRenderLightItem.h"

using namespace macrosim;

MaterialRenderLightItem::MaterialRenderLightItem(double power, QString name, QObject *parent) :
	MaterialItem(REFRACTING, name, parent),
		m_power(power)
{
}


MaterialRenderLightItem::~MaterialRenderLightItem()
{
}

bool MaterialRenderLightItem::writeToXML(QDomDocument &document, QDomElement &root) const
{
	QDomElement material = document.createElement("material");
	material.setAttribute("materialType", "RENDERLIGHT");
	material.setAttribute("power", QString::number(m_power));
    material.setAttribute("pupilRoot.x", QString::number(m_pupilRoot.X));
    material.setAttribute("pupilRoot.y", QString::number(m_pupilRoot.Y));
    material.setAttribute("pupilRoot.z", QString::number(m_pupilRoot.Z));

    material.setAttribute("pupilTilt.x", QString::number(m_pupilTilt.X));
    material.setAttribute("pupilTilt.y", QString::number(m_pupilTilt.Y));
    material.setAttribute("pupilTilt.z", QString::number(m_pupilTilt.Z));

    material.setAttribute("pupilAptRad.x", QString::number(m_pupilAptRad.X));
    material.setAttribute("pupilAptRad.y", QString::number(m_pupilAptRad.Y));

	// write parameters inherited from base class
	if (!MaterialItem::writeToXML(document, material))
		return false;

	root.appendChild(material);

	return true;
}

bool MaterialRenderLightItem::readFromXML(const QDomElement &node)
{
	if (!MaterialItem::readFromXML(node))
		return false;

    m_power=node.attribute("glassName").toDouble();

    m_pupilRoot.X=node.attribute("pupilRoot.x").toDouble();
    m_pupilRoot.Y=node.attribute("pupilRoot.y").toDouble();
    m_pupilRoot.Z=node.attribute("pupilRoot.z").toDouble();

    m_pupilTilt.X=node.attribute("pupilTilt.x").toDouble();
    m_pupilTilt.Y=node.attribute("pupilTilt.y").toDouble();
    m_pupilTilt.Z=node.attribute("pupilTilt.z").toDouble();

    m_pupilAptRad.X=node.attribute("pupilAptRad.x").toDouble();
    m_pupilAptRad.Y=node.attribute("pupilAptRad.y").toDouble();

	return true;
}

