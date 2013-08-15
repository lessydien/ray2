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

#include "MaterialVolumeScatterItem.h"

using namespace macrosim;

MaterialVolumeScatterItem::MaterialVolumeScatterItem(double n1, double n2, QString name, QObject *parent) :
	MaterialItem(VOLUMESCATTER, name, parent),
		m_n1(n1),
		m_n2(n2)
{
}


MaterialVolumeScatterItem::~MaterialVolumeScatterItem()
{
}

bool MaterialVolumeScatterItem::writeToXML(QDomDocument &document, QDomElement &root) const
{
	QDomElement material = document.createElement("material");
	material.setAttribute("n1", QString::number(m_n1));
	material.setAttribute("n2", QString::number(m_n2));
	material.setAttribute("materialType", "VOLUMESCATTER");
	material.setAttribute("meanFreePath", QString::number(m_meanFreePath));
	material.setAttribute("maxNrBounces", QString::number(m_maxNrBounces));
	material.setAttribute("absorptionCoeff", QString::number(m_absorptionCoeff));
	material.setAttribute("anisotropyFac", QString::number(m_anisotropyFac));

	// write parameters inherited from base class
	if (!MaterialItem::writeToXML(document, material))
		return false;

	root.appendChild(material);

	return true;
}

bool MaterialVolumeScatterItem::readFromXML(const QDomElement &node)
{
	if (!MaterialItem::readFromXML(node))
		return false;

	m_meanFreePath=node.attribute("meanFreePath").toDouble();
	m_absorptionCoeff=node.attribute("absorptionCoeff").toDouble();
	m_anisotropyFac=node.attribute("anistropyFac").toDouble();
	m_maxNrBounces=node.attribute("maxNrBounces").toDouble();
	m_n1=node.attribute("n1").toDouble();
	m_n2=node.attribute("n2").toDouble();

	return true;
}

