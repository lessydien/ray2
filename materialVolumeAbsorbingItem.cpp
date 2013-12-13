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

#include "materialVolumeAbsorbingItem.h"

using namespace macrosim;

MaterialVolumeAbsorbingItem::MaterialVolumeAbsorbingItem(QString name, QObject *parent) :
	MaterialItem(VOLUMEABSORBING, name, parent),
		m_absorbCoeff(0)
{
}

MaterialVolumeAbsorbingItem::~MaterialVolumeAbsorbingItem(void) 
{
	m_childs.clear();
}

bool MaterialVolumeAbsorbingItem::writeToXML(QDomDocument &document, QDomElement &root) const
{
	QDomElement material = document.createElement("material");
	material.setAttribute("materialType", "VOLUMEABSORBING");
	// write parameters inherited from base class
	if (!MaterialItem::writeToXML(document, material))
		return false;

	material.setAttribute("absorbCoeff", QString::number(m_absorbCoeff));

	root.appendChild(material);

	return true;
}

bool MaterialVolumeAbsorbingItem::readFromXML(const QDomElement &node)
{
	if (!MaterialItem::readFromXML(node))
		return false;

	m_absorbCoeff=node.attribute("absorbCoeff").toDouble();

	return true;
}

