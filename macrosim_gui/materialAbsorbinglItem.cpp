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

#include "materialAbsorbingItem.h"

using namespace macrosim;

MaterialAbsorbingItem::MaterialAbsorbingItem(QString name, QObject *parent) :
	MaterialItem(ABSORBING, name, parent)
{
}

MaterialAbsorbingItem::~MaterialAbsorbingItem(void) 
{
	m_childs.clear();
}

bool MaterialAbsorbingItem::writeToXML(QDomDocument &document, QDomElement &root) const
{
	QDomElement material = document.createElement("material");
	material.setAttribute("materialType", "ABSORBING");
	// write parameters inherited from base class
	if (!MaterialItem::writeToXML(document, material))
		return false;

//	QModelIndex l_index=this->getModelIndex();
//	QModelIndex l_parentIndex=l_index.parent();
//	AbstractItem* l_pItem=reinterpret_cast<AbstractItem*>(l_parentIndex.internalPointer());

	root.appendChild(material);

	return true;
}

bool MaterialAbsorbingItem::readFromXML(const QDomElement &node)
{
	if (!MaterialItem::readFromXML(node))
		return false;

	return true;
}

