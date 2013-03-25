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

#include "abstractItem.h"
#include "geometryItemLib.h"
#include "materialItemLib.h"
#include <iostream>
using namespace std;

using namespace macrosim;

bool AbstractItem::writeToXML(QDomDocument &document, QDomElement &root) const 
{ 
	root.setAttribute("objectType", objectTypeToString(m_objectType)); 
	root.setAttribute("name", m_name);
	return true;
};

bool AbstractItem::readFromXML(const QDomElement &node) 
{	
	QString str = node.attribute("objectType" );
	m_objectType=stringToObjectType(str);
	m_root.X=(node.attribute("root.x")).toDouble();
	m_root.Y=(node.attribute("root.y")).toDouble();
	m_root.Z=(node.attribute("root.z")).toDouble();
	m_name=node.attribute("name");
	return true;
};

void AbstractItem::setMaterialType(const Abstract_MaterialType type) 
{
	// if materialtype changed
	if (m_materialType != type)
	{
		m_materialType = type; 
		MaterialItemLib l_matLib;
		MaterialItem::MaterialType l_newMatType=l_matLib.abstractMatTypeToMatType(type);
		MaterialItem* l_pNewMat=l_matLib.createMaterial(l_newMatType);
		// remove old material
		this->removeChild(0);
		this->setChild(l_pNewMat);
		emit itemChanged(m_index, m_index);
	}
};

void AbstractItem::render(QMatrix4x4 &m, RenderOptions &options)
{

};
