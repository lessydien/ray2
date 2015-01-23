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

#include "pathIntTissueFieldItem.h"
#include "materialItemLib.h"
#include "geometryItemLib.h"

using namespace macrosim;

PathIntTissueFieldItem::PathIntTissueFieldItem(QString name, QObject *parent) :
	FieldItem(name, PATHINTTISSUERAYFIELD, parent),
		m_tilt(Vec3d(0.0,0.0,0.0)),
		m_power(1.0),
		m_width(10),
		m_height(10),
		m_widthLayout(10),
		m_heightLayout(10),
		m_volumeWidth(Vec3d(10.0, 10.0, 10.0)),
		m_sourcePos(Vec3d(0.0,0.0,0.0)),
		m_meanFreePath(1.0),
		m_anisotropy(0.0)
{
	this->setRender(false); //per default we dont render the ray field
}

PathIntTissueFieldItem::~PathIntTissueFieldItem()
{
	m_childs.clear();
}

bool PathIntTissueFieldItem::signalDataChanged() 
{

	return true;
};

bool PathIntTissueFieldItem::writeToXML(QDomDocument &document, QDomElement &root) const 
{
	QDomElement node = document.createElement("field");

	if (!FieldItem::writeToXML(document, node))
		return false;

	node.setAttribute("tilt.x", QString::number(m_tilt.X));
	node.setAttribute("tilt.y", QString::number(m_tilt.Y));
	node.setAttribute("tilt.z", QString::number(m_tilt.Z));
	node.setAttribute("power", QString::number(m_power));
	node.setAttribute("width", QString::number(m_width));
	node.setAttribute("widthLayout", QString::number(m_widthLayout));
	node.setAttribute("heightLayout", QString::number(m_heightLayout));
	node.setAttribute("height", QString::number(m_height));
	node.setAttribute("volumeWidth.x", QString::number(m_volumeWidth.X));
	node.setAttribute("volumeWidth.y", QString::number(m_volumeWidth.Y));
	node.setAttribute("volumeWidth.z", QString::number(m_volumeWidth.Z));
	node.setAttribute("sourcePos.x", QString::number(m_sourcePos.X));
	node.setAttribute("sourcePos.y", QString::number(m_sourcePos.Y));
	node.setAttribute("sourcePos.z", QString::number(m_sourcePos.Z));
	node.setAttribute("meanFreePath", QString::number(m_meanFreePath));
	node.setAttribute("anisotropy", QString::number(m_anisotropy));
	root.appendChild(node);

	// write material
	if (!this->getChild()->writeToXML(document,node))
		return false;

	return true;
}

bool PathIntTissueFieldItem::readFromXML(const QDomElement &node)
{
	if (!FieldItem::readFromXML(node))
		return false;

	m_tilt.X=node.attribute("tilt.x").toDouble();
	m_tilt.Y=node.attribute("tilt.y").toDouble();
	m_tilt.Z=node.attribute("tilt.z").toDouble();
	m_power=node.attribute("power").toDouble();
	m_width=node.attribute("width").toDouble();
	m_height=node.attribute("height").toDouble();
	m_widthLayout=node.attribute("widthLayout").toDouble();
	m_heightLayout=node.attribute("heightLayout").toDouble();
	m_volumeWidth.X=node.attribute("volumeWidth.x").toDouble();
	m_volumeWidth.Y=node.attribute("volumeWidth.y").toDouble();
	m_volumeWidth.Z=node.attribute("volumeWidth.z").toDouble();
	m_sourcePos.X=node.attribute("sourcePos.x").toDouble();
	m_sourcePos.Y=node.attribute("sourcePos.y").toDouble();
	m_sourcePos.Z=node.attribute("sourcePos.z").toDouble();
	m_meanFreePath=node.attribute("meanFreePath").toDouble();
	m_anisotropy=node.attribute("anisotropy").toDouble();

	// read material
	// look for material
	QDomNodeList l_matNodeList=node.elementsByTagName("material");
	if (l_matNodeList.count()==0)
		return false;
	QDomElement l_matElementXML=l_matNodeList.at(0).toElement();
	MaterialItemLib l_materialLib;
	MaterialItem l_materialItem;
	QString l_matTypeStr=l_matElementXML.attribute("materialType");
	MaterialItem* l_pMaterialItem = l_materialLib.createMaterial(l_materialLib.stringToMaterialType(l_matTypeStr));
	if (!l_pMaterialItem->readFromXML(l_matElementXML))
		return false;

	GeometryItemLib l_geomItemLib;
	m_materialType=l_geomItemLib.stringToGeomMatType(l_matTypeStr);

	this->setChild(l_pMaterialItem);

	return true;
}

void PathIntTissueFieldItem::render(QMatrix4x4 &m, RenderOptions &options)
{

}