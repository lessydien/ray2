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

#include "MaterialDOEItem.h"

using namespace macrosim;

MaterialDOEItem::MaterialDOEItem(double n1, double n2, QString name, QObject *parent) :
	MaterialItem(DOE, name, parent),
		m_n1(n1),
		m_n2(n2),
		m_stepHeight(0),
		m_dOEnr(0),
		m_glassName("USERDEFINED"),
		m_immersionName("USERDEFINED"),
		m_filenameDOE("filename.txt"),
		m_filenameBaseDOEeffs("filename")
{
}


MaterialDOEItem::~MaterialDOEItem()
{
}

bool MaterialDOEItem::writeToXML(QDomDocument &document, QDomElement &root) const
{
	QDomElement material = document.createElement("material");
	material.setAttribute("n1", QString::number(m_n1));
	material.setAttribute("n2", QString::number(m_n2));
	material.setAttribute("materialType", "DOE");
	material.setAttribute("glassName", m_glassName);
	material.setAttribute("immersionName", m_immersionName);
	
	Vec3d l_tilt;
	Vec3d l_root;

	QModelIndex l_index=this->getModelIndex();
	QModelIndex l_parentIndex=l_index.parent();
	QModelIndex test=QModelIndex();

	AbstractItem* l_pAbstractItem=reinterpret_cast<AbstractItem*>(l_parentIndex.internalPointer());
	if (l_pAbstractItem->getObjectType() == GEOMETRY)
	{
		GeometryItem* l_pGeomItem=reinterpret_cast<GeometryItem*>(l_pAbstractItem);
		l_root=l_pGeomItem->getRoot();
		l_tilt=l_pGeomItem->getTilt();
	}

	material.setAttribute("geomRoot.x", QString::number(l_root.X));
	material.setAttribute("geomRoot.y", QString::number(l_root.Y));
	material.setAttribute("geomRoot.z", QString::number(l_root.Z));
	material.setAttribute("geomTilt.x", QString::number(l_tilt.X));
	material.setAttribute("geomTilt.y", QString::number(l_tilt.Y));
	material.setAttribute("geomTilt.z", QString::number(l_tilt.Z));
	material.setAttribute("DOEnr", QString::number(m_dOEnr));
	material.setAttribute("filenameDOE", m_filenameDOE);
	material.setAttribute("filenameBaseDOEEffs", m_filenameBaseDOEeffs);
	material.setAttribute("stepHeight", QString::number(m_stepHeight));

	// write parameters inherited from base class
	if (!MaterialItem::writeToXML(document, material))
		return false;

	root.appendChild(material);

	return true;
}

bool MaterialDOEItem::readFromXML(const QDomElement &node)
{
	if (!MaterialItem::readFromXML(node))
		return false;

	m_n1=node.attribute("n1").toDouble();
	m_n2=node.attribute("n2").toDouble();
	m_stepHeight=node.attribute("stepHeight").toDouble();
	m_dOEnr=node.attribute("DOEnr").toInt();
	m_glassName=node.attribute("glassName");
	m_immersionName=node.attribute("immersionName");
	m_filenameDOE=node.attribute("filenameDOE");
	m_filenameBaseDOEeffs=node.attribute("filenameBaseDOEEffs");

	return true;
}

