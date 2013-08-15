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

#include "geomGroupItem.h"
#include <iostream>
using namespace std;

using namespace macrosim;

GeomGroupItem::GeomGroupItem(QString name, QObject *parent) :
	MiscItem(name, MiscItem::GEOMETRYGROUP, parent),
		m_accelType(NOACCEL)
{

};

GeomGroupItem::~GeomGroupItem()
{
	m_childs.clear();
}

QModelIndex GeomGroupItem::hasActor(void *actor) const
{
	for (int i=0; i<this->m_childs.size(); i++)
	{
		QModelIndex l_modelIndex=this->m_childs.at(i)->hasActor(actor);
		if (QModelIndex() != l_modelIndex)
		{
			return l_modelIndex;
		}
	}
	return QModelIndex();
}

QString GeomGroupItem::accelerationTypeToString(const AccelerationType in) const
{
	QString str;
	switch (in)
	{
	case NOACCEL:
		str= "NOACCEL";
		break;
	case SBVH:
		str="SBVH";
		break;
	case BVH:
		str="BVH";
		break;
	case MEDIANBVH:
		str="MEDIANBVH";
		break;
	case LBVH:
		str="LBVH";
		break;
	case TRIANGLEKDTREE:
		str="TRIANGLEKDTREE";
		break;
	default:
		str="NOACCEL";
		break;
	}
	return str;
};

GeomGroupItem::AccelerationType GeomGroupItem::stringToAccelerationType(const QString in) const
{
	if (in.isNull())
		return GeomGroupItem::NOACCEL;
	if (!in.compare("NOACCEL") )
		return GeomGroupItem::NOACCEL;
	if (!in.compare("SBVH") )
		return GeomGroupItem::SBVH;
	if (!in.compare("BVH") )
		return GeomGroupItem::BVH;
	if (!in.compare("MEDIANBVH") )
		return GeomGroupItem::MEDIANBVH;
	if (!in.compare("LBVH") )
		return GeomGroupItem::LBVH;
	if (!in.compare("TRIANGLEKDTREE") )
		return GeomGroupItem::TRIANGLEKDTREE;

	return GeomGroupItem::NOACCEL;
};

bool GeomGroupItem::writeToXML(QDomDocument &document, QDomElement &root) const 
{
	QDomElement node = document.createElement("geometryGroup");
	node.setAttribute("accelType", accelerationTypeToString(m_accelType));

	if (!AbstractItem::writeToXML(document, root))
		return false;

	for (unsigned int i=0; i<this->getNumberOfChilds(); i++)
	{
		this->getChild(i)->writeToXML(document, node);
	}

	root.appendChild(node);
	return true;
};

bool GeomGroupItem::readFromXML(const QDomElement &node) 
{	
	if (!AbstractItem::readFromXML(node) )
		return false;
	return true;
};

void GeomGroupItem::render(QMatrix4x4 &m, RenderOptions &options)
{
	if (this->getRender())
	{
		for (unsigned int i=0; i<m_childs.size(); i++)
		{
			m_childs.at(i)->render(m, options);
		}
	}
};

void GeomGroupItem::renderVtk(vtkSmartPointer<vtkRenderer> renderer)
{
	if (this->getRender())
	{
		for (unsigned int i=0; i<m_childs.size(); i++)
		{
			m_childs.at(i)->renderVtk(renderer);
		}
	}
};


void GeomGroupItem::setRenderOptions(RenderOptions options)
{
	if (this->getRender())
	{
		for (unsigned int i=0; i<m_childs.size(); i++)
		{
			m_childs.at(i)->setRenderOptions(options);
		}
	}
};

void GeomGroupItem::removeFromView(vtkSmartPointer<vtkRenderer> renderer)
{
	for (unsigned int i=0; i<m_childs.size(); i++)
	{
		m_childs.at(i)->removeFromView(renderer);
	}
};