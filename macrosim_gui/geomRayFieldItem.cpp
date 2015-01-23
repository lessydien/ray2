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

#include "geomRayFieldItem.h"
#include <vtkVersion.h>
#include <vtkPlaneSource.h> 
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkSphereSource.h>

using namespace macrosim;

GeomRayFieldItem::GeomRayFieldItem(QString name, QObject *parent) :
	RayFieldItem(name, GEOMRAYFIELD, parent)
{
	this->setRender(false); //per default we dont render the ray field
}

GeomRayFieldItem::~GeomRayFieldItem()
{
	m_childs.clear();
}

bool GeomRayFieldItem::signalDataChanged() 
{

	return true;
};

bool GeomRayFieldItem::writeToXML(QDomDocument &document, QDomElement &root) const 
{
	QDomElement node = document.createElement("field");

	if (!RayFieldItem::writeToXML(document, node))
		return false;

	root.appendChild(node);
	return true;
}

bool GeomRayFieldItem::readFromXML(const QDomElement &node)
{
	if (!RayFieldItem::readFromXML(node))
		return false;


	return true;
}

void GeomRayFieldItem::render(QMatrix4x4 &m, RenderOptions &options)
{

}

void GeomRayFieldItem::renderVtk(vtkSmartPointer<vtkRenderer> renderer)
{

}