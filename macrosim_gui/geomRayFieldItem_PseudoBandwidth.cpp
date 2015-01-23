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

#include "geomRayFieldItem_PseudoBandwidth.h"
#include <vtkVersion.h>
#include <vtkPlaneSource.h> 
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkSphereSource.h>

using namespace macrosim;

GeomRayFieldItem_PseudoBandwidth::GeomRayFieldItem_PseudoBandwidth(QString name, QObject *parent) :
	RayFieldItem(name, GEOMRAYFIELD_PSEUDOBANDWIDTH, parent)
{
}

GeomRayFieldItem_PseudoBandwidth::~GeomRayFieldItem_PseudoBandwidth()
{
	m_childs.clear();
}


bool GeomRayFieldItem_PseudoBandwidth::writeToXML(QDomDocument &document, QDomElement &root) const 
{
	QDomElement node = document.createElement("field");

	if (!RayFieldItem::writeToXML(document, node))
		return false;

	node.setAttribute("pseudoBandwidth", QString::number(m_pseudoBandwidth));
	node.setAttribute("nrPseudoLambdas", QString::number(m_nrPseudoLambdas));

	root.appendChild(node);
	return true;
}

bool GeomRayFieldItem_PseudoBandwidth::readFromXML(const QDomElement &node)
{
	if (!RayFieldItem::readFromXML(node))
		return false;

	m_pseudoBandwidth=node.attribute("pseudoBandwidth").toDouble();
	m_nrPseudoLambdas=node.attribute("nrPseudoLambdas").toInt();

	return true;
}

void GeomRayFieldItem_PseudoBandwidth::render(QMatrix4x4 &m, RenderOptions &options)
{

}

void GeomRayFieldItem_PseudoBandwidth::renderVtk(vtkSmartPointer<vtkRenderer> renderer)
{

}