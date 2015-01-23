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

#include "ScatterPhongItem.h"

using namespace macrosim;

ScatterPhongItem::ScatterPhongItem(double tis, QString name, QObject *parent) :
//ScatterPhongItem::ScatterPhongItem(double tis, QString name,double phongParam, double coefPhong) :
	ScatterItem(PHONG, name, parent),
		m_coefLambertian(0.0),
		m_phongParam(0.0),
		m_coefPhong(1.0)

{
}


ScatterPhongItem::~ScatterPhongItem()
{
}

bool ScatterPhongItem::writeToXML(QDomDocument &document, QDomElement &root) const 
{
	QDomElement scatter = document.createElement("scatter");

	// call base class function
	if (!ScatterItem::writeToXML(document, scatter))
		return false;

	scatter.setAttribute("scatterType", "PHONG");
	scatter.setAttribute("coefLambertian", QString::number(m_coefLambertian));
	scatter.setAttribute("phongParam", QString::number(m_phongParam));
	scatter.setAttribute("coefPhong", QString::number(m_coefPhong));
	root.appendChild(scatter);
	return true;
}

bool ScatterPhongItem::readFromXML(const QDomElement &node)
{
	if (!ScatterItem::readFromXML(node))
		return false;

	m_coefLambertian=node.attribute("coefLambertian").toDouble();
	m_phongParam=node.attribute("phongParam").toDouble();
	m_coefPhong=node.attribute("coefPhong").toDouble();
	return true;
}