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

#include "scalarGaussianFieldItem.h"

using namespace macrosim;

ScalarGaussianFieldItem::ScalarGaussianFieldItem(QString name, QObject *parent) :
ScalarFieldItem(name, FieldItem::SCALARGAUSSIANWAVE, parent)
{
}

ScalarGaussianFieldItem::~ScalarGaussianFieldItem()
{
	m_childs.clear();
}

bool ScalarGaussianFieldItem::signalDataChanged() 
{

	return true;
};

bool ScalarGaussianFieldItem::writeToXML(QDomDocument &document, QDomElement &root) const 
{
	QDomElement node = document.createElement("field");

	if (!ScalarFieldItem::writeToXML(document, node))
		return false;

	node.setAttribute("focusWidth.x", m_focusWidth.X);
	node.setAttribute("focusWidth.y", m_focusWidth.Y);
	node.setAttribute("distToFocus.x", m_distToFocus.X);
	node.setAttribute("distToFocus.y", m_distToFocus.Y);

	root.appendChild(node);
	return true;
}

bool ScalarGaussianFieldItem::readFromXML(const QDomElement &node)
{
	if (!ScalarFieldItem::readFromXML(node))
		return false;

	m_focusWidth.X=node.attribute("focusWidth.x").toDouble();
	m_focusWidth.Y=node.attribute("focusWidth.y").toDouble();
	m_distToFocus.X=node.attribute("distToFocus.x").toDouble();
	m_distToFocus.Y=node.attribute("distToFocus.y").toDouble();

	return true;
}