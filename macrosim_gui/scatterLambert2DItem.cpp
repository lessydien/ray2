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

#include "ScatterLambert2DItem.h"

using namespace macrosim;

ScatterLambert2DItem::ScatterLambert2DItem(double tis, QString name, QObject *parent) :
	ScatterItem(LAMBERT2D, name, parent),
		m_Tis(tis)
{
}


ScatterLambert2DItem::~ScatterLambert2DItem()
{
}

bool ScatterLambert2DItem::writeToXML(QDomDocument &document, QDomElement &root) const 
{
	QDomElement scatter = document.createElement("scatter");

	// call base class function
	if (!ScatterItem::writeToXML(document, scatter))
		return false;

	scatter.setAttribute("scatterType", "LAMBERT2D");
	scatter.setAttribute("Tis", QString::number(m_Tis));

	root.appendChild(scatter);
	return true;
}

bool ScatterLambert2DItem::readFromXML(const QDomElement &node)
{
	if (!ScatterItem::readFromXML(node))
		return false;

	m_Tis=node.attribute("Tis").toDouble();

	return true;
}