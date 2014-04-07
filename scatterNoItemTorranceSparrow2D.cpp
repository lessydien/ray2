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

#include "scatterTorranceSparrow2DItem.h"

using namespace macrosim;

ScatterTorrSparr2DItem::ScatterTorrSparr2DItem(QString name, QObject *parent) :
	ScatterItem(TORRSPARR2D, name, parent)
{
}


ScatterTorrSparr2DItem::~ScatterTorrSparr2DItem()
{
}

bool ScatterTorrSparr2DItem::writeToXML(QDomDocument &document, QDomElement &root) const
{
	QDomElement scatter = document.createElement("scatter");

	// call base class function
	if (!ScatterItem::writeToXML(document, scatter))
		return false;

	scatter.setAttribute("scatterType", "TORRSPARR2D");
	scatter.setAttribute("kDl", QString::number(m_kDl));
	scatter.setAttribute("kSl", QString::number(m_kSl));
	scatter.setAttribute("kSp", QString::number(m_kSp));
	scatter.setAttribute("sigmaSp", QString::number(m_sigmaSp));
	scatter.setAttribute("sigmaSl", QString::number(m_sigmaSl));

	root.appendChild(scatter);
	return true;
}

bool ScatterTorrSparr2DItem::readFromXML(const QDomElement &node)
{
	if (!ScatterItem::readFromXML(node))
		return false;

	m_kDl=node.attribute("kDl").toDouble();
	m_kSl=node.attribute("kSl").toDouble();
    m_kSp=node.attribute("kSp").toDouble();
    m_sigmaSl=node.attribute("sigmaSl").toDouble();
    m_sigmaSp=node.attribute("sigmaSp").toDouble();

	return true;
}