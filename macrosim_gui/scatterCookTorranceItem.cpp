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

#include "ScatterCookTorranceItem.h"

using namespace macrosim;

ScatterCookTorranceItem::ScatterCookTorranceItem(double tis, QString name, QObject *parent) :
	ScatterItem(COOKTORRANCE, name, parent),
		m_coefLambertian(0.0),
		m_fresnelParam(0.85),
		m_roughnessFactor(1.0)

{
}

ScatterCookTorranceItem::~ScatterCookTorranceItem()
{
}

bool ScatterCookTorranceItem::writeToXML(QDomDocument &document, QDomElement &root) const 
{
	QDomElement scatter = document.createElement("scatter");

	// call base class function
	if (!ScatterItem::writeToXML(document, scatter))
		return false;

	scatter.setAttribute("scatterType", "COOKTORRANCE");
	scatter.setAttribute("coefLambertian", QString::number(m_coefLambertian));
	scatter.setAttribute("fresnelParam", QString::number(m_fresnelParam));
	scatter.setAttribute("roughnessFactor", QString::number(m_roughnessFactor));
	root.appendChild(scatter);
	return true;
}

bool ScatterCookTorranceItem::readFromXML(const QDomElement &node)
{
	if (!ScatterItem::readFromXML(node))
		return false;

	m_coefLambertian=node.attribute("coefLambertian").toDouble();
	m_fresnelParam=node.attribute("fresnelParam").toDouble();
	m_roughnessFactor=node.attribute("roughnessFactor").toDouble();
	return true;
}