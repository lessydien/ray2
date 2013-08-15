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

#include "CoatingNumCoeffsItem.h"

using namespace macrosim;

CoatingNumCoeffsItem::CoatingNumCoeffsItem(double rA, double tA, QString name, QObject *parent) :
	CoatingItem(NUMCOEFFS, name, parent),
		m_rA(rA),
		m_tA(tA)
{
}


CoatingNumCoeffsItem::~CoatingNumCoeffsItem()
{
}

bool CoatingNumCoeffsItem::writeToXML(QDomDocument &document, QDomElement &root) const
{
	QDomElement coating = document.createElement("coating");
	coating.setAttribute("coatingType", "NUMCOEFFS");
	coating.setAttribute("tA", QString::number(m_tA));
	coating.setAttribute("rA", QString::number(m_rA));
	if (!CoatingItem::writeToXML(document, coating))
		return false;
	root.appendChild(coating);
	return true;
}

bool CoatingNumCoeffsItem::readFromXML(const QDomElement &node)
{
	if (!CoatingItem::readFromXML(node))
		return false;
	m_tA=node.attribute("tA").toDouble();
	m_rA=node.attribute("rA").toDouble();
	return true;
}

