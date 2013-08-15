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

#include "CoatingNoItem.h"

using namespace macrosim;

CoatingNoItem::CoatingNoItem(QString name, QObject *parent) :
	CoatingItem(NOCOATING, name, parent)
{
}


CoatingNoItem::~CoatingNoItem()
{
}

bool CoatingNoItem::writeToXML(QDomDocument &document, QDomElement &root) const
{
	QDomElement coating = document.createElement("coating");
	coating.setAttribute("coatingType", "NOCOATING");
	if (!CoatingItem::writeToXML(document, coating))
		return false;
	root.appendChild(coating);
	return true;
}

bool CoatingNoItem::readFromXML(const QDomElement &node)
{
	return true;
}
