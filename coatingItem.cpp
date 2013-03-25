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

#include "CoatingItem.h"

using namespace macrosim;

CoatingItem::CoatingItem(CoatingType CoatType, QString name, QObject *parent) :
	AbstractItem(COATING, name, parent),
		m_coatingType(CoatType)
{
}


CoatingItem::~CoatingItem()
{
}

void CoatingItem::changeItem(const QModelIndex &topLeft, const QModelIndex & bottomRight)
{
	emit itemChanged(m_index, m_index);
}

bool CoatingItem::writeToXML(QDomDocument &document, QDomElement &root) const
{
	if (!AbstractItem::writeToXML(document, root))
		return false;
	return true;
};

bool CoatingItem::readFromXML(const QDomElement &node)
{
//	m_coatingType=stringToCoatingType(node.attribute("coatingType"));
	return true;
}