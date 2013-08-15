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

#include "miscItem.h"
#include "miscItemLib.h"

using namespace macrosim;

MiscItem::MiscItem(QString name, MiscType type, QObject *parent) :
AbstractItem(AbstractItem::MISCITEM, name, parent),
	m_miscType(type)
{
}

MiscItem::~MiscItem()
{
	m_childs.clear();
}

bool MiscItem::writeToXML(QDomDocument &document, QDomElement &node) const 
{

	return true;
}

bool MiscItem::readFromXML(const QDomElement &node)
{
	
	return true;
}