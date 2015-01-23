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

#include "miscItemLib.h"

using namespace macrosim;

MiscItem* MiscItemLib::createMiscItem(MiscItem::MiscType type)
{
	MiscItem* l_pItem=NULL;
		// if material changed, we need to create an instance of the new material and append it
		switch (type)
		{
		case MiscItem::GEOMETRYGROUP:
			l_pItem=new GeomGroupItem();
			break;
		//case GeometryItem::DETECTOR:
		//	l_pItem=new MaterialPathTraceSourceItem();
		//	break;
		default:
			break;
		}
		return l_pItem;
}

QString MiscItemLib::miscTypeToString(const MiscItem::MiscType type) const
{
	QString str;
	switch (type)
	{
	case MiscItem::GEOMETRYGROUP:
		str="GEOMETRYGROUP";
		break;
	default:
		str="UNKNOWN";
		break;
	}
	return str;
};

MiscItem::MiscType MiscItemLib::stringToMiscType(const QString str) const
{
	if (str.isNull())
		return MiscItem::UNDEFINED;
	if (!str.compare("GEOMETRYGROUP") )
		return MiscItem::GEOMETRYGROUP;

	return MiscItem::UNDEFINED;
};

QList<AbstractItem*> MiscItemLib::fillLibrary() const
{
	QList<AbstractItem*> l_list;
	l_list.append(new GeomGroupItem());
	return l_list;
}
