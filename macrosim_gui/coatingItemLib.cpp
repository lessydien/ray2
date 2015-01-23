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

#include "CoatingItemLib.h"

using namespace macrosim;

CoatingItem* CoatingItemLib::createCoating(CoatingItem::CoatingType type)
{
	CoatingItem* l_pItem;
		// if material changed, we need to create an instance of the new material and append it
		switch (type)
		{
		case CoatingItem::NUMCOEFFS:
			l_pItem=new CoatingNumCoeffsItem();
			break;
		default:
			l_pItem=new CoatingNoItem();
			break;
		}
		return l_pItem;
}

QString CoatingItemLib::coatingTypeToString(const CoatingItem::CoatingType type) const
{
	QString str;
	switch (type)
	{
	case CoatingItem::NUMCOEFFS:
		str="NUMCOEFFS";
		break;
	case CoatingItem::NOCOATING:
		str="NOCOATING";
		break;
	default:
		str="NOCOATING";
		break;
	}
	return str;
}

CoatingItem::CoatingType CoatingItemLib::stringToCoatingType(const QString str) const
{
	if (str.isNull())
		return CoatingItem::NOCOATING;
	if (!str.compare("NOCOATING"))
		return CoatingItem::NOCOATING;
	if (!str.compare("NUMCOEFFS"))
		return CoatingItem::NUMCOEFFS;
	return CoatingItem::NOCOATING;
}

CoatingItem::CoatingType CoatingItemLib::matCoatTypeToCoatType(const MaterialItem::Mat_CoatingType type) const
{
	CoatingItem::CoatingType typeOut;
	switch (type)
	{
	case MaterialItem::NUMCOEFFS:
		typeOut=CoatingItem::NUMCOEFFS;
		break;
	case MaterialItem::NOCOATING:
		typeOut=CoatingItem::NOCOATING;
		break;
	default:
		typeOut=CoatingItem::NOCOATING;
		break;
	}
	return typeOut;
}

MaterialItem::Mat_CoatingType CoatingItemLib::coatTypeToMatCoatType(const CoatingItem::CoatingType type) const
{
	MaterialItem::Mat_CoatingType typeOut;
	switch (type)
	{
	case CoatingItem::NUMCOEFFS:
		typeOut=MaterialItem::NUMCOEFFS;
		break;
	case CoatingItem::NOCOATING:
		typeOut=MaterialItem::NOCOATING;
		break;
	default:
		typeOut=MaterialItem::NOCOATING;
		break;
	}
	return typeOut;
}
