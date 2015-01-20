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

#include "ScatterItemLib.h"

using namespace macrosim;

ScatterItem* ScatterItemLib::createScatter(ScatterItem::ScatterType type)
{
	ScatterItem* l_pItem;
		// if material changed, we need to create an instance of the new material and append it
		switch (type)
		{
		case ScatterItem::LAMBERT2D:
			l_pItem=new ScatterLambert2DItem();
			break;
		case ScatterItem::TORRSPARR1D:
			l_pItem=new ScatterTorrSparr1DItem();
			break;
		case ScatterItem::TORRSPARR2D:
			l_pItem=new ScatterTorrSparr2DItem();
			break;
		case ScatterItem::TORRSPARR2DPATHTRACE:
			l_pItem=new ScatterTorrSparr2DPathTraceItem();
			break;
		case ScatterItem::DISPDOUBLECAUCHY1D:
			l_pItem=new ScatterDispersiveDoubleCauchy1DItem();
			break;
		case ScatterItem::DOUBLECAUCHY1D:
			l_pItem=new ScatterDoubleCauchy1DItem();
			break;
		case ScatterItem::PHONG:
			l_pItem=new ScatterPhongItem();
			break;
        case ScatterItem::COOKTORRANCE:
			l_pItem=new ScatterCookTorranceItem();
			break;
		default:
			l_pItem=new ScatterNoItem();
			break;
		}
		return l_pItem;
}

QString ScatterItemLib::scatterTypeToString(const ScatterItem::ScatterType type) const
{
	QString str;
	switch (type)
	{
	case ScatterItem::LAMBERT2D:
		str= "LAMBERT2D";
		break;
	case ScatterItem::TORRSPARR1D:
		str= "TORRSPARR1D";
		break;
	case ScatterItem::TORRSPARR2D:
		str= "TORRSPARR2D";
		break;
	case ScatterItem::TORRSPARR2DPATHTRACE:
		str= "TORRSPARR2DPATHTRACE";
		break;
	case ScatterItem::DISPDOUBLECAUCHY1D:
		str= "DISPDOUBLECAUCHY1D";
		break;
	case ScatterItem::DOUBLECAUCHY1D:
		str= "DOUBLECAUCHY1D";
		break;
	case ScatterItem::PHONG:
		str= "PHONG";
		break;
    case ScatterItem::COOKTORRANCE:
		str= "COOKTORRANCE";
		break;
	case ScatterItem::NOSCATTER:
		str= "NOSCATTER";
		break;
	default:
		str="NOSCATTER";
		break;
	}

	return str;
}

ScatterItem::ScatterType ScatterItemLib::stringToScatterType(const QString str) const
{
	if (str.isNull())
		return ScatterItem::NOSCATTER;
	if (!str.compare("LAMBERT2D"))
		return ScatterItem::LAMBERT2D;
	if (!str.compare("TORRSPARR1D"))
		return ScatterItem::TORRSPARR1D;
	if (!str.compare("TORRSPARR2D"))
		return ScatterItem::TORRSPARR2D;
	if (!str.compare("TORRSPARR2DPATHTRACE"))
		return ScatterItem::TORRSPARR2DPATHTRACE;
	if (!str.compare("DISPDOUBLECAUCHY1D"))
		return ScatterItem::DISPDOUBLECAUCHY1D;
	if (!str.compare("DOUBLECAUCHY1D"))
		return ScatterItem::DOUBLECAUCHY1D;
	if (!str.compare("PHONG"))
		return ScatterItem::PHONG;
	if (!str.compare("COOKTORRANCE"))
		return ScatterItem::COOKTORRANCE;
	if (!str.compare("NOSCATTER"))
		return ScatterItem::NOSCATTER;

	return ScatterItem::NOSCATTER;
}

QString ScatterItemLib::scatterPupilTypeToString(const ScatterItem::ScatterPupilType type) const
{
	QString str;
	switch (type)
	{
    case ScatterItem::NOPUPIL:
		str= "INFTY";
		break;
	case ScatterItem::RECTPUPIL:
		str= "RECTANGULAR";
		break;
	case ScatterItem::ELLIPTPUPIL:
		str= "ELLIPTICAL";
		break;
	default:
		str="INFTY";
		break;
	}

	return str;
}

ScatterItem::ScatterPupilType ScatterItemLib::stringToScatterPupilType(const QString str) const
{
	if (str.isNull())
		return ScatterItem::NOPUPIL;
	if (!str.compare("INFTY"))
		return ScatterItem::NOPUPIL;
	if (!str.compare("RECTANGULAR"))
		return ScatterItem::RECTPUPIL;
	if (!str.compare("ELLIPTICAL"))
		return ScatterItem::ELLIPTPUPIL;

	return ScatterItem::NOPUPIL;
}

ScatterItem::ScatterType ScatterItemLib::matScatTypeToScatType(const MaterialItem::Mat_ScatterType type) const
{
	ScatterItem::ScatterType typeOut;
	switch (type)
	{
	case MaterialItem::LAMBERT2D:
		typeOut= ScatterItem::LAMBERT2D;
		break;
	case MaterialItem::TORRSPARR1D:
		typeOut= ScatterItem::TORRSPARR1D;
		break;
	case MaterialItem::TORRSPARR2D:
		typeOut= ScatterItem::TORRSPARR2D;
		break;
	case MaterialItem::TORRSPARR2DPATHTRACE:
		typeOut= ScatterItem::TORRSPARR2DPATHTRACE;
		break;
	case MaterialItem::DISPDOUBLECAUCHY1D:
		typeOut= ScatterItem::DISPDOUBLECAUCHY1D;
		break;
	case MaterialItem::DOUBLECAUCHY1D:
		typeOut= ScatterItem::DOUBLECAUCHY1D;
		break;
	case MaterialItem::PHONG:
		typeOut= ScatterItem::PHONG;
		break;
	case MaterialItem::COOKTORRANCE:
		typeOut= ScatterItem::COOKTORRANCE;
		break;
	case MaterialItem::NOSCATTER:
		typeOut= ScatterItem::NOSCATTER;
		break;
	default:
		typeOut=ScatterItem::NOSCATTER;
		break;
	}

	return typeOut;
};

MaterialItem::Mat_ScatterType ScatterItemLib::scatTypeToMatScatType(const ScatterItem::ScatterType type) const
{
	MaterialItem::Mat_ScatterType typeOut;
	switch (type)
	{
	case ScatterItem::LAMBERT2D:
		typeOut= MaterialItem::LAMBERT2D;
		break;
	case ScatterItem::TORRSPARR1D:
		typeOut= MaterialItem::TORRSPARR1D;
		break;
	case ScatterItem::TORRSPARR2D:
		typeOut= MaterialItem::TORRSPARR2D;
		break;
	case ScatterItem::TORRSPARR2DPATHTRACE:
		typeOut= MaterialItem::TORRSPARR2DPATHTRACE;
		break;
	case ScatterItem::DISPDOUBLECAUCHY1D:
		typeOut= MaterialItem::DISPDOUBLECAUCHY1D;
		break;
	case ScatterItem::DOUBLECAUCHY1D:
		typeOut= MaterialItem::DOUBLECAUCHY1D;
		break;
	case ScatterItem::PHONG:
		typeOut= MaterialItem::PHONG;
		break;
	case ScatterItem::COOKTORRANCE:
		typeOut= MaterialItem::COOKTORRANCE;
		break;
	case ScatterItem::NOSCATTER:
		typeOut= MaterialItem::NOSCATTER;
		break;
	default:
		typeOut=MaterialItem::NOSCATTER;
		break;
	}

	return typeOut;
}
