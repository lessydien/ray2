
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

#include "materialItemLib.h"

#include "materialItem.h"

using namespace macrosim;

MaterialItem* MaterialItemLib::createMaterial(MaterialItem::MaterialType type)
{
	MaterialItem* l_pItem;
		// if material changed, we need to create an instance of the new material and append it
		switch (type)
		{
		case MaterialItem::REFRACTING:
			l_pItem=new MaterialRefractingItem();
			break;
		case MaterialItem::DOE:
			l_pItem=new MaterialDOEItem();
			break;
		case MaterialItem::ABSORBING:
			l_pItem=new MaterialAbsorbingItem();
			break;
		case MaterialItem::DIFFRACTING:
			l_pItem=new MaterialDiffractingItem();
			break;
		case MaterialItem::FILTER:
			l_pItem=new MaterialFilterItem();
			break;
		case MaterialItem::LINGRAT1D:
			l_pItem=new MaterialLinearGrating1DItem();
			break;
		case MaterialItem::MATIDEALLENSE:
			l_pItem=new MaterialIdealLenseItem();
			break;
		case MaterialItem::REFLECTING:
			l_pItem=new MaterialReflectingItem();
			break;
		case MaterialItem::REFLECTINGCOVGLASS:
			l_pItem=new MaterialReflectingCovGlassItem();
			break;
		case MaterialItem::PATHTRACESOURCE:
			l_pItem=new MaterialPathTraceSourceItem();
			break;
		case MaterialItem::VOLUMESCATTER:
			l_pItem=new MaterialVolumeScatterItem();
			break;
		case MaterialItem::VOLUMEABSORBING:
			l_pItem=new MaterialVolumeAbsorbingItem();
			break;
        case MaterialItem::RENDERLIGHT:
            l_pItem=new MaterialRenderLightItem();
            break;
		default:
			break;
		}
		return l_pItem;
}

MaterialItem::MaterialType MaterialItemLib::stringToMaterialType(const QString str) const
{
	if (str.isNull())
		return MaterialItem::ABSORBING;
	if (!str.compare("REFRACTING"))
		return MaterialItem::REFRACTING;
	if (!str.compare("DOE"))
		return MaterialItem::DOE;
	if (!str.compare("ABSORBING"))
		return MaterialItem::ABSORBING;
	if (!str.compare("DIFFRACTING"))
		return MaterialItem::DIFFRACTING;
	if (!str.compare("FILTER"))
		return MaterialItem::FILTER;
	if (!str.compare("LINGRAT1D"))
		return MaterialItem::LINGRAT1D;
	if (!str.compare("MATIDEALLENSE"))
		return MaterialItem::MATIDEALLENSE;
	if (!str.compare("REFLECTING"))
		return MaterialItem::REFLECTING;
	if (!str.compare("REFLECTINGCOVGLASS"))
		return MaterialItem::REFLECTINGCOVGLASS;
	if (!str.compare("PATHTRACESOURCE"))
		return MaterialItem::PATHTRACESOURCE;
	if (!str.compare("VOLUMESCATTER"))
		return MaterialItem::VOLUMESCATTER;
	if (!str.compare("VOLUMEABSORBING"))
		return MaterialItem::VOLUMEABSORBING;
	if (!str.compare("VOLUMESCATTERBOX"))
		return MaterialItem::VOLUMESCATTER;
	if (!str.compare("RENDERLIGHT"))
		return MaterialItem::RENDERLIGHT;

	return MaterialItem::ABSORBING;
}

QString MaterialItemLib::materialTypeToString(const MaterialItem::MaterialType type) const
{
	QString str;
	switch (type)
	{
	case MaterialItem::REFRACTING:
		str = "REFRACTING";
		break;
	case MaterialItem::DOE:
		str = "DOE";
		break;
	case MaterialItem::ABSORBING:
		str = "ABSORBING";
		break;
	case MaterialItem::DIFFRACTING:
		str = "DIFFRACTING";
		break;
	case MaterialItem::FILTER:
		str = "FILTER";
		break;
	case MaterialItem::LINGRAT1D:
		str = "LINGRAT1D";
		break;
	case MaterialItem::MATIDEALLENSE:
		str = "MATIDEALLENSE";
		break;
	case MaterialItem::REFLECTING:
		str = "REFLECTING";
		break;
	case MaterialItem::REFLECTINGCOVGLASS:
		str = "REFLECTINGCOVGLASS";
		break;
	case MaterialItem::PATHTRACESOURCE:
		str = "PATHTRACESOURCE";
		break;
	case MaterialItem::VOLUMESCATTER:
		str = "VOLUMESCATTER";
		break;
	case MaterialItem::VOLUMEABSORBING:
		str = "VOLUMEABSORBING";
		break;
	case MaterialItem::RENDERLIGHT:
		str = "RENDERLIGHT";
		break;
	default:
		break;
	}
	return str;
}

MaterialItem::Mat_ScatterType MaterialItemLib::stringToMatScatterType(const QString str) const
{
	if (str.isNull())
		return MaterialItem::NOSCATTER;
	if (!str.compare("NOSCATTER"))
		return MaterialItem::NOSCATTER;
	if (!str.compare("LAMBERT2D"))
		return MaterialItem::LAMBERT2D;
	if (!str.compare("TORRSPARR1D"))
		return MaterialItem::TORRSPARR1D;
	if (!str.compare("TORRSPARR2D"))
		return MaterialItem::TORRSPARR2D;
	if (!str.compare("TORRSPARR2DPATHTRACE"))
		return MaterialItem::TORRSPARR2DPATHTRACE;
	if (!str.compare("DISPDOUBLECAUCHY1D"))
		return MaterialItem::DISPDOUBLECAUCHY1D;
	if (!str.compare("DOUBLECAUCHY1D"))
		return MaterialItem::DOUBLECAUCHY1D;
	return MaterialItem::NOSCATTER;
}

QString MaterialItemLib::matScatterTypeToString(const MaterialItem::Mat_ScatterType type) const
{
	QString str;
	switch (type)
	{
	case MaterialItem::NOSCATTER:
		str="NOSCATTER";
		break;
	case MaterialItem::LAMBERT2D:
		str="LAMBERT2D";
		break;
	case MaterialItem::TORRSPARR1D:
		str="TORRSPARR1D";
		break;
	case MaterialItem::TORRSPARR2D:
		str="TORRSPARR2D";
		break;
	case MaterialItem::TORRSPARR2DPATHTRACE:
		str="TORRSPARR2DPATHTRACE";
		break;
	case MaterialItem::DISPDOUBLECAUCHY1D:
		str="DISPDOUBLECAUCHY1D";
		break;
	case MaterialItem::DOUBLECAUCHY1D:
		str="DOUBLECAUCHY1D";
		break;
	default:
		str="NOSCATTER";
		break;
	}

	return str;
}

MaterialItem::Mat_CoatingType MaterialItemLib::stringToMatCoatingType(const QString str) const
{
	if (str.isNull())
		return MaterialItem::NOCOATING;
	if (!str.compare("NOCOATING"))
		return MaterialItem::NOCOATING;
	if (!str.compare("NUMCOEFFS"))
		return MaterialItem::NUMCOEFFS;

	return MaterialItem::NOCOATING;
}

QString MaterialItemLib::matCoatingTypeToString(const MaterialItem::Mat_CoatingType type) const
{
	QString str;
	switch (type)
	{
	case MaterialItem::NOCOATING:
		str="NOCOATING";
		break;
	case MaterialItem::NUMCOEFFS:
		str="NUMCOEFFS";
		break;
	default:
		str="NOCOATING";
		break;
	}
	return str;
}

MaterialItem::MaterialType MaterialItemLib::abstractMatTypeToMatType(const AbstractItem::Abstract_MaterialType type) const
{
	MaterialItem::MaterialType typeOut;
	switch (type)
	{
	case AbstractItem::REFRACTING:
		typeOut = MaterialItem::REFRACTING;
		break;
	case AbstractItem::DOE:
		typeOut = MaterialItem::DOE;
		break;
	case AbstractItem::ABSORBING:
		typeOut = MaterialItem::ABSORBING;
		break;
	case AbstractItem::DIFFRACTING:
		typeOut = MaterialItem::DIFFRACTING;
		break;
	case AbstractItem::FILTER:
		typeOut = MaterialItem::FILTER;
		break;
	case AbstractItem::LINGRAT1D:
		typeOut = MaterialItem::LINGRAT1D;
		break;
	case AbstractItem::MATIDEALLENSE:
		typeOut = MaterialItem::MATIDEALLENSE;
		break;
	case AbstractItem::REFLECTING:
		typeOut = MaterialItem::REFLECTING;
		break;
	case AbstractItem::REFLECTINGCOVGLASS:
		typeOut = MaterialItem::REFLECTINGCOVGLASS;
		break;
	case AbstractItem::PATHTRACESOURCE:
		typeOut = MaterialItem::PATHTRACESOURCE;
		break;
	case AbstractItem::VOLUMESCATTER:
		typeOut = MaterialItem::VOLUMESCATTER;
		break;
	case AbstractItem::VOLUMEABSORBING:
		typeOut = MaterialItem::VOLUMEABSORBING;
		break;
	case AbstractItem::RENDERLIGHT:
		typeOut = MaterialItem::RENDERLIGHT;
		break;
	default:
		break;
	}
	return typeOut;
};

AbstractItem::Abstract_MaterialType MaterialItemLib::matTypeToAbstractMatType(const MaterialItem::MaterialType type) const
{
	AbstractItem::Abstract_MaterialType typeOut;
	switch (type)
	{
	case MaterialItem::REFRACTING:
		typeOut = AbstractItem::REFRACTING;
		break;
	case MaterialItem::DOE:
		typeOut = AbstractItem::DOE;
		break;
	case MaterialItem::ABSORBING:
		typeOut = AbstractItem::ABSORBING;
		break;
	case MaterialItem::DIFFRACTING:
		typeOut = AbstractItem::DIFFRACTING;
		break;
	case MaterialItem::FILTER:
		typeOut = AbstractItem::FILTER;
		break;
	case MaterialItem::LINGRAT1D:
		typeOut = AbstractItem::LINGRAT1D;
		break;
	case MaterialItem::MATIDEALLENSE:
		typeOut = AbstractItem::MATIDEALLENSE;
		break;
	case MaterialItem::REFLECTING:
		typeOut = AbstractItem::REFLECTING;
		break;
	case MaterialItem::REFLECTINGCOVGLASS:
		typeOut = AbstractItem::REFLECTINGCOVGLASS;
		break;
	case MaterialItem::PATHTRACESOURCE:
		typeOut = AbstractItem::PATHTRACESOURCE;
		break;
	case MaterialItem::VOLUMESCATTER:
		typeOut = AbstractItem::VOLUMESCATTER;
		break;
	case MaterialItem::VOLUMEABSORBING:
		typeOut = AbstractItem::VOLUMEABSORBING;
		break;
	case MaterialItem::RENDERLIGHT:
		typeOut = AbstractItem::RENDERLIGHT;
		break;
	default:
		break;
	}
	return typeOut;
}
