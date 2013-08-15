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

#include "geometryItemLib.h"

using namespace macrosim;

GeometryItem* GeometryItemLib::createGeometry(GeometryItem::GeomType type)
{
	GeometryItem* l_pItem=NULL;
		// if material changed, we need to create an instance of the new material and append it
		switch (type)
		{
		case GeometryItem::SPHERICALLENSE:
			l_pItem=new SphericalLenseItem();
			break;
		case GeometryItem::CYLLENSESURF:
			l_pItem=new CylLensSurfaceItem();
			break;
		case GeometryItem::SPHERICALSURFACE:
			l_pItem=new SphericalSurfaceItem();
			break;
		case GeometryItem::PARABOLICSURFACE:
			l_pItem=new ParabolicSurfaceItem();
			break;
		case GeometryItem::PLANESURFACE:
			l_pItem=new PlaneSurfaceItem();
			break;
		case GeometryItem::IDEALLENSE:
			l_pItem=new IdealLenseItem();
			break;
		case GeometryItem::APERTURESTOP:
			l_pItem=new ApertureStopItem();
			break;
		case GeometryItem::ASPHERICALSURF:
			l_pItem=new AsphericalSurfaceItem();
			break;
		case GeometryItem::CYLPIPE:
			l_pItem=new CylPipeItem();
			break;
		case GeometryItem::CONEPIPE:
			l_pItem=new ConePipeItem();
			break;
		case GeometryItem::MICROLENSARRAY:
			l_pItem=new MicroLensArrayItem();
			break;
		case GeometryItem::APERTUREARRAY:
			l_pItem=new ApertureArrayItem();
			break;
		case GeometryItem::STOPARRAY:
			l_pItem=new StopArrayItem();
			break;
		case GeometryItem::CADOBJECT:
			l_pItem=new CadObjectItem();
			break;
		case GeometryItem::SUBSTRATE:
			l_pItem=new SubstrateItem();
			break;
		case GeometryItem::VOLUMESCATTERER:
			l_pItem=new VolumeScattererItem();
			break;

		//case GeometryItem::DETECTOR:
		//	l_pItem=new MaterialPathTraceSourceItem();
		//	break;
		default:
			break;
		}
		return l_pItem;
}

//QString GeometryItemLib::apertureTypeToString(const GeometryItem::ApertureType type) const
//{
//	QString str;
//	switch (type)
//	{
//	case GeometryItem::RECTANGULAR:
//		str="RECTANGULAR";
//		break;
//	case GeometryItem::ELLIPTICAL:
//		str="ELLIPTICAL";
//		break;
//	default:
//		str="UNKNOWN";
//		break;
//	}
//	return str;
//};


//GeometryItem::ApertureType GeometryItemLib::stringToApertureType(const QString str) const
//{
//	if (!str.compare("RECTANGULAR") == 0)
//		return GeometryItem::RECTANGULAR;
//	if (!str.compare("ELLIPTICAL") == 0)
//		return GeometryItem::ELLIPTICAL;
//	return GeometryItem::UNKNOWN;
//};

QString GeometryItemLib::geomTypeToString(const GeometryItem::GeomType type) const
{
	QString str;
	switch (type)
	{
	case GeometryItem::SPHERICALLENSE:
		str="SPHERICALLENSE";
		break;
	case GeometryItem::CYLLENSESURF:
		str="CYLLENSESURF";
		break;
	case GeometryItem::SPHERICALSURFACE:
		str="SPHERICALSURFACE";
		break;
	case GeometryItem::PARABOLICSURFACE:
		str="PARABOLICSURFACE";
		break;
	case GeometryItem::PLANESURFACE:
		str="PLANESURFACE";
		break;
	case GeometryItem::IDEALLENSE:
		str="IDEALLENSE";
		break;
	case GeometryItem::APERTURESTOP:
		str="APERTURESTOP";
		break;
	case GeometryItem::ASPHERICALSURF:
		str="ASPHERICALSURF";
		break;
	case GeometryItem::CYLPIPE:
		str="CYLPIPE";
		break;
	case GeometryItem::CONEPIPE:
		str="CONEPIPE";
		break;
	case GeometryItem::DETECTOR:
		str="DETECTOR";
		break;
	case GeometryItem::MICROLENSARRAY:
		str="MICROLENSARRAY";
		break;
	case GeometryItem::APERTUREARRAY:
		str="APERTUREARRAY";
		break;
	case GeometryItem::STOPARRAY:
		str="STOPARRAY";
		break;
	case GeometryItem::CADOBJECT:
		str="CADOBJECT";
		break;
	case GeometryItem::SUBSTRATE:
		str="SUBSTRATE";
		break;
	case GeometryItem::VOLUMESCATTERER:
		str="VOLUMESCATTERER";
		break;

	default:
		str="UNKNOWN";
		break;
	}
	return str;
};

GeometryItem::GeomType GeometryItemLib::stringToGeomType(const QString str) const
{
	if (str.isNull())
		return GeometryItem::UNDEFINED;
	if (!str.compare("SPHERICALLENSE") )
		return GeometryItem::SPHERICALLENSE;
	if (!str.compare("PARABOLICSURFACE") )
		return GeometryItem::PARABOLICSURFACE;
	if (!str.compare("CYLLENSESURF"))
		return GeometryItem::CYLLENSESURF;
	if (!str.compare("SPHERICALSURFACE"))
		return GeometryItem::SPHERICALSURFACE;
	if (!str.compare("PLANESURFACE"))
		return GeometryItem::PLANESURFACE;
	if (!str.compare("APERTURESTOP"))
		return GeometryItem::APERTURESTOP;
	if (!str.compare("ASPHERICALSURF"))
		return GeometryItem::ASPHERICALSURF;
	if (!str.compare("CYLPIPE"))
		return GeometryItem::CYLPIPE;
	if (!str.compare("CONEPIPE"))
		return GeometryItem::CONEPIPE;
	if (!str.compare("IDEALLENSE"))
		return GeometryItem::IDEALLENSE;
	if (!str.compare("MICROLENSARRAY"))
		return GeometryItem::MICROLENSARRAY;
	if (!str.compare("APERTUREARRAY"))
		return GeometryItem::APERTUREARRAY;
	if (!str.compare("STOPARRAY"))
		return GeometryItem::STOPARRAY;
	if (!str.compare("CADOBJECT"))
		return GeometryItem::CADOBJECT;
	if (!str.compare("SUBSTRATE"))
		return GeometryItem::SUBSTRATE;
	if (!str.compare("VOLUMESCATTERER"))
		return GeometryItem::VOLUMESCATTERER;

	//if (!str.compare("DETECTOR"))
	//	return GeometryItem::DETECTOR;
	return GeometryItem::UNDEFINED;
};

GeometryItem::Abstract_MaterialType GeometryItemLib::stringToGeomMatType(const QString str) const
{
	if (str.isNull())
		return GeometryItem::ABSORBING;
	if (!str.compare("REFRACTING"))
		return GeometryItem::REFRACTING;
	if (!str.compare("ABSORBING"))
		return GeometryItem::ABSORBING;
	if (!str.compare("DIFFRACTING"))
		return GeometryItem::DIFFRACTING;
	if (!str.compare("FILTER"))
		return GeometryItem::FILTER;
	if (!str.compare("LINGRAT1D"))
		return GeometryItem::LINGRAT1D;
	if (!str.compare("IDEALLENSE"))
		return GeometryItem::MATIDEALLENSE;
	if (!str.compare("REFLECTING"))
		return GeometryItem::REFLECTING;
	if (!str.compare("REFLECTINGCOVGLASS"))
		return GeometryItem::REFLECTINGCOVGLASS;
	if (!str.compare("PATHTRACESOURCE"))
		return GeometryItem::PATHTRACESOURCE;
	if (!str.compare("DOE"))
		return GeometryItem::DOE;
	if (!str.compare("VOLUMESCATTER"))
		return GeometryItem::VOLUMESCATTER;

	return GeometryItem::ABSORBING;
}

QString GeometryItemLib::geomMatTypeToString(const GeometryItem::Abstract_MaterialType type) const
{
	QString str;
	switch (type)
	{
	case GeometryItem::REFRACTING:
		str = "REFRACTING";
		break;
	case GeometryItem::ABSORBING:
		str = "ABSORBING";
		break;
	case GeometryItem::DIFFRACTING:
		str = "DIFFRACTING";
		break;
	case GeometryItem::FILTER:
		str = "FILTER";
		break;
	case GeometryItem::LINGRAT1D:
		str = "LINGRAT1D";
		break;
	case GeometryItem::MATIDEALLENSE:
		str = "IDEALLENSE";
		break;
	case GeometryItem::REFLECTING:
		str = "REFLECTING";
		break;
	case GeometryItem::REFLECTINGCOVGLASS:
		str = "REFLECTINGCOVGLASS";
		break;
	case GeometryItem::PATHTRACESOURCE:
		str = "PATHTRACESOURCE";
		break;
	case GeometryItem::DOE:
		str = "DOE";
		break;
	case GeometryItem::VOLUMESCATTER:
		str = "VOLUMESCATTER";
		break;

	default:
		break;
	}
	return str;
}

QList<AbstractItem*> GeometryItemLib::fillLibrary() const
{
	QList<AbstractItem*> l_list;
	l_list.append(new SphericalLenseItem());
	l_list.append(new SphericalSurfaceItem());
	l_list.append(new PlaneSurfaceItem());
	l_list.append(new AsphericalSurfaceItem());
	l_list.append(new CylLensSurfaceItem());
	l_list.append(new CylPipeItem());
	l_list.append(new ConePipeItem());
	l_list.append(new IdealLenseItem());
	l_list.append(new ApertureStopItem());
	l_list.append(new MicroLensArrayItem());
	l_list.append(new ApertureArrayItem());
	l_list.append(new ParabolicSurfaceItem());
	l_list.append(new CadObjectItem());
	l_list.append(new SubstrateItem());
	l_list.append(new StopArrayItem());
	l_list.append(new VolumeScattererItem());
	return l_list;
}
