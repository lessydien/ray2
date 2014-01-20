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

#include "fieldItemLib.h"

using namespace macrosim;

FieldItem* FieldItemLib::createField(FieldItem::FieldType type)
{
	FieldItem* l_pItem=NULL;
		// if material changed, we need to create an instance of the new material and append it
		switch (type)
		{
		case FieldItem::GEOMRAYFIELD:
			l_pItem=new GeomRayFieldItem();
			break;
		case FieldItem::GEOMRAYFIELD_PSEUDOBANDWIDTH:
			l_pItem=new GeomRayFieldItem_PseudoBandwidth();
			break;
		case FieldItem::DIFFRAYFIELD:
			l_pItem=new DiffRayFieldItem();
			break;
		case FieldItem::DIFFRAYFIELDRAYAIM:
			l_pItem=new DiffRayField_RayAiming_Item();
			break;
		case FieldItem::INTENSITYFIELD:
			l_pItem=new IntensityFieldItem();
			break;
		case FieldItem::SCALARGAUSSIANWAVE:
			l_pItem=new ScalarGaussianFieldItem();
			break;
		case FieldItem::SCALARPLANEWAVE:
			l_pItem=new ScalarPlaneFieldItem();
			break;
		case FieldItem::SCALARSPHERICALWAVE:
			l_pItem=new ScalarSphericalFieldItem();
			break;
		case FieldItem::SCALARUSERWAVE:
			l_pItem=new ScalarUserFieldItem();
			break;
		case FieldItem::PATHINTTISSUERAYFIELD:
			l_pItem=new PathIntTissueFieldItem();
			break;

		default:
			break;
		}
		return l_pItem;
}


QString FieldItemLib::fieldTypeToString(const FieldItem::FieldType type) const
{
	QString str;
	switch (type)
	{
	case FieldItem::GEOMRAYFIELD:
		str="GEOMRAYFIELD";
		break;
	case FieldItem::GEOMRAYFIELD_PSEUDOBANDWIDTH:
		str="GEOMRAYFIELD_PSEUDOBANDWIDTH";
		break;
	case FieldItem::DIFFRAYFIELD:
		str="DIFFRAYFIELD";
		break;
	case FieldItem::DIFFRAYFIELDRAYAIM:
		str="DIFFRAYFIELDRAYAIM";
		break;
	case FieldItem::INTENSITYFIELD:
		str="INTENSITYFIELD";
		break;
	case FieldItem::SCALARPLANEWAVE:
		str="SCALARPLANEWAVE";
		break;
	case FieldItem::SCALARSPHERICALWAVE:
		str="SCALARSPHERICALWAVE";
		break;
	case FieldItem::SCALARGAUSSIANWAVE:
		str="SCALARGAUSSIANWAVE";
		break;
	case FieldItem::SCALARUSERWAVE:
		str="SCALARUSERWAVE";
		break;
	case FieldItem::PATHINTTISSUERAYFIELD:
		str="PATHINTTISSUERAYFIELD";
		break;
	default:
		str="UNKNOWN";
		break;
	}
	return str;
};

FieldItem::FieldType FieldItemLib::stringToFieldType(const QString str) const
{
	if (str.isNull())
		return FieldItem::UNDEFINED;
	if (!str.compare("GEOMRAYFIELD_PSEUDOBANDWIDTH") )
		return FieldItem::GEOMRAYFIELD_PSEUDOBANDWIDTH;
	if (!str.compare("GEOMRAYFIELD") )
		return FieldItem::GEOMRAYFIELD;
	if (!str.compare("DIFFRAYFIELD") )
		return FieldItem::DIFFRAYFIELD;
	if (!str.compare("DIFFRAYFIELDRAYAIM") )
		return FieldItem::DIFFRAYFIELDRAYAIM;
	if (!str.compare("SCALARPLANEWAVE") )
		return FieldItem::SCALARPLANEWAVE;
	if (!str.compare("SCALARSPHERICALWAVE") )
		return FieldItem::SCALARSPHERICALWAVE;
	if (!str.compare("SCALARGAUSSIANWAVE") )
		return FieldItem::SCALARGAUSSIANWAVE;
	if (!str.compare("SCALARUSERWAVE") )
		return FieldItem::SCALARUSERWAVE;
	if (!str.compare("INTENSITYFIELD") )
		return FieldItem::INTENSITYFIELD;
	if (!str.compare("PATHINTTISSUERAYFIELD") )
		return FieldItem::PATHINTTISSUERAYFIELD;

	return FieldItem::UNDEFINED;
};

QList<AbstractItem*> FieldItemLib::fillLibrary() const
{
	QList<AbstractItem*> l_list;
	l_list.append(new GeomRayFieldItem());
	l_list.append(new GeomRayFieldItem_PseudoBandwidth());
	l_list.append(new DiffRayFieldItem());
	l_list.append(new DiffRayField_RayAiming_Item());
	//l_list.append(new IntensityFieldItem());
	//l_list.append(new ScalarPlaneFieldItem());
	//l_list.append(new ScalarSphericalFieldItem());
	//l_list.append(new ScalarGaussianFieldItem());
	//l_list.append(new ScalarUserFieldItem());
	//l_list.append(new PathIntTissueFieldItem());
	return l_list;
}
