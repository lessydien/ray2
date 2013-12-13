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

#include "detectorItemLib.h"

using namespace macrosim;

DetectorItem* DetectorItemLib::createDetector(DetectorItem::DetType type)
{
	DetectorItem* l_pItem=NULL;
		// if material changed, we need to create an instance of the new material and append it
		switch (type)
		{
		case DetectorItem::INTENSITY:
			l_pItem=new DetectorIntensityItem();
			break;
		case DetectorItem::VOLUMEINTENSITY:
			l_pItem=new DetectorVolumeIntensityItem();
			break;
		case DetectorItem::RAYDATA:
			l_pItem=new DetectorRayDataItem();
			break;
		case DetectorItem::FIELD:
			l_pItem=new DetectorFieldItem();
			break;
		default:
			break;
		}
		return l_pItem;
}

QString DetectorItemLib::detTypeToString(const DetectorItem::DetType type) const
{
	QString str;
	switch (type)
	{
	case DetectorItem::INTENSITY:
		str="INTENSITY";
		break;
	case DetectorItem::VOLUMEINTENSITY:
		str="VOLUMEINTENSITY";
		break;
	case DetectorItem::RAYDATA:
		str="RAYDATA";
		break;
	case DetectorItem::FIELD:
		str="FIELD";
		break;
	default:
		str="UNKNOWN";
		break;
	}
	return str;
};


QString DetectorItemLib::detOutFormatToString(const DetectorItem::DetOutFormat format) const
{
	QString str;
	switch (format)
	{
	case DetectorItem::MAT:
		str="MAT";
		break;
	case DetectorItem::TEXT:
		str="TEXT";
		break;
	default:
		str="TEXT";
		break;
	}
	return str;
};

DetectorItem::DetOutFormat DetectorItemLib::stringToDetOutFormat(const QString str) const
{
	if (!str.compare("MAT"))
		return DetectorItem::MAT;
	if (!str.compare("TEXT"))
		return DetectorItem::TEXT;
	// default to TEXT
	return DetectorItem::TEXT;
};

DetectorItem::DetType DetectorItemLib::stringToDetType(const QString str) const
{
	if (str.isNull())
		return DetectorItem::UNDEFINED;
	if (!str.compare("INTENSITY") )
		return DetectorItem::INTENSITY;
	if (!str.compare("VOLUMEINTENSITY") )
		return DetectorItem::VOLUMEINTENSITY;
	if (!str.compare("RAYDATA") )
		return DetectorItem::RAYDATA;
	if (!str.compare("FIELD") )
		return DetectorItem::FIELD;

	return DetectorItem::UNDEFINED;
};

QList<AbstractItem*> DetectorItemLib::fillLibrary() const
{
	QList<AbstractItem*> l_list;
	l_list.append(new DetectorIntensityItem());
	l_list.append(new DetectorFieldItem());
	l_list.append(new DetectorRayDataItem());
	l_list.append(new DetectorVolumeIntensityItem());
	return l_list;
}
