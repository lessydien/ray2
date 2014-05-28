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

#include "MaterialRenderFringeProjItem.h"

using namespace macrosim;

MaterialRenderFringeProjItem::MaterialRenderFringeProjItem(double power, FringeType fringeType, FringeOrientation orientation, double fringePeriod, int nrBits, int codeNr, double fringePhase, QString name, QObject *parent) :
	MaterialItem(REFRACTING, name, parent),
		m_power(power),
        m_fringeType(fringeType),
        m_fringeOrientation(orientation),
        m_fringePeriod(fringePeriod),
        m_nrBits(nrBits),
        m_codeNr(codeNr),
        m_fringePhase(fringePhase)
{
}


MaterialRenderFringeProjItem::~MaterialRenderFringeProjItem()
{
}

MaterialRenderFringeProjItem::FringeOrientation MaterialRenderFringeProjItem::strToFringeOrientation(QString str) const
{
	if (str.isNull())
		return FO_X;
	if (!str.compare("X"))
		return FO_X;
    if (!str.compare("Y"))
        return FO_Y;
    return FO_X;
}

QString MaterialRenderFringeProjItem::fringeOrientationToString(FringeOrientation fringeOrientation) const
{
	QString str;
	switch (fringeOrientation)
	{
	case FO_X:
		str="X";
		break;
    case FO_Y:
        str="Y";
        break;
    default:
        str="X";
        break;
    }
    return str;
}

MaterialRenderFringeProjItem::FringeType MaterialRenderFringeProjItem::strToFringeType(QString str) const
{
	if (str.isNull())
		return UNKNOWN;
	if (!str.compare("GRAYCODE"))
		return GRAYCODE;
    if (!str.compare("SINUS"))
        return SINUS;
    return UNKNOWN;
}

QString MaterialRenderFringeProjItem::fringeTypeToString(FringeType fringeType) const
{
	QString str;
	switch (fringeType)
	{
	case UNKNOWN:
		str="UNKNOWN";
		break;
    case GRAYCODE:
        str="GRAYCODE";
        break;
    case SINUS:
        str="SINUS";
        break;
    default:
        str="UNKNOWN";
        break;
    }
    return str;
}

bool MaterialRenderFringeProjItem::writeToXML(QDomDocument &document, QDomElement &root) const
{
	QDomElement material = document.createElement("material");
	material.setAttribute("materialType", "RENDERFRINGEPROJ");
	material.setAttribute("power", QString::number(m_power));
    material.setAttribute("pupilRoot.x", QString::number(m_pupilRoot.X));
    material.setAttribute("pupilRoot.y", QString::number(m_pupilRoot.Y));
    material.setAttribute("pupilRoot.z", QString::number(m_pupilRoot.Z));

    material.setAttribute("pupilTilt.x", QString::number(m_pupilTilt.X));
    material.setAttribute("pupilTilt.y", QString::number(m_pupilTilt.Y));
    material.setAttribute("pupilTilt.z", QString::number(m_pupilTilt.Z));

    material.setAttribute("pupilAptRad.x", QString::number(m_pupilAptRad.X));
    material.setAttribute("pupilAptRad.y", QString::number(m_pupilAptRad.Y));

    material.setAttribute("fringeType", fringeTypeToString(m_fringeType));
    material.setAttribute("fringeOrientation", fringeOrientationToString(m_fringeOrientation));
    material.setAttribute("fringePeriod", QString::number(m_fringePeriod));
    material.setAttribute("nrBits", QString::number(m_nrBits));
    material.setAttribute("codeNr", QString::number(m_codeNr));
    material.setAttribute("fringePhase", QString::number(m_fringePhase));

	// write parameters inherited from base class
	if (!MaterialItem::writeToXML(document, material))
		return false;

	Vec3d l_tilt;
	Vec3d l_root;

	QModelIndex l_index=this->getModelIndex();
	QModelIndex l_parentIndex=l_index.parent();
	QModelIndex test=QModelIndex();

	AbstractItem* l_pAbstractItem=reinterpret_cast<AbstractItem*>(l_parentIndex.internalPointer());
	if (l_pAbstractItem != NULL)
	{
		if (l_pAbstractItem->getObjectType() == GEOMETRY)
		{
			GeometryItem* l_pGeomItem=reinterpret_cast<GeometryItem*>(l_pAbstractItem);
			l_root=l_pGeomItem->getRoot();
			l_tilt=l_pGeomItem->getTilt();
		}
		else
		{
			cout << "error in materialDOEItem.writeToXML(): parent seems to not be a geometry. Probably the model is messed up" << endl;
		}
	}
	material.setAttribute("geomRoot.x", QString::number(l_root.X));
	material.setAttribute("geomRoot.y", QString::number(l_root.Y));
	material.setAttribute("geomRoot.z", QString::number(l_root.Z));
	material.setAttribute("geomTilt.x", QString::number(l_tilt.X));
	material.setAttribute("geomTilt.y", QString::number(l_tilt.Y));
	material.setAttribute("geomTilt.z", QString::number(l_tilt.Z));

	root.appendChild(material);

	return true;
}

bool MaterialRenderFringeProjItem::readFromXML(const QDomElement &node)
{
	if (!MaterialItem::readFromXML(node))
		return false;

    m_pupilRoot.X=node.attribute("pupilRoot.x").toDouble();
    m_pupilRoot.Y=node.attribute("pupilRoot.y").toDouble();
    m_pupilRoot.Z=node.attribute("pupilRoot.z").toDouble();

    m_pupilTilt.X=node.attribute("pupilTilt.x").toDouble();
    m_pupilTilt.Y=node.attribute("pupilTilt.y").toDouble();
    m_pupilTilt.Z=node.attribute("pupilTilt.z").toDouble();

    m_pupilAptRad.X=node.attribute("pupilAptRad.x").toDouble();
    m_pupilAptRad.Y=node.attribute("pupilAptRad.y").toDouble();

    m_power=node.attribute("power").toDouble();

    m_fringeType=strToFringeType(node.attribute("fringeType"));
    m_fringeOrientation=strToFringeOrientation(node.attribute("fringeOrientation"));
    m_fringePeriod=node.attribute("fringePeriod").toDouble();
    m_fringePhase=node.attribute("fringePhase").toDouble();
    m_nrBits=node.attribute("nrBits").toInt();
    m_codeNr=node.attribute("codeNr").toInt();

	return true;
}

