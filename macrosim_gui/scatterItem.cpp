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

#include "ScatterItem.h"
#include "scatterItemLib.h"

using namespace macrosim;

ScatterItem::ScatterItem(ScatterType type, QString name, QObject *parent) :
	AbstractItem(SCATTER, name, parent),
		m_scatterType(type),
        m_pupRoot(Vec3d(0.0,0.0,0.0)),
        m_pupTilt(Vec3d(0.0,0.0,0.0)),
        m_pupAptRad(Vec2d(0.0,0.0)),
        m_pupAptType(NOPUPIL)
{
}


ScatterItem::~ScatterItem()
{
}

void ScatterItem::changeItem(const QModelIndex &topLeft, const QModelIndex &bottomRight)
{
	emit itemChanged(m_index, m_index);
}

bool ScatterItem::writeToXML(QDomDocument &document, QDomElement &scatter) const
{
	if (!AbstractItem::writeToXML(document, scatter))
		return false;

    scatter.setAttribute("pupRoot.x", QString::number(m_pupRoot.X));
    scatter.setAttribute("pupRoot.y", QString::number(m_pupRoot.Y));
    scatter.setAttribute("pupRoot.z", QString::number(m_pupRoot.Z));
    scatter.setAttribute("pupTilt.x", QString::number(m_pupTilt.X));
    scatter.setAttribute("pupTilt.y", QString::number(m_pupTilt.Y));
    scatter.setAttribute("pupTilt.z", QString::number(m_pupTilt.Z));
    scatter.setAttribute("pupAptRad.x", QString::number(m_pupAptRad.X));
    scatter.setAttribute("pupAptRad.y", QString::number(m_pupAptRad.Y));
    
    ScatterItemLib l_lib;
    scatter.setAttribute("pupAptType", l_lib.scatterPupilTypeToString(m_pupAptType));
	return true;
}

bool ScatterItem::readFromXML(const QDomElement &node)
{
	m_pupRoot.X=node.attribute("pupRoot.x").toDouble();
    m_pupRoot.Y=node.attribute("pupRoot.y").toDouble();
    m_pupRoot.Z=node.attribute("pupRoot.z").toDouble();
    m_pupTilt.X=node.attribute("pupTilt.x").toDouble();
    m_pupTilt.Y=node.attribute("pupTilt.y").toDouble();
    m_pupTilt.Z=node.attribute("pupTilt.z").toDouble();
    m_pupAptRad.X=node.attribute("pupAptRad.x").toDouble();
    m_pupAptRad.Y=node.attribute("pupAptRad.y").toDouble();

    ScatterItemLib l_lib;
    m_pupAptType=l_lib.stringToScatterPupilType(node.attribute("pupAptType"));
	return true;
}