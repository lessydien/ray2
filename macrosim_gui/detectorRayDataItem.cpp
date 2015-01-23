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

#include "detectorRayDataItem.h"

using namespace macrosim;

DetectorRayDataItem::DetectorRayDataItem(QString name, QObject *parent) :
	DetectorItem(name, RAYDATA, parent)
{
}

DetectorRayDataItem::~DetectorRayDataItem()
{
	m_childs.clear();
}


bool DetectorRayDataItem::writeToXML(QDomDocument &document, QDomElement &root) const 
{
	QDomElement node = document.createElement("detector");

	if (!DetectorItem::writeToXML(document, node))
		return false;
	if (m_listAllRays)
		node.setAttribute("listAllRays", "true");
	else
		node.setAttribute("listAllRays", "false");
	if (m_reduceData)
		node.setAttribute("reduceData", "true");
	else
		node.setAttribute("reduceData", "false");
	root.appendChild(node);
	return true;
}

bool DetectorRayDataItem::readFromXML(const QDomElement &node)
{
	// read base class
	if (!DetectorItem::readFromXML(node))
		return false;

	QString listAllRaysString=node.attribute("listAllRays");
	if (!listAllRaysString.compare("true"))
		m_listAllRays=true;
	else
		m_listAllRays=false;
	QString reduceDataString=node.attribute("reduceData");
	if (!reduceDataString.compare("true"))
		m_reduceData=true;
	else
		m_reduceData=false;

	return true;
}