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

#include "materialLinearGrating1DItem.h"

using namespace macrosim;

MaterialLinearGrating1DItem::MaterialLinearGrating1DItem(QString name, QObject *parent) :
	MaterialItem(LINGRAT1D, name, parent),
		m_diffAxis(Vec3d(1.0,0.0,0.0)),
		m_diffOrders(Vec9si(0,0,0,0,0,0,0,0,0)),
		m_diffEffs(Vec9d(0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0)),
		m_n1(1),
		m_n2(1),
		m_glassName("USERDEFINED"),
		m_immersionName("USERDEFINED"),
		m_diffFileName("USERDEFINED")
{
}


MaterialLinearGrating1DItem::~MaterialLinearGrating1DItem()
{
}

bool MaterialLinearGrating1DItem::writeToXML(QDomDocument &document, QDomElement &root) const
{
	QDomElement material = document.createElement("material");
	material.setAttribute("diffAxis.x", QString::number(m_diffAxis.X));
	material.setAttribute("diffAxis.y", QString::number(m_diffAxis.Y));
	material.setAttribute("diffAxis.z", QString::number(m_diffAxis.Z));
	material.setAttribute("materialType", "LINGRAT1D");
	material.setAttribute("diffOrder.x1", QString::number(m_diffOrders.X1));
	material.setAttribute("diffOrder.x2", QString::number(m_diffOrders.X2));
	material.setAttribute("diffOrder.x3", QString::number(m_diffOrders.X3));
	material.setAttribute("diffOrder.x4", QString::number(m_diffOrders.X4));
	material.setAttribute("diffOrder.x5", QString::number(m_diffOrders.X5));
	material.setAttribute("diffOrder.x6", QString::number(m_diffOrders.X6));
	material.setAttribute("diffOrder.x7", QString::number(m_diffOrders.X7));
	material.setAttribute("diffOrder.x8", QString::number(m_diffOrders.X8));
	material.setAttribute("diffOrder.x9", QString::number(m_diffOrders.X9));
	material.setAttribute("diffEff.x1", QString::number(m_diffEffs.X1));
	material.setAttribute("diffEff.x2", QString::number(m_diffEffs.X2));
	material.setAttribute("diffEff.x3", QString::number(m_diffEffs.X3));
	material.setAttribute("diffEff.x4", QString::number(m_diffEffs.X4));
	material.setAttribute("diffEff.x5", QString::number(m_diffEffs.X5));
	material.setAttribute("diffEff.x6", QString::number(m_diffEffs.X6));
	material.setAttribute("diffEff.x7", QString::number(m_diffEffs.X7));
	material.setAttribute("diffEff.x8", QString::number(m_diffEffs.X8));
	material.setAttribute("diffEff.x9", QString::number(m_diffEffs.X9));
	material.setAttribute("gratingPeriod", QString::number(m_gratingPeriod));

	material.setAttribute("n1", QString::number(m_n1));
	material.setAttribute("n2", QString::number(m_n2));
	material.setAttribute("materialType", "LINGRAT1D");
	material.setAttribute("glassName", m_glassName);
	material.setAttribute("immersionName", m_immersionName);
	material.setAttribute("diffFileName", m_diffFileName);

	// write parameters inherited from base class
	if (!MaterialItem::writeToXML(document, material))
		return false;

	root.appendChild(material);

	return true;
}

bool MaterialLinearGrating1DItem::readFromXML(const QDomElement &node)
{
	if (!MaterialItem::readFromXML(node))
		return false;

	m_diffAxis.X=node.attribute("diffAxis.x").toDouble();
	m_diffAxis.Y=node.attribute("diffAxis.y").toDouble();
	m_diffAxis.Z=node.attribute("diffAxis.z").toDouble();
	m_diffOrders.X1=node.attribute("diffOrder.x1").toInt();
	m_diffOrders.X2=node.attribute("diffOrder.x2").toInt();
	m_diffOrders.X3=node.attribute("diffOrder.x3").toInt();
	m_diffOrders.X4=node.attribute("diffOrder.x4").toInt();
	m_diffOrders.X5=node.attribute("diffOrder.x5").toInt();
	m_diffOrders.X6=node.attribute("diffOrder.x6").toInt();
	m_diffOrders.X7=node.attribute("diffOrder.x7").toInt();
	m_diffOrders.X8=node.attribute("diffOrder.x8").toInt();
	m_diffOrders.X9=node.attribute("diffOrder.x9").toInt();
	m_diffEffs.X1=node.attribute("diffEff.x1").toDouble();
	m_diffEffs.X2=node.attribute("diffEff.x2").toDouble();
	m_diffEffs.X3=node.attribute("diffEff.x3").toDouble();
	m_diffEffs.X4=node.attribute("diffEff.x4").toDouble();
	m_diffEffs.X5=node.attribute("diffEff.x5").toDouble();
	m_diffEffs.X6=node.attribute("diffEff.x6").toDouble();
	m_diffEffs.X7=node.attribute("diffEff.x7").toDouble();
	m_diffEffs.X8=node.attribute("diffEff.x8").toDouble();
	m_diffEffs.X9=node.attribute("diffEff.x9").toDouble();
	m_gratingPeriod=node.attribute("gratingPeriod").toDouble();
	m_n1=node.attribute("n1").toDouble();
	m_n2=node.attribute("n2").toDouble();
	m_glassName=node.attribute("glassName");
	m_immersionName=node.attribute("immersionName");
	m_diffFileName=node.attribute("diffFileName");


	return true;
}


