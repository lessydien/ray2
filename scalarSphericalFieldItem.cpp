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

#include "scalarSphericalFieldItem.h"

using namespace macrosim;

ScalarSphericalFieldItem::ScalarSphericalFieldItem(QString name, QObject *parent) :
ScalarFieldItem(name, FieldItem::SCALARSPHERICALWAVE, parent),
	m_radius(Vec2d(1,1)),
	m_numApt(Vec2d(1,1))
{
};

ScalarSphericalFieldItem::~ScalarSphericalFieldItem()
{
	m_childs.clear();
};

bool ScalarSphericalFieldItem::signalDataChanged() 
{

	return true;
};

bool ScalarSphericalFieldItem::writeToXML(QDomDocument &document, QDomElement &root) const 
{
	QDomElement node = document.createElement("field");

	if (!ScalarFieldItem::writeToXML(document, node))
		return false;

	node.setAttribute("radius.x", QString::number(m_radius.X));
	node.setAttribute("radius.y", QString::number(m_radius.Y));
	node.setAttribute("numApt.x", QString::number(m_numApt.X));
	node.setAttribute("numApt.y", QString::number(m_numApt.Y));
	root.appendChild(node);
	return true;
};

bool ScalarSphericalFieldItem::readFromXML(const QDomElement &node)
{
	if (!ScalarFieldItem::readFromXML(node))
		return false;

	m_radius.X=node.attribute("radius.x").toDouble();
	m_radius.Y=node.attribute("radius.y").toDouble();
	m_numApt.X=node.attribute("numApt.x").toDouble();
	m_numApt.Y=node.attribute("numApt.y").toDouble();

	return true;
};

void ScalarSphericalFieldItem::render(QMatrix4x4 &m, RenderOptions &options)
{
	if (this->getRender())
	{
		loadGlMatrix(m);

		glPushMatrix();

		glColor3f(1.0f, 0.0f, 0.0f);

		glTranslatef(this->getRoot().X, this->getRoot().Y, this->getRoot().Z);
		glRotatef(this->getTilt().X,1.0f,0.0f,0.0f);
		glRotatef(this->getTilt().Y,0.0f,1.0f,0.0f);
		glRotatef(this->getTilt().Z,0.0f,0.0f,1.0f);

		renderSemiSphere(this->getApertureHalfWidth().X, this->getRadius().X, m_numApt.X, options);
	}
};