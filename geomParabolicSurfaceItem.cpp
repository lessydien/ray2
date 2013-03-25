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

#include "geomParabolicSurfaceItem.h"

using namespace macrosim;

ParabolicSurfaceItem::ParabolicSurfaceItem(QString name, QObject *parent) :
	GeometryItem(name, PARABOLICSURFACE, parent),
	m_radius(0)
{

};

ParabolicSurfaceItem::~ParabolicSurfaceItem()
{
	m_childs.clear();
};

bool ParabolicSurfaceItem::writeToXML(QDomDocument &document, QDomElement &root) const
{
	QDomElement node = document.createElement("geometry");

	// call base class method
	if (!GeometryItem::writeToXML(document, node))
		return false;

	node.setAttribute("geomType", "PARABOLICSURFACE");
	node.setAttribute("nrSurfacesSeq", "1");
	node.setAttribute("nrSurfacesNonSeq", "1");
	node.setAttribute("radius", QString::number(m_radius));

	root.appendChild(node);
	return true;
};

bool ParabolicSurfaceItem::readFromXML(const QDomElement &node)
{
	// read base class from XML
	if (!GeometryItem::readFromXML(node))
		return false;
	m_radius=node.attribute("radius").toDouble();
	return true;
};

void ParabolicSurfaceItem::render(QMatrix4x4 &m, RenderOptions &options)
{
	if (this->getRender())
	{
		loadGlMatrix(m);

		glPushMatrix();

		if (this->m_focus)
			glColor3f(0.0f,1.0f,0.0f); //green
		else
			glColor3f(0.0f,0.0f,1.0f); //blue

		glTranslatef(this->getRoot().X,this->getRoot().Y,this->getRoot().Z);
		glRotatef(this->getTilt().X,1.0f,0.0f,0.0f);
		glRotatef(this->getTilt().Y,0.0f,1.0f,0.0f);
		glRotatef(this->getTilt().Z,0.0f,0.0f,1.0f);

		if (this->getApertureType()==RECTANGULAR)
		{
			// ?????????????
		}
		else
		{
			renderSemiSphere(this->getApertureRadius().X, this->getRadius(), 1, options);
		}

		glPopMatrix();
	}
};

Vec3f ParabolicSurfaceItem::calcNormal(Vec3f vertex, Vec3f* neighbours, int nr)
{
	// calc centre of sphere
	Vec3f orientation=Vec3f(0,0,1);
	Vec3f centre=Vec3f(0,0,0)+orientation*this->getRadius();
	Vec3f normal=vertex-centre;
	// normalize
	normal=normal/(sqrt(normal*normal));
	if (this->getRadius() <0)
		normal=normal*-1;
	return normal;
};