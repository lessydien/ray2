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

#include "geomCylPipeItem.h"

using namespace macrosim;

CylPipeItem::CylPipeItem(QString name, QObject *parent) :
	GeometryItem(name, CYLPIPE, parent),
	m_radius(0),
	m_thickness(0)
{

}

CylPipeItem::~CylPipeItem()
{
	m_childs.clear();
}

bool CylPipeItem::writeToXML(QDomDocument &document, QDomElement &root) const
{
	QDomElement node = document.createElement("geometry");

	// call base class method
	if (!GeometryItem::writeToXML(document, node))
		return false;

	node.setAttribute("geomType", "CYLPIPE");
	node.setAttribute("nrSurfacesSeq", "1");
	node.setAttribute("nrSurfacesNonSeq", "1");
	node.setAttribute("radius", QString::number(m_radius));
	node.setAttribute("thickness", QString::number(m_thickness));

	root.appendChild(node);
	return true;
}

bool CylPipeItem::readFromXML(const QDomElement &node)
{
	// read base class from XML
	if (!GeometryItem::readFromXML(node))
		return false;
	m_radius=node.attribute("radius").toDouble();
	m_thickness=node.attribute("thickness").toDouble();
	return true;
}

void CylPipeItem::render(QMatrix4x4 &m, RenderOptions &options)
{
	if (this->getRender())
	{
		// apply current global transformations
		loadGlMatrix(m);

		glPushMatrix();

		if (this->m_focus)
			glColor3f(0.0f,1.0f,0.0f); //green
		else
			glColor3f(0.0f,0.0f,1.0f); //blue

		// apply current global transform
		glTranslatef(this->getRoot().X,this->getRoot().Y,this->getRoot().Z);
		glRotatef(this->getTilt().X,1.0f,0.0f,0.0f);
		glRotatef(this->getTilt().Y,0.0f,1.0f,0.0f);
		glRotatef(this->getTilt().Z,0.0f,0.0f,1.0f);

		double deltaU=2*PI/(options.m_slicesWidth);
		double deltaV=this->getThickness();
		double r=this->getRadius();

		Vec3f neighbours[8];
	
		glBegin(GL_TRIANGLE_STRIP);
		float x=r*cos(0*deltaU);
		float y=r*sin(0*deltaU);
		float z=0;
		Vec3f normal=calcNormal(Vec3f(x,y,z),&neighbours[0],0);
		glNormal3f(normal.X, normal.Y, normal.Z);
		glVertex3f(x, y ,z);

		x=r*cos(0*deltaU);
		y=r*sin(0*deltaU);
		z=deltaV;
		normal=calcNormal(Vec3f(x,y,z),&neighbours[0],0);
		glNormal3f(normal.X, normal.Y, normal.Z);
		glVertex3f(x, y, z);

		x=r*cos((0+1)*deltaU);
		y=r*sin((0+1)*deltaU);
		z=0;
		normal=calcNormal(Vec3f(x,y,z),&neighbours[0],0);
		glNormal3f(normal.X, normal.Y, normal.Z);
		glVertex3f(x, y, z);

		x=r*cos((0+1)*deltaU);
		y=r*sin((0+1)*deltaU) ;
		z=deltaV;
		normal=calcNormal(Vec3f(x,y,z),&neighbours[0],0);
		glNormal3f(normal.X, normal.Y, normal.Z);
		glVertex3f(x, y, z);

		for (int iu=1; iu<options.m_slicesWidth; iu++)
		{
			x=r*cos((iu+1)*deltaU);
			y=r*sin((iu+1)*deltaU);
			z=0;
			normal=calcNormal(Vec3f(x,y,z),&neighbours[0],0);
			glNormal3f(normal.X, normal.Y, normal.Z);
			glVertex3f(x, y, z);

			x=r*cos((iu+1)*deltaU);
			y=r*sin((iu+1)*deltaU);
			z=deltaV;
			normal=calcNormal(Vec3f(x,y,z),&neighbours[0],0);
			glNormal3f(normal.X, normal.Y, normal.Z);
			glVertex3f(x, y, z);
		}
		glEnd();



		glPopMatrix();
	}

}

Vec3f CylPipeItem::calcNormal(Vec3f vertex, Vec3f* neighbours, int nr)
{
	Vec3f normal=vertex-Vec3f(0, 0, vertex.Z);
	return normal/(sqrt(normal*normal));
}