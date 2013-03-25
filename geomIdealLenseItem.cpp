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

#include "geomIdealLenseItem.h"

using namespace macrosim;

IdealLenseItem::IdealLenseItem(QString name, QObject *parent) :
	GeometryItem(name, IDEALLENSE, parent)
{

}

IdealLenseItem::~IdealLenseItem()
{
	m_childs.clear();
}

bool IdealLenseItem::writeToXML(QDomDocument &document, QDomElement &root) const
{
	QDomElement node = document.createElement("geometry");

	// call base class method
	if (!GeometryItem::writeToXML(document, node))
		return false;

	node.setAttribute("geomType", "IDEALLENSE");
	node.setAttribute("nrSurfacesSeq", "1");
	node.setAttribute("nrSurfacesNonSeq", "1");

	root.appendChild(node);
	return true;

}

bool IdealLenseItem::readFromXML(const QDomElement &node)
{
	// read base class from XML
	if (!GeometryItem::readFromXML(node))
		return false;
	return true;
}

void IdealLenseItem::render(QMatrix4x4 &m, RenderOptions &options)
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
		Vec3d root=this->getRoot();
		Vec2d aptRadius=this->getApertureRadius();
		glTranslatef(root.X,root.Y,root.Z);
		glRotatef(this->getTilt().X,1.0f,0.0f,0.0f);
		glRotatef(this->getTilt().Y,0.0f,1.0f,0.0f);
		glRotatef(this->getTilt().Z,0.0f,0.0f,1.0f);

		Vec3f neighbours[8];
		Vec3f normal=calcNormal(Vec3f(root.X,root.Y,root.Z),&neighbours[0],0);
	
		if (this->getApertureType()==RECTANGULAR)
		{
			glBegin(GL_QUADS);
			glNormal3f(normal.X, normal.Y, normal.Z); // this normal holds to all vertices
			float x=root.X-aptRadius.X;
			float y=root.Y-aptRadius.Y;
			float z=0;
			glVertex3f(x,y,z);

			x=root.X-aptRadius.X;
			y=root.Y+aptRadius.Y;
			z=0;
			glVertex3f(x,y,z);

			x=root.X+aptRadius.X;
			y=root.Y+aptRadius.Y;
			z=0;
			glVertex3f(x,y,z);

			x=root.X+aptRadius.X;
			y=root.Y-aptRadius.Y;
			z=0;
			glVertex3f(x,y,z);
			glEnd();
		}
		else
		{
			glBegin(GL_TRIANGLE_FAN);
			glNormal3f(normal.X, normal.Y, normal.Z); // this normal holds to all vertices
			float deltaU=2*PI/options.m_slicesWidth;
			double a=this->getApertureRadius().X;
			double b=this->getApertureRadius().Y;
			glVertex3f(root.X, root.Y, root.Z);
			for (int i=0; i<=options.m_slicesWidth; i++)
			{
				glVertex3f(root.X+a*cos(-i*deltaU), root.Y+b*sin(-i*deltaU), 0);
			}
			glEnd();
		}

		glPopMatrix();
	}
}

Vec3f IdealLenseItem::calcNormal(Vec3f vertex, Vec3f* neighbours, int nr)
{
	return Vec3f(0,0,1);
}