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

#include "geomApertureStopItem.h"
#include <math.h>

using namespace macrosim;

ApertureStopItem::ApertureStopItem(QString name, QObject *parent) :
	GeometryItem(name, APERTURESTOP, parent),
	m_apertureStopRadius(0,0)
{

}

ApertureStopItem::~ApertureStopItem()
{
	m_childs.clear();
}

bool ApertureStopItem::writeToXML(QDomDocument &document, QDomElement &root) const
{
	QDomElement node = document.createElement("geometry");

	// call base class method
	if (!GeometryItem::writeToXML(document, node))
		return false;

	node.setAttribute("geomType", "APERTURESTOP");
	node.setAttribute("nrSurfacesSeq", "1");
	node.setAttribute("nrSurfacesNonSeq", "1");
	node.setAttribute("apertureStopRadius.x", QString::number(m_apertureStopRadius.X));
	node.setAttribute("apertureStopRadius.y", QString::number(m_apertureStopRadius.Y));

	root.appendChild(node);
	return true;
}

bool ApertureStopItem::readFromXML(const QDomElement &node)
{
	// read base class from XML
	if (!GeometryItem::readFromXML(node))
		return false;
	m_apertureStopRadius.X=node.attribute("apertureStopRadius.x").toDouble();
	m_apertureStopRadius.Y=node.attribute("apertureStopRadius.y").toDouble();
	return true;
}

void ApertureStopItem::render(QMatrix4x4 &m, RenderOptions &options)
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
		glTranslatef(root.X,root.Y,root.Z);
		glRotatef(this->getTilt().X,1.0f,0.0f,0.0f);
		glRotatef(this->getTilt().Y,0.0f,1.0f,0.0f);
		glRotatef(this->getTilt().Z,0.0f,0.0f,1.0f);

		Vec2d ar=this->getApertureRadius();
		Vec2d asr=this->getApertureStopRadius();

		Vec3f neighbours[8];
		Vec3f normal=calcNormal(Vec3f(root.X,root.Y,root.Z),&neighbours[0],0);
	
		if ( (asr.X<ar.X) && (asr.Y<ar.Y))
		{

			if (this->getApertureType()==RECTANGULAR)
			{
				glBegin(GL_QUAD_STRIP);
				glNormal3f(-normal.X, -normal.Y, -normal.Z); // this normal holds to all vertices
				glVertex3f(-ar.X, ar.Y, 0);
				glVertex3f(-asr.X, asr.Y, 0);
				glVertex3f(+ar.X, ar.Y, 0);
				glVertex3f(+asr.X, asr.Y, 0);
			
				glVertex3f(ar.X, -ar.Y, 0);
				glVertex3f(asr.X, -asr.Y, 0);

				glVertex3f(-ar.X, -ar.Y,0);
				glVertex3f(-asr.X, -asr.Y, 0);
			
				glVertex3f(-ar.X, ar.Y,0);
				glVertex3f(-asr.X, asr.Y,0);
			
			

				glEnd();
			}
			else
			{
				double deltaU=2*PI/options.m_slicesWidth;
				double deltaV=2*PI/options.m_slicesWidth;

				glBegin(GL_TRIANGLE_STRIP);
				glNormal3f(normal.X, normal.Y, normal.Z); // this normal holds to all vertices
				for (int i=0; i<options.m_slicesWidth;i++)
				//for (int i=0; i<2;i++)
				{
					glVertex3f(asr.X*cos(-i*deltaU), asr.Y*sin(-i*deltaU), 0);
					glVertex3f(ar.X*cos(-i*deltaU), ar.Y*sin(-i*deltaU), 0);
					glVertex3f(asr.X*cos(-(i+1)*deltaU), asr.Y*sin(-(i+1)*deltaU), 0);
					glVertex3f(ar.X*cos(-(i+1)*deltaU), ar.Y*sin(-(i+1)*deltaU), 0);
				}
				glEnd();
			}
		}

		glPopMatrix();
	}
}

Vec3f ApertureStopItem::calcNormal(Vec3f vertex, Vec3f* neighbours, int nr)
{
	return Vec3f(0,0,1);
}