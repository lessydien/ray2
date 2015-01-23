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

#include "detectorVolumeIntensityItem.h"
#include "detectorItemLib.h"

using namespace macrosim;

DetectorVolumeIntensityItem::DetectorVolumeIntensityItem(QString name, QObject *parent) :
	DetectorItem(name, VOLUMEINTENSITY, parent),
		m_ignoreDepth(-1),
		m_detPixel(1,1,1),
		m_apertureHalfWidth(2,2,2)
{
	this->m_render=false;
}

DetectorVolumeIntensityItem::~DetectorVolumeIntensityItem()
{
	m_childs.clear();
}


bool DetectorVolumeIntensityItem::writeToXML(QDomDocument &document, QDomElement &root) const 
{
	QDomElement node = document.createElement("detector");

	if (!DetectorItem::writeToXML(document, node))
		return false;

	node.setAttribute("detPixel.x", QString::number(m_detPixel.X));
	node.setAttribute("detPixel.y", QString::number(m_detPixel.Y));
	node.setAttribute("detPixel.z", QString::number(m_detPixel.Z));
	node.setAttribute("apertureHalfWidth.x", QString::number(m_apertureHalfWidth.X));
	node.setAttribute("apertureHalfWidth.y", QString::number(m_apertureHalfWidth.Y));
	node.setAttribute("apertureHalfWidth.z", QString::number(m_apertureHalfWidth.Z));
	node.setAttribute("ignoreDepth", QString::number(m_ignoreDepth));

	root.appendChild(node);
	return true;
}

bool DetectorVolumeIntensityItem::readFromXML(const QDomElement &node)
{
	// read base class
	if (!DetectorItem::readFromXML(node))
		return false;

	m_detPixel.X=node.attribute("detPixel.x").toDouble();
	m_detPixel.Y=node.attribute("detPixel.y").toDouble();
	m_detPixel.Z=node.attribute("detPixel.z").toDouble();
	m_apertureHalfWidth.X=node.attribute("apertureHalfWidth.x").toDouble();
	m_apertureHalfWidth.Y=node.attribute("apertureHalfWidth.y").toDouble();
	m_apertureHalfWidth.Z=node.attribute("apertureHalfWidth.z").toDouble();

	m_ignoreDepth=node.attribute("ignoreDepth").toInt();

	return true;
}

void DetectorVolumeIntensityItem::render(QMatrix4x4 &m, RenderOptions &options)
{
	//if (this->getRender() )//&& this->getResultFieldPtr())
	//{
	//	// apply current global transformations
	//	loadGlMatrix(m);

	//	glPushMatrix();

	//	// apply current global transform
	//	Vec3d root=this->getRoot();
	//	glTranslatef(root.X,root.Y,root.Z);
	//	glRotatef(this->getTilt().X,1.0f,0.0f,0.0f);
	//	glRotatef(this->getTilt().Y,0.0f,1.0f,0.0f);
	//	glRotatef(this->getTilt().Z,0.0f,0.0f,1.0f);

	//	renderIntensityField(this->getResultField(), options);

	//	glPopMatrix();
	//}

	////	// create vertices of our field
	////	int sizeX=256;//(this->getResultFieldPtr())->getSize()[0];
	////	int sizeY=256;//(this->getResultFieldPtr())->getSize()[1];
	////	// we do not render line fields
	////	if ( (sizeX>1) && (sizeY>1) )
	////	{
	////		unsigned long nrOfQuads=(sizeX-1)*(sizeY-1);
	////		unsigned long nrOfIndices=4*nrOfQuads;

	////		GLfloat *vertices=(GLfloat*)malloc(sizeX*sizeY*3*sizeof(GLfloat));
	////		GLfloat *colors=(GLfloat*)malloc(sizeX*sizeY*3*sizeof(GLfloat));
	////		GLfloat *normals=(GLfloat*)malloc(sizeX*sizeY*3*sizeof(GLfloat));
	////		GLuint *indices=(GLuint*)malloc(nrOfIndices*sizeof(GLuint));

	////		float dx=30;//float(this->getResultFieldPtr()->getAxisScales(0));
	////		float dy=30;//float(this->getResultFieldPtr()->getAxisScales(1));

	////		float x0=0;//float(this->getResultFieldPtr()->getAxisOffset(0));
	////		float y0=0;//float(this->getResultFieldPtr()->getAxisOffset(1));
	////	
	////		// apply current global transformations
	////		loadGlMatrix(m);

	////		glPushMatrix();

	////		// apply current global transform
	////		Vec3d root=this->getRoot();
	////		glTranslatef(root.X,root.Y,root.Z);
	////		glRotatef(this->getTilt().X,1.0f,0.0f,0.0f);
	////		glRotatef(this->getTilt().Y,0.0f,1.0f,0.0f);
	////		glRotatef(this->getTilt().Z,0.0f,0.0f,1.0f);

	////		for (int iy=0; iy<sizeY; iy++)
	////		{
	////			for (int ix=0; ix<sizeX; ix++)
	////			{
	////				//..vertices...
	////				// x-coordinate
	////				vertices[3*ix+iy*sizeX*3]=x0+ix*dx;
	////				// y-coordinate
	////				vertices[3*ix+iy*sizeX*3+1]=y0+iy*dy;
	////				// z-coordinate
	////				vertices[3*ix+iy*sizeX*3+2]=0;
	////				//..normals...
	////				normals[3*ix+iy*sizeX*3]=0.0f;
	////				normals[3*ix+iy*sizeX*3+1]=0.0f;
	////				normals[3*ix+iy*sizeX*3+2]=-1.0f;
	////				//..colors...
	////				colors[3*ix+iy*sizeX*3]=1.0f;
	////				colors[3*ix+iy*sizeX*3+1]=0.0f;
	////				colors[3*ix+iy*sizeX*3+2]=0.0f;
	////			}
	////		}

	////		// create indices
	////		unsigned int yIdx=0;
	////		unsigned int xIdx=0;
	////		indices[0]=0;
	////		for (unsigned long i=1; i<nrOfIndices; i++)
	////		{
	////			unsigned int iQuad=i%4;
	////			if (iQuad<2)
	////				indices[i]=xIdx+iQuad+sizeX*yIdx;
	////			else
	////				indices[i]=xIdx+sizeX*(yIdx+1)-iQuad+3;
	////			if (iQuad==3)
	////				xIdx++;
	////			if ( ((i+1)%((sizeX-1)*4))==0 )
	////			{
	////				xIdx=0;
	////				yIdx++;
	////			}
	////		}

	//////		GLfloat colors[][3]={{1.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {1.0, 0.0, 0.0}};
	//////		GLfloat normals[][3]={{0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, 1.0f}};

	////		glEnableClientState(GL_COLOR_ARRAY);
	////		glEnableClientState(GL_VERTEX_ARRAY);
	////		glEnableClientState(GL_NORMAL_ARRAY);

	////		glVertexPointer(3, GL_FLOAT, 0, vertices);
	////		glColorPointer(3, GL_FLOAT, 0, colors);
	////		glNormalPointer(GL_FLOAT, 0, normals);

	////		glDrawElements(GL_QUADS, nrOfIndices, GL_UNSIGNED_INT, indices);

	////		delete vertices;
	////		delete colors;
	////		delete normals;
	////		delete indices;

	////		glPopMatrix();
	////	}
}