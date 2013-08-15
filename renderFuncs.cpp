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

#include "renderFuncs.h"
#include "DataObject/dataObjectFuncs.h"
#include <GL/glew.h>

//using namespace macrosim;

void loadGlMatrix(const QMatrix4x4& m)
 {
     // static to prevent glLoadMatrixf to fail on certain drivers
     static GLfloat mat[16];
     const qreal *data = m.constData();
     for (int index = 0; index < 16; ++index)
         mat[index] = data[index];
     glLoadMatrixf(mat);
 }

void renderIntensityField(ito::DataObject &field, RenderOptions &options)
{
	// create vertices of our field
	if (field.getSize())
	{
		int sizeX=field.getSize()[0];
		int sizeY=field.getSize()[1];

		// we do not render line fields
		if ( (sizeX>1) && (sizeY>1) )
		{
			unsigned long nrOfQuads=(sizeX-1)*(sizeY-1);
			unsigned long nrOfIndices=4*nrOfQuads;

			GLfloat *vertices=(GLfloat*)malloc(sizeX*sizeY*3*sizeof(GLfloat));
			GLfloat *colors=(GLfloat*)malloc(sizeX*sizeY*3*sizeof(GLfloat));
			GLfloat *normals=(GLfloat*)malloc(sizeX*sizeY*3*sizeof(GLfloat));
			GLuint *indices=(GLuint*)malloc(nrOfIndices*sizeof(GLuint));

			float dx=field.getAxisScale(0);
			float dy=field.getAxisScale(1);

			float x0=field.getAxisOffset(0)-(sizeX)/2*dx+dx/2;
			float y0=field.getAxisOffset(1)-(sizeY)/2*dy+dy/2;

			double maxVal, minVal;
			unsigned int* locationMin=(unsigned int*)malloc(3*sizeof(unsigned int));
			unsigned int* locationMax=(unsigned int*)malloc(3*sizeof(unsigned int));
			ito::dObjHelper::minMaxValue(&field, minVal, locationMin, maxVal, locationMax);
		
			for (unsigned int iy=0; iy<sizeY; iy++)
			{
				double* linePtr=(double*)field.rowPtr(0,iy);
				for (unsigned int ix=0; ix<sizeX; ix++)
				{
					//..vertices...
					// x-coordinate
					vertices[3*ix+iy*sizeX*3]=x0+ix*dx;
					// y-coordinate
					vertices[3*ix+iy*sizeX*3+1]=y0+iy*dy;
					// z-coordinate
					vertices[3*ix+iy*sizeX*3+2]=0;
					//..normals...
					normals[3*ix+iy*sizeX*3]=0.0f;
					normals[3*ix+iy*sizeX*3+1]=0.0f;
					normals[3*ix+iy*sizeX*3+2]=-1.0f;
					//..colors...
					double value=linePtr[ix];
					colors[3*ix+iy*sizeX*3]=GLfloat((value-minVal)/(maxVal-minVal));
					colors[3*ix+iy*sizeX*3+1]=0.0f;
					colors[3*ix+iy*sizeX*3+2]=0.0f;
				}
			}

			// create indices
			unsigned int yIdx=0;
			unsigned int xIdx=0;
			indices[0]=0;
			for (unsigned long i=1; i<nrOfIndices; i++)
			{
				unsigned int iQuad=i%4;
				if (iQuad<2)
					indices[i]=xIdx+iQuad+sizeX*yIdx;
				else
					indices[i]=xIdx+sizeX*(yIdx+1)-iQuad+3;
				if (iQuad==3)
					xIdx++;
				if ( ((i+1)%((sizeX-1)*4))==0 )
				{
					xIdx=0;
					yIdx++;
				}
			}

	//		GLfloat colors[][3]={{1.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {1.0, 0.0, 0.0}};
	//		GLfloat normals[][3]={{0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, 1.0f}};

			glEnableClientState(GL_COLOR_ARRAY);
			glEnableClientState(GL_VERTEX_ARRAY);
			glEnableClientState(GL_NORMAL_ARRAY);

			glVertexPointer(3, GL_FLOAT, 0, vertices);
			glColorPointer(3, GL_FLOAT, 0, colors);
			glNormalPointer(GL_FLOAT, 0, normals);

			glDrawElements(GL_QUADS, nrOfIndices, GL_UNSIGNED_INT, indices);

			delete vertices;
			delete colors;
			delete normals;
			delete indices;

//			glPopMatrix();
		}
	}
};

void renderSemiSphereVtk(double aptRadius, double radius, double numApt, RenderOptions &options)
{
}

void renderSemiSphere(double aptRadius, double radius, double numApt, RenderOptions &options)
{
	double ar=aptRadius;
	double r=-radius;

	double deltaU=2*PI/(options.m_slicesWidth);
	double deltaV;

	if (ar>=abs(r) && numApt>=1)
		// if aperture is bigger than radius, we need to draw the full semi-sphere
		deltaV=PI/2/options.m_slicesHeight;
	else
	{
		if (ar < abs(r))
		{
			// if not, we only draw part of the hemi-sphere
			deltaV=std::min(asin(ar/r), std::min(asin(numApt), PI/2))/options.m_slicesHeight;
		}
		else
			deltaV=std::min(asin(numApt), PI/2)/options.m_slicesHeight;
	}

	//for (float phi=0; phi <= PI/2; phi+=factor)
	for (int iv=0; iv<options.m_slicesHeight; iv++)
	{
		double phi=0+iv*deltaV;
		glBegin(GL_TRIANGLE_STRIP);

		float x=r*sin(phi)*cos(0.0f);
		float y=r*sin(phi)*sin(0.0f);
		float z=r*cos(phi)-r;
		// set vertex and normal
		Vec3f normal=calcSemiSphereNormal(Vec3f(x,y,z), radius);
		glNormal3f(normal.X, normal.Y, normal.Z);
		glVertex3f(x,y,z);
		//glVertex3f(x,y,z);

		x=r*sin(phi+deltaV)*cos(0.0f);
		y=r*sin(phi+deltaV)*sin(0.0f);
		z=r*cos(phi+deltaV)-r;
		// set vertex and normal
		normal=calcSemiSphereNormal(Vec3f(x,y,z), radius);
		glNormal3f(normal.X, normal.Y, normal.Z);
		glVertex3f(x,y,z);

		x=r*sin(phi)*cos(deltaU);
		y=r*sin(phi)*sin(deltaU);
		z=r*cos(phi)-r;
		// set vertex and normal
		normal=calcSemiSphereNormal(Vec3f(x,y,z), radius);
		glNormal3f(normal.X, normal.Y, normal.Z);
		glVertex3f(x,y,z);

		x=r*sin(phi+deltaV)*cos(deltaU);
		y=r*sin(phi+deltaV)*sin(deltaU);
		z=r*cos(phi+deltaV)-r;
		// set vertex and normal
		normal=calcSemiSphereNormal(Vec3f(x,y,z), radius);
		glNormal3f(normal.X, normal.Y, normal.Z);
		glVertex3f(x,y,z);

		//for (float theta=factor; theta <= 2*PI; theta+= factor)
		for (int iu=1; iu<=options.m_slicesWidth; iu++)
		{
			double theta=0+iu*deltaU;
			x=r*sin(phi)*cos(theta);
			y=r*sin(phi)*sin(theta);
			z=r*cos(phi)-r;
			// set vertex and normal
			normal=calcSemiSphereNormal(Vec3f(x,y,z), radius);
			glNormal3f(normal.X, normal.Y, normal.Z);
			glVertex3f(x, y, z);

			x=r*sin(phi+deltaV)*cos(theta);
			y=r*sin(phi+deltaV)*sin(theta);
			z=r*cos(phi+deltaV)-r;
			// set vertex and normal
			normal=calcSemiSphereNormal(Vec3f(x,y,z), radius);
			glNormal3f(normal.X, normal.Y, normal.Z);
			glVertex3f(x, y, z);
		}


		glEnd();
	}
};

Vec3f calcSemiSphereNormal(Vec3f vertex, double radius)
{
	Vec3f normal=vertex-Vec3f(0,0,radius);
	normal=normal/(sqrt(normal*normal));
	if (radius < 0)
		normal=normal*-1;
	return normal;
}

void rotateVec(Vec3d *vec, Vec3d tilt)
{
	Mat3x3d Mx=Mat3x3d(1,0,0, 0,cos(tilt.X),-sin(tilt.X), 0,sin(tilt.X),cos(tilt.X));
	Mat3x3d My=Mat3x3d(cos(tilt.Y),0,sin(tilt.Y), 0,1,0, -sin(tilt.Y),0,cos(tilt.Y));
	Mat3x3d Mz=Mat3x3d(cos(tilt.Z),-sin(tilt.Z),0, sin(tilt.Z),cos(tilt.Z),0, 0,0,1);
	Mat3x3d Mxy=Mx*My;
	Mat3x3d M=Mxy*Mz;
	Vec3d tmpVec=M*(*vec);
	*vec=tmpVec;
}

RenderMode stringToRenderMode(QString str)
{
	if (str.isNull())
		return RENDER_SOLID;
	if (!str.compare("Wireframe"))
		return RENDER_WIREGRID;
	if (!str.compare("Transparent"))
		return RENDER_TRANSPARENCY;

	return RENDER_SOLID;
};

int renderModeToComboBoxIndex(RenderMode in)
{
	switch (in)
	{
	case RENDER_SOLID:
		return 0;
		break;
	case RENDER_WIREGRID:
		return 1;
		break;
	case RENDER_TRANSPARENCY:
		return 2;
		break;
	default:
		return 0;
		break;
	}
}