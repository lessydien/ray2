/*
Copyright (C) 2012 ITO university stuttgart

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef OGLDRAWWIDGET
#define OGLDRAWWIDGET

#include <QGLWidget>
#include "glut.h"
//#include <qwidget.h>

class OGLdrawWidget : public QGLWidget
{
	Q_OBJECT

public:
	OGLdrawWidget() 
		: QGLWidget(QGLFormat(QGL::SampleBuffers)) 
	{
	}

protected:
	void initializeGL()
	{
		glEnable(GL_DEPTH_TEST);
		glEnable(GL_COLOR_MATERIAL);
		glEnable(GL_LIGHTING);
		glEnable(GL_LIGHT0);
		glEnable(GL_NORMALIZE);
		glClearColor(0.0,0.0,1.0,0.5);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	}

	void resizeGL(int w, int h)
	{
		// setup viewport, projection, etc
		glViewport(0, 0, (GLint)w, (GLint)h);

		// add ambient light
		GLfloat ambientColor[] = {0.5f, 0.5f, 0.5f, 1.0f};
		glLightModelfv(GL_LIGHT_MODEL_AMBIENT, ambientColor);

		// add positioned light
		GLfloat lightColor0[] = {0.5f, 0.5f, 0.5f, 1.0f};
		GLfloat lightPos0[] = {-50.0f, -50.0f, 100.0f, 0.0f};
		glLightfv(GL_LIGHT0, GL_DIFFUSE, lightColor0);
		glLightfv(GL_LIGHT0, GL_POSITION, lightPos0);

	}

	void paintGL()
	{
		glColor3f(1.0f,0.0f,0.0f); //red

		glMatrixMode(GL_MODELVIEW);

		GLfloat t_pMatrix[16];
		glGetFloatv(GL_MODELVIEW_MATRIX, &t_pMatrix[0]);

		glPushMatrix();

		glTranslatef(100,100,0);
		glRotatef(0.0f,1.0f,0.0f,0.0f);
		glRotatef(0.0f,0.0f,1.0f,0.0f);
		glRotatef(0.0f,0.0f,0.0f,1.0f);
//		glutSolidSphere(50, 20, 20);

		glPopMatrix();
	}

};

#endif