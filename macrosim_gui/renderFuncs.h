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

#ifndef RENDERFUNCS_H
#define RENDERFUNCS_H

//#include <QtOpenGL\qglfunctions.h>
#include "DataObject/dataobj.h"
#include "QPropertyEditor/CustomTypes.h"
#include "qmatrix4x4.h"

#ifndef PI
	#define PI 3.14159265358979
#endif

//namespace macrosim
//{
typedef enum
{
	RENDER_SOLID,
	RENDER_TRANSPARENCY,
	RENDER_WIREGRID
} RenderMode;

class RenderOptions
{
public:
	RenderOptions() :
	  m_slicesHeight(31),
		  m_slicesWidth(31),
		  m_showCoordAxes(true),
		  m_ambientInt(0.25),
		  m_diffuseInt(0.75),
		  m_specularInt(0.5),
		  m_backgroundColor(Vec3d(0.7, 0.7, 0.7)),
		  m_lightPos(Vec3d(0.0, 0.0, 150.0)),
		  m_renderMode(RENDER_SOLID)
	  {
	  }

	int m_slicesHeight;
	int m_slicesWidth;
	double m_ambientInt;
	double m_diffuseInt;
	double m_specularInt;
	Vec3d m_backgroundColor;
	Vec3d m_lightPos;
	bool m_showCoordAxes;
	RenderMode m_renderMode;
};

RenderMode stringToRenderMode(QString str);
int renderModeToComboBoxIndex(RenderMode in);

void loadGlMatrix(const QMatrix4x4& m);

void renderIntensityField(ito::DataObject &field, RenderOptions &options);
void renderSemiSphere(double aptRadius, double radius, double numApt, RenderOptions &options);
Vec3f calcSemiSphereNormal(Vec3f vertex, double radius);
void rotateVec(Vec3d *vec, Vec3d tilt);

//}; // end namespace

#endif