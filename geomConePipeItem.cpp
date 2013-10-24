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

#include "geomConePipeItem.h"
//#include "glut.h"

#include <vtkVertex.h>
#include <vtkCellArray.h>
#include <vtkProperty.h>
#include <vtkDoubleArray.h>
#include <vtkPointData.h>

using namespace macrosim;

ConePipeItem::ConePipeItem(QString name, QObject *parent) :
	GeometryItem(name, CONEPIPE, parent),
	m_apertureRadius2(0,0),
	m_thickness(0)
{
	// Create a polydata to store everything in
	m_pPolydata = vtkSmartPointer<vtkPolyData>::New();

	// Setup actor and mapper
	vtkSmartPointer<vtkPolyDataMapper> m_pMapper =	vtkSmartPointer<vtkPolyDataMapper>::New();

#if VTK_MAJOR_VERSION <= 5
	m_pMapper->SetInput(m_pPolydata);
#else
	m_pMapper->SetInputData(m_pPolydata);
#endif
	m_pActor->SetMapper(m_pMapper);
}

ConePipeItem::~ConePipeItem()
{
	m_childs.clear();
}

bool ConePipeItem::writeToXML(QDomDocument &document, QDomElement &root) const
{
	QDomElement node = document.createElement("geometry");

	// call base class method
	if (!GeometryItem::writeToXML(document, node))
		return false;

	node.setAttribute("geomType", "CONEPIPE");
	node.setAttribute("nrSurfacesSeq", "1");
	node.setAttribute("nrSurfacesNonSeq", "1");
	node.setAttribute("apertureRadius2.x", QString::number(m_apertureRadius2.X));
	node.setAttribute("apertureRadius2.y", QString::number(m_apertureRadius2.Y));
	node.setAttribute("thickness", QString::number(m_thickness));

	root.appendChild(node);
	return true;
}

bool ConePipeItem::readFromXML(const QDomElement &node)
{
	// read base class from XML
	if (!GeometryItem::readFromXML(node))
		return false;
	m_apertureRadius2.X=node.attribute("apertureRadius2.x").toDouble();
	m_apertureRadius2.Y=node.attribute("apertureRadius2.y").toDouble();
	m_thickness=node.attribute("thickness").toDouble();
	return true;
}

void ConePipeItem::render(QMatrix4x4 &m, RenderOptions &options)
{
	//if (this->getRender())
	//{
	//	// apply current global transformations
	//	loadGlMatrix(m);

	//	glPushMatrix();

	//	if (this->m_focus)
	//		glColor3f(0.0f,1.0f,0.0f); //green
	//	else
	//		glColor3f(0.0f,0.0f,1.0f); //blue

	//	// apply current global transform
	//	glTranslatef(this->getRoot().X,this->getRoot().Y,this->getRoot().Z);
	//	glRotatef(this->getTilt().X,1.0f,0.0f,0.0f);
	//	glRotatef(this->getTilt().Y,0.0f,1.0f,0.0f);
	//	glRotatef(this->getTilt().Z,0.0f,0.0f,1.0f);

	//	double deltaU=2*PI/(options.m_slicesWidth);
	//	double deltaV=this->getThickness();
	//	Vec2d r1=this->getApertureRadius();
	//	Vec2d r2=this->getApertureRadius2();


	//	if ( (r2.X>0) && (r2.Y>0) )
	//	{
	//		Vec3f neighbours[8];

	//		glBegin(GL_TRIANGLE_STRIP);
	//		float x, y, z;
	//		x=r1.X*cos(0*deltaU);
	//		y=r1.Y*sin(0*deltaU);
	//		z=0;
	//		Vec3f normal=calcNormal(Vec3f(x,y,z), &neighbours[0], 0);
	//		glNormal3f(normal.X, normal.Y, normal.Z);
	//		glVertex3f(x, y,z);

	//		x=r2.X*cos(0*deltaU);
	//		y=r2.Y*sin(0*deltaU);
	//		z=deltaV;
	//		normal=calcNormal(Vec3f(x,y,z), &neighbours[0], 0);
	//		glNormal3f(normal.X, normal.Y, normal.Z);
	//		glVertex3f(x, y, z);

	//		x=r1.X*cos((0+1)*deltaU);
	//		y=r1.Y*sin((0+1)*deltaU);
	//		z=0;
	//		normal=calcNormal(Vec3f(x,y,z), &neighbours[0], 0);
	//		glNormal3f(normal.X, normal.Y, normal.Z);
	//		glVertex3f(x, y, z);

	//		x=r2.X*cos((0+1)*deltaU);
	//		y=r2.Y*sin((0+1)*deltaU);
	//		z=deltaV;
	//		normal=calcNormal(Vec3f(x,y,z), &neighbours[0], 0);
	//		glNormal3f(normal.X, normal.Y, normal.Z);
	//		glVertex3f(x, y, z);

	//		for (int iu=1; iu<options.m_slicesWidth; iu++)
	//		{
	//			x=r1.X*cos((iu+1)*deltaU);
	//			y=r1.X*sin((iu+1)*deltaU);
	//			z=0;
	//			normal=calcNormal(Vec3f(x,y,z), &neighbours[0], 0);
	//			glNormal3f(normal.X, normal.Y, normal.Z);
	//			glVertex3f(x, y, z);

	//			x=r2.Y*cos((iu+1)*deltaU);
	//			y=r2.Y*sin((iu+1)*deltaU);
	//			z=deltaV;
	//			normal=calcNormal(Vec3f(x,y,z), &neighbours[0], 0);
	//			glNormal3f(normal.X, normal.Y, normal.Z);
	//			glVertex3f(x, y, z);
	//		}
	//		glEnd();
	//	}

	//	glPopMatrix();
	//}
}

Vec3f ConePipeItem::calcNormal(Vec3f vertex, Vec3f* neighbours, int nr)
{
	// tangens of opening angle of cone
	double tanTheta=(m_apertureRadius2.X-this->getApertureRadius().X)/this->getThickness();
	// distance of vertx to middle axis
	double r=sqrt(vertex.X*vertex.X+vertex.Y*vertex.Y);
	Vec3f normal=vertex-Vec3f(0, 0, vertex.Z+tanTheta*r);
	return normal/(sqrt(normal*normal));
}

Vec3f ConePipeItem::calcNormal(Vec3f vertex)
{
	// tangens of opening angle of cone
	double tanTheta=(m_apertureRadius2.X-this->getApertureRadius().X)/this->getThickness();
	// distance of vertx to middle axis
	double r=sqrt(vertex.X*vertex.X+vertex.Y*vertex.Y);
	Vec3f normal=vertex-Vec3f(0, 0, vertex.Z+tanTheta*r);
	return normal/(sqrt(normal*normal));
}


void ConePipeItem::renderVtk(vtkSmartPointer<vtkRenderer> renderer)
{
	renderer->AddActor(m_pActor);

	this->updateVtk();
}

void ConePipeItem::updateVtk()
{
	vtkSmartPointer<vtkPoints> points =  vtkSmartPointer<vtkPoints>::New();
	vtkSmartPointer<vtkDoubleArray> pointNormalsArray =  vtkSmartPointer<vtkDoubleArray>::New();
	pointNormalsArray->SetNumberOfComponents(3); //3d normals (ie x,y,z)

	vtkSmartPointer<vtkVertex> vertex = vtkSmartPointer<vtkVertex>::New();
	// Create a cell array to store the vertices
	vtkSmartPointer<vtkCellArray> cells =  vtkSmartPointer<vtkCellArray>::New();

	double deltaU=2*PI/(m_renderOptions.m_slicesWidth);
	double deltaV=this->getThickness();
	Vec2d r1=this->getApertureRadius();
	Vec2d r2=this->getApertureRadius2();

	// calc number of vertices 
	unsigned long numVert=4+2*(m_renderOptions.m_slicesWidth-1);

	pointNormalsArray->SetNumberOfTuples(numVert);
	vertex->GetPointIds()->SetNumberOfIds(numVert);

	vtkIdType pid;
	unsigned long vertexIndex=0;

	float x, y, z;
	x=r2.X*cos(0*deltaU);
	y=r2.Y*sin(0*deltaU);
	z=deltaV;
	Vec3f normal=calcNormal(Vec3f(x,y,z));
	pid=points->InsertNextPoint(x,y,z);
	pointNormalsArray->SetTuple(pid, &normal.X);
	vertex->GetPointIds()->SetId(vertexIndex,vertexIndex);
	vertexIndex++;

	x=r1.X*cos(0*deltaU);
	y=r1.Y*sin(0*deltaU);
	z=0;
	normal=calcNormal(Vec3f(x,y,z));
	pid=points->InsertNextPoint(x,y,z);
	pointNormalsArray->SetTuple(pid, &normal.X);
	vertex->GetPointIds()->SetId(vertexIndex,vertexIndex);
	vertexIndex++;

	x=r2.X*cos((0+1)*deltaU);
	y=r2.Y*sin((0+1)*deltaU);
	z=deltaV;
	normal=calcNormal(Vec3f(x,y,z));
	pid=points->InsertNextPoint(x,y,z);
	pointNormalsArray->SetTuple(pid, &normal.X);
	vertex->GetPointIds()->SetId(vertexIndex,vertexIndex);
	vertexIndex++;

	x=r1.X*cos((0+1)*deltaU);
	y=r1.Y*sin((0+1)*deltaU);
	z=0;
	normal=calcNormal(Vec3f(x,y,z));
	pid=points->InsertNextPoint(x,y,z);
	pointNormalsArray->SetTuple(pid, &normal.X);
	vertex->GetPointIds()->SetId(vertexIndex,vertexIndex);
	vertexIndex++;

	for (int iu=1; iu<m_renderOptions.m_slicesWidth; iu++)
	{
		x=r2.X*cos((iu+1)*deltaU);
		y=r2.Y*sin((iu+1)*deltaU);
		z=deltaV;
		normal=calcNormal(Vec3f(x,y,z));
		pid=points->InsertNextPoint(x,y,z);
		pointNormalsArray->SetTuple(pid, &normal.X);
		vertex->GetPointIds()->SetId(vertexIndex,vertexIndex);
		vertexIndex++;

		x=r1.X*cos((iu+1)*deltaU);
		y=r1.Y*sin((iu+1)*deltaU);
		z=0;
		normal=calcNormal(Vec3f(x,y,z));
		pid=points->InsertNextPoint(x,y,z);
		pointNormalsArray->SetTuple(pid, &normal.X);
		vertex->GetPointIds()->SetId(vertexIndex,vertexIndex);
		vertexIndex++;
	}
	cells->InsertNextCell(vertex);
	// store everything in polydata
	m_pPolydata->SetPoints(points);
	// Add the normals to the points in the polydata
	m_pPolydata->GetPointData()->SetNormals(pointNormalsArray);
	m_pPolydata->SetStrips(cells);

	if (this->getRender())
		m_pActor->SetVisibility(1);
	else
		m_pActor->SetVisibility(0);

	// apply root and tilt
	//m_pActor->SetOrigin(this->getRoot().X, this->getRoot().Y, this->getRoot().Z);
	m_pActor->SetPosition(this->getRoot().X, this->getRoot().Y, this->getRoot().Z);
	m_pActor->SetOrientation(this->getTilt().X, this->getTilt().Y, this->getTilt().Z);

	// set lighting properties
	m_pActor->GetProperty()->SetAmbient(m_renderOptions.m_ambientInt);
	m_pActor->GetProperty()->SetDiffuse(m_renderOptions.m_diffuseInt);
	m_pActor->GetProperty()->SetSpecular(m_renderOptions.m_specularInt);

	// Set shading
	m_pActor->GetProperty()->SetInterpolationToGouraud();

	if (this->m_focus)
		m_pActor->GetProperty()->SetColor(0.0,1.0,0.0); // green
	else
		m_pActor->GetProperty()->SetColor(0.0,0.0,1.0); // red

	// request the update
	m_pPolydata->Update();

}