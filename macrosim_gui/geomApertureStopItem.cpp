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
//#include "glut.h"
#include <vtkCellArray.h>
#include <vtkPoints.h>
#include <vtkDoubleArray.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkVertex.h>
#include <vtkPointData.h>

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
	//	Vec3d root=this->getRoot();
	//	glTranslatef(root.X,root.Y,root.Z);
	//	glRotatef(this->getTilt().X,1.0f,0.0f,0.0f);
	//	glRotatef(this->getTilt().Y,0.0f,1.0f,0.0f);
	//	glRotatef(this->getTilt().Z,0.0f,0.0f,1.0f);

	//	Vec2d ar=this->getApertureRadius();
	//	Vec2d asr=this->getApertureStopRadius();

	//	Vec3f neighbours[8];
	//	Vec3f normal=calcNormal(Vec3f(root.X,root.Y,root.Z),&neighbours[0],0);
	//
	//	if ( (asr.X<ar.X) && (asr.Y<ar.Y))
	//	{

	//		if (this->getApertureType()==RECTANGULAR)
	//		{
	//			glBegin(GL_QUAD_STRIP);
	//			glNormal3f(-normal.X, -normal.Y, -normal.Z); // this normal holds to all vertices
	//			glVertex3f(-ar.X, ar.Y, 0);
	//			glVertex3f(-asr.X, asr.Y, 0);
	//			glVertex3f(+ar.X, ar.Y, 0);
	//			glVertex3f(+asr.X, asr.Y, 0);
	//		
	//			glVertex3f(ar.X, -ar.Y, 0);
	//			glVertex3f(asr.X, -asr.Y, 0);

	//			glVertex3f(-ar.X, -ar.Y,0);
	//			glVertex3f(-asr.X, -asr.Y, 0);
	//		
	//			glVertex3f(-ar.X, ar.Y,0);
	//			glVertex3f(-asr.X, asr.Y,0);
	//		
	//		

	//			glEnd();
	//		}
	//		else
	//		{
	//			double deltaU=2*PI/options.m_slicesWidth;
	//			double deltaV=2*PI/options.m_slicesWidth;

	//			glBegin(GL_TRIANGLE_STRIP);
	//			glNormal3f(normal.X, normal.Y, normal.Z); // this normal holds to all vertices
	//			for (int i=0; i<options.m_slicesWidth;i++)
	//			//for (int i=0; i<2;i++)
	//			{
	//				glVertex3f(asr.X*cos(-i*deltaU), asr.Y*sin(-i*deltaU), 0);
	//				glVertex3f(ar.X*cos(-i*deltaU), ar.Y*sin(-i*deltaU), 0);
	//				glVertex3f(asr.X*cos(-(i+1)*deltaU), asr.Y*sin(-(i+1)*deltaU), 0);
	//				glVertex3f(ar.X*cos(-(i+1)*deltaU), ar.Y*sin(-(i+1)*deltaU), 0);
	//			}
	//			glEnd();
	//		}
	//	}

	//	glPopMatrix();
	//}
}

Vec3f ApertureStopItem::calcNormal(Vec3f vertex, Vec3f* neighbours, int nr)
{
	return Vec3f(0,0,1);
}

Vec3f ApertureStopItem::calcNormal(Vec3f vertex)
{
	return Vec3f(0,0,-1);
}

void ApertureStopItem::renderVtk(vtkSmartPointer<vtkRenderer> renderer)
{
	// Create a polydata to store everything in
	m_pPolydata = vtkSmartPointer<vtkPolyData>::New();
  
	// Setup actor and mapper
	m_pMapper =	vtkSmartPointer<vtkPolyDataMapper>::New();

#if VTK_MAJOR_VERSION <= 5
	m_pMapper->SetInput(m_pPolydata);
#else
	m_pMapper->SetInputData(m_pPolydata);
#endif

	m_pActor->SetMapper(m_pMapper);

	renderer->AddActor(m_pActor);

	this->updateVtk();
}

void ApertureStopItem::updateVtk()
{
	vtkSmartPointer<vtkPoints> points =  vtkSmartPointer<vtkPoints>::New();
	vtkSmartPointer<vtkDoubleArray> pointNormalsArray =  vtkSmartPointer<vtkDoubleArray>::New();
	pointNormalsArray->SetNumberOfComponents(3); //3d normals (ie x,y,z)

	vtkSmartPointer<vtkVertex> vertex = vtkSmartPointer<vtkVertex>::New();
	// Create a cell array to store the vertices
	vtkSmartPointer<vtkCellArray> cells =  vtkSmartPointer<vtkCellArray>::New();

	vtkIdType pid;

	Vec2d ar=this->getApertureRadius();
	Vec2d asr=this->getApertureStopRadius();

	unsigned long vertexIndex=0;

	if (this->getApertureType() == RECTANGULAR)
	{
		pointNormalsArray->SetNumberOfTuples(11);
		vertex->GetPointIds()->SetNumberOfIds(11);

		double x=-asr.X;
		double y=-asr.Y;
		double z=0;
		pid=points->InsertNextPoint(x,y,z);
		Vec3f normal=calcNormal(Vec3f());
		pointNormalsArray->SetTuple(pid, &normal.X);
		vertex->GetPointIds()->SetId(vertexIndex,vertexIndex);
		vertexIndex++;

		x=-ar.X;
		y=-ar.Y;
		z=0;
		pid=points->InsertNextPoint(x,y,z);
		normal=calcNormal(Vec3f());
		pointNormalsArray->SetTuple(pid, &normal.X);
		vertex->GetPointIds()->SetId(vertexIndex,vertexIndex);
		vertexIndex++;

		x=-asr.X;
		y=+asr.Y;
		z=0;
		pid=points->InsertNextPoint(x,y,z);
		normal=calcNormal(Vec3f());
		pointNormalsArray->SetTuple(pid, &normal.X);
		vertex->GetPointIds()->SetId(vertexIndex,vertexIndex);
		vertexIndex++;

		x=-ar.X;
		y=+ar.Y;
		z=0;
		pid=points->InsertNextPoint(x,y,z);
		normal=calcNormal(Vec3f());
		pointNormalsArray->SetTuple(pid, &normal.X);
		vertex->GetPointIds()->SetId(vertexIndex,vertexIndex);
		vertexIndex++;

		x=-asr.X;
		y=+asr.Y;
		z=0;
		pid=points->InsertNextPoint(x,y,z);
		normal=calcNormal(Vec3f());
		pointNormalsArray->SetTuple(pid, &normal.X);
		vertex->GetPointIds()->SetId(vertexIndex,vertexIndex);
		vertexIndex++;

		x=+ar.X;
		y=+ar.Y;
		z=0;
		pid=points->InsertNextPoint(x,y,z);
		normal=calcNormal(Vec3f());
		pointNormalsArray->SetTuple(pid, &normal.X);
		vertex->GetPointIds()->SetId(vertexIndex,vertexIndex);
		vertexIndex++;

		x=+asr.X;
		y=+asr.Y;
		z=0;
		pid=points->InsertNextPoint(x,y,z);
		normal=calcNormal(Vec3f());
		pointNormalsArray->SetTuple(pid, &normal.X);
		vertex->GetPointIds()->SetId(vertexIndex,vertexIndex);
		vertexIndex++;

		x=+ar.X;
		y=-ar.Y;
		z=0;
		pid=points->InsertNextPoint(x,y,z);
		normal=calcNormal(Vec3f());
		pointNormalsArray->SetTuple(pid, &normal.X);
		vertex->GetPointIds()->SetId(vertexIndex,vertexIndex);
		vertexIndex++;

		x=+asr.X;
		y=-asr.Y;
		z=0;
		pid=points->InsertNextPoint(x,y,z);
		normal=calcNormal(Vec3f());
		pointNormalsArray->SetTuple(pid, &normal.X);
		vertex->GetPointIds()->SetId(vertexIndex,vertexIndex);
		vertexIndex++;

		x=-ar.X;
		y=-ar.Y;
		z=0;
		pid=points->InsertNextPoint(x,y,z);
		normal=calcNormal(Vec3f());
		pointNormalsArray->SetTuple(pid, &normal.X);
		vertex->GetPointIds()->SetId(vertexIndex,vertexIndex);
		vertexIndex++;

		x=-asr.X;
		y=-asr.Y;
		z=0;
		pid=points->InsertNextPoint(x,y,z);
		normal=calcNormal(Vec3f());
		pointNormalsArray->SetTuple(pid, &normal.X);
		vertex->GetPointIds()->SetId(vertexIndex,vertexIndex);
		vertexIndex++;

	}
	else
	{
		unsigned long numVertices=4*(m_renderOptions.m_slicesWidth);

		vtkIdType pid;
		pointNormalsArray->SetNumberOfTuples(numVertices);
		vertex->GetPointIds()->SetNumberOfIds(numVertices);

		unsigned long vertexIndex=0;

		double deltaU=2*PI/m_renderOptions.m_slicesWidth;
		double deltaV=2*PI/m_renderOptions.m_slicesWidth;

		double x=0;
		double y=0;
		double z=0;
		Vec3f normal;

		for (unsigned int i=1; i<=m_renderOptions.m_slicesWidth; i++)
		//for (unsigned int i=1; i<=1; i++)
		{
			x=asr.X*cos(double(-i)*deltaU);
			y=asr.Y*sin(double(-i)*deltaU);
			z=0;
			pid=points->InsertNextPoint(x,y,z);
			normal=calcNormal(Vec3f());
			pointNormalsArray->SetTuple(pid, &normal.X);
			vertex->GetPointIds()->SetId(vertexIndex,vertexIndex);
			vertexIndex++;

			x=ar.X*cos(double(-i)*deltaU);
			y=ar.Y*sin(double(-i)*deltaU);
			z=0;
			pid=points->InsertNextPoint(x,y,z);
			normal=calcNormal(Vec3f());
			pointNormalsArray->SetTuple(pid, &normal.X);
			vertex->GetPointIds()->SetId(vertexIndex,vertexIndex);
			vertexIndex++;

			x=asr.X*cos(double(-(i+1))*deltaU);
			y=asr.Y*sin(double(-(i+1))*deltaU);
			z=0;
			pid=points->InsertNextPoint(x,y,z);
			normal=calcNormal(Vec3f());
			pointNormalsArray->SetTuple(pid, &normal.X);
			vertex->GetPointIds()->SetId(vertexIndex,vertexIndex);
			vertexIndex++;

			x=ar.X*cos(double(-(i+1))*deltaU);
			y=ar.Y*sin(double(-(i+1))*deltaU);
			z=0;
			pid=points->InsertNextPoint(x,y,z);
			normal=calcNormal(Vec3f());
			pointNormalsArray->SetTuple(pid, &normal.X);
			vertex->GetPointIds()->SetId(vertexIndex,vertexIndex);
			vertexIndex++;
		}

	}

	cells->InsertNextCell(vertex);
	// Add the points and quads to the dataset
	m_pPolydata->SetPoints(points);
	m_pPolydata->GetPointData()->SetNormals(pointNormalsArray);
	m_pPolydata->SetStrips(cells);

	// apply root and tilt
	//m_pActor->SetOrigin(this->getRoot().X, this->getRoot().Y, this->getRoot().Z);
	m_pActor->SetPosition(this->getRoot().X, this->getRoot().Y, this->getRoot().Z);
	m_pActor->SetOrientation(this->getTilt().X, this->getTilt().Y, this->getTilt().Z);

	// set lighting properties
	m_pActor->GetProperty()->SetAmbient(m_renderOptions.m_ambientInt);
	m_pActor->GetProperty()->SetDiffuse(m_renderOptions.m_diffuseInt);
	m_pActor->GetProperty()->SetSpecular(m_renderOptions.m_specularInt);

	// set color to red
	if (this->m_focus)
		m_pActor->GetProperty()->SetColor(0.0,1.0,0.0);
	else
		m_pActor->GetProperty()->SetColor(0.0,0.0,1.0);
	
	if (this->getRender())
		m_pActor->SetVisibility(1);
	else
		m_pActor->SetVisibility(0);

#if  (VTK_MAJOR_VERSION <= 5)
	// request the update
	m_pPolydata->Update();
#else
    m_pMapper->Update();
#endif

};