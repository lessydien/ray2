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
#include "materialIdealLenseItem.h"
//#include "glut.h"

#include <vtkCellArray.h>
#include <vtkPoints.h>
#include <vtkDoubleArray.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkVertex.h>
#include <vtkPointData.h>

using namespace macrosim;

IdealLenseItem::IdealLenseItem(QString name, QObject *parent) :
	GeometryItem(name, IDEALLENSE, parent)
{
	this->m_materialType=MATIDEALLENSE;
	this->setChild(new MaterialIdealLenseItem());

	// Create a polydata to store everything in
	m_pPolydata = vtkSmartPointer<vtkPolyData>::New();
  
	// Setup actor and mapper
	m_pMapper =	vtkSmartPointer<vtkPolyDataMapper>::New();

	m_pActor->SetMapper(m_pMapper);
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
	//	Vec2d aptRadius=this->getApertureRadius();
	//	glTranslatef(root.X,root.Y,root.Z);
	//	glRotatef(this->getTilt().X,1.0f,0.0f,0.0f);
	//	glRotatef(this->getTilt().Y,0.0f,1.0f,0.0f);
	//	glRotatef(this->getTilt().Z,0.0f,0.0f,1.0f);

	//	Vec3f neighbours[8];
	//	Vec3f normal=calcNormal(Vec3f(root.X,root.Y,root.Z),&neighbours[0],0);
	//
	//	if (this->getApertureType()==RECTANGULAR)
	//	{
	//		glBegin(GL_QUADS);
	//		glNormal3f(normal.X, normal.Y, normal.Z); // this normal holds to all vertices
	//		float x=root.X-aptRadius.X;
	//		float y=root.Y-aptRadius.Y;
	//		float z=0;
	//		glVertex3f(x,y,z);

	//		x=root.X-aptRadius.X;
	//		y=root.Y+aptRadius.Y;
	//		z=0;
	//		glVertex3f(x,y,z);

	//		x=root.X+aptRadius.X;
	//		y=root.Y+aptRadius.Y;
	//		z=0;
	//		glVertex3f(x,y,z);

	//		x=root.X+aptRadius.X;
	//		y=root.Y-aptRadius.Y;
	//		z=0;
	//		glVertex3f(x,y,z);
	//		glEnd();
	//	}
	//	else
	//	{
	//		glBegin(GL_TRIANGLE_FAN);
	//		glNormal3f(normal.X, normal.Y, normal.Z); // this normal holds to all vertices
	//		float deltaU=2*PI/options.m_slicesWidth;
	//		double a=this->getApertureRadius().X;
	//		double b=this->getApertureRadius().Y;
	//		glVertex3f(root.X, root.Y, root.Z);
	//		for (int i=0; i<=options.m_slicesWidth; i++)
	//		{
	//			glVertex3f(root.X+a*cos(-i*deltaU), root.Y+b*sin(-i*deltaU), 0);
	//		}
	//		glEnd();
	//	}

	//	glPopMatrix();
	//}
}

Vec3f IdealLenseItem::calcNormal(Vec3f vertex, Vec3f* neighbours, int nr)
{
	return Vec3f(0,0,1);
}

Vec3f IdealLenseItem::calcNormal(Vec3f vertex)
{
	return Vec3f(0,0,-1);
}

void IdealLenseItem::updateVtk()
{
	if (this->getApertureType() == RECTANGULAR)
	{
		vtkSmartPointer<vtkPoints> points =  vtkSmartPointer<vtkPoints>::New();
		vtkSmartPointer<vtkDoubleArray> pointNormalsArray =  vtkSmartPointer<vtkDoubleArray>::New();
		pointNormalsArray->SetNumberOfComponents(3); //3d normals (ie x,y,z)

		vtkSmartPointer<vtkVertex> vertex = vtkSmartPointer<vtkVertex>::New();
		// Create a cell array to store the vertices
		vtkSmartPointer<vtkCellArray> cells =  vtkSmartPointer<vtkCellArray>::New();

		vtkIdType pid;
		pointNormalsArray->SetNumberOfTuples(4);
		vertex->GetPointIds()->SetNumberOfIds(4);
		// Create four points (must be in counter clockwise order)
		float p0[3] = {float(-this->getApertureRadius().X), float(-this->getApertureRadius().Y), 0.0f};
		float p1[3] = {float(-this->getApertureRadius().X), float(this->getApertureRadius().Y), 0.0f};
		float p3[3] = {float(this->getApertureRadius().X), float(this->getApertureRadius().Y), 0.0f};
		float p2[3] = {float(this->getApertureRadius().X), float(-this->getApertureRadius().Y), 0.0f};
 
		// Add the points to a vtkPoints object
		pid=points->InsertNextPoint(p0);
		Vec3f normal=calcNormal(Vec3f());
		pointNormalsArray->SetTuple(pid, &normal.X);
		vertex->GetPointIds()->SetId(0,0);

		pid=points->InsertNextPoint(p1);
		normal=calcNormal(Vec3f());
		pointNormalsArray->SetTuple(pid, &normal.X);
		vertex->GetPointIds()->SetId(1,1);

		pid=points->InsertNextPoint(p2);
		normal=calcNormal(Vec3f());
		pointNormalsArray->SetTuple(pid, &normal.X);
		vertex->GetPointIds()->SetId(2,2);

		pid=points->InsertNextPoint(p3);
		normal=calcNormal(Vec3f());
		pointNormalsArray->SetTuple(pid, &normal.X);
		vertex->GetPointIds()->SetId(3,3);

		cells->InsertNextCell(vertex);
		// Add the points and quads to the dataset
		m_pPolydata->SetPoints(points);
		m_pPolydata->GetPointData()->SetNormals(pointNormalsArray);
		m_pPolydata->SetStrips(cells);
#if VTK_MAJOR_VERSION <= 5
		m_pMapper->SetInput(m_pPolydata);
#else
		m_pMapper->SetInputData(m_pPolydata);
#endif
	}
	else
	{
		vtkSmartPointer<vtkPoints> points =  vtkSmartPointer<vtkPoints>::New();
		vtkSmartPointer<vtkDoubleArray> pointNormalsArray =  vtkSmartPointer<vtkDoubleArray>::New();
		pointNormalsArray->SetNumberOfComponents(3); //3d normals (ie x,y,z)

		vtkSmartPointer<vtkVertex> vertex = vtkSmartPointer<vtkVertex>::New();
		// Create a cell array to store the vertices
		vtkSmartPointer<vtkCellArray> cells =  vtkSmartPointer<vtkCellArray>::New();

		unsigned long numVertices=(m_renderOptions.m_slicesWidth+1)*2;

		vtkIdType pid;
		pointNormalsArray->SetNumberOfTuples(numVertices);
		vertex->GetPointIds()->SetNumberOfIds(numVertices);


		unsigned long vertexIndex=0;

		double deltaU=2*PI/m_renderOptions.m_slicesWidth;
		double a=this->getApertureRadius().X;
		double b=this->getApertureRadius().Y;

		double x=0;
		double y=0;
		double z=0;
		pid=points->InsertNextPoint(x,y,z);
		Vec3f normal=calcNormal(Vec3f());
		pointNormalsArray->SetTuple(pid, &normal.X);
		vertex->GetPointIds()->SetId(vertexIndex,vertexIndex);
		vertexIndex++;

		x=a*cos(0*deltaU);
		y=b*sin(0*deltaU);
		z=0;
		pid=points->InsertNextPoint(x,y,z);
		normal=calcNormal(Vec3f());
		pointNormalsArray->SetTuple(pid, &normal.X);
		vertex->GetPointIds()->SetId(vertexIndex,vertexIndex);
		vertexIndex++;

		for (unsigned int i=1; i<=m_renderOptions.m_slicesWidth; i++)
		//for (unsigned int i=1; i<=1; i++)
		{
			x=a*cos(double(i)*deltaU);
			y=b*sin(double(i)*deltaU);
			z=0;
			pid=points->InsertNextPoint(x,y,z);
			normal=calcNormal(Vec3f());
			pointNormalsArray->SetTuple(pid, &normal.X);
			vertex->GetPointIds()->SetId(vertexIndex,vertexIndex);
			vertexIndex++;

			x=0;
			y=0;
			z=0;
			pid=points->InsertNextPoint(x,y,z);
			normal=calcNormal(Vec3f());
			pointNormalsArray->SetTuple(pid, &normal.X);
			vertex->GetPointIds()->SetId(vertexIndex,vertexIndex);
			vertexIndex++;
		}

		cells->InsertNextCell(vertex);
		// Add the points and quads to the dataset
		m_pPolydata->SetPoints(points);
		m_pPolydata->GetPointData()->SetNormals(pointNormalsArray);
		m_pPolydata->SetStrips(cells);
#if VTK_MAJOR_VERSION <= 5
		m_pMapper->SetInput(m_pPolydata);
#else
		m_pMapper->SetInputData(m_pPolydata);
#endif
	}



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
	
	// request the update
	m_pPolydata->Update();
};

void IdealLenseItem::renderVtk(vtkSmartPointer<vtkRenderer> renderer)
{
	renderer->AddActor(m_pActor);

	this->updateVtk();
}