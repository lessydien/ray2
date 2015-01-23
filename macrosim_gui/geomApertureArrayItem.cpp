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

#include "geomApertureArrayItem.h"
#include "materialItemLib.h"
#include "geometryItemLib.h"
//#include "glut.h"

#include <vtkVertex.h>
#include <vtkCellArray.h>
#include <vtkProperty.h>
#include <vtkDoubleArray.h>
#include <vtkPointData.h>

#ifdef max
#undef max
#endif
#include <limits>

#include <iostream>
using namespace std;

using namespace macrosim;

ApertureArrayItem::ApertureArrayItem(QString name, QObject *parent) :
	GeometryItem(name, APERTUREARRAY, parent),
		m_microAptPitch(Vec2d(2,2)),
		m_microAptRad(Vec2d(2,2)),
		m_microAptType(MICROAPT_ELLIPTICAL)
{
	// Create a polydata to store everything in
	m_pPolydata = vtkSmartPointer<vtkPolyData>::New();
  
	// Setup actor and mapper
	m_pMapper =	vtkSmartPointer<vtkPolyDataMapper>::New();

	m_pActor->SetMapper(m_pMapper);
}

ApertureArrayItem::~ApertureArrayItem()
{
	m_childs.clear();
}

QString ApertureArrayItem::microAptTypeToString(MicroAptType in) const
{
	QString out;
	switch (in)
	{
	case MICROAPT_RECTANGULAR:
		out="MICRORECTANGULAR";
		break;
	case MICROAPT_ELLIPTICAL:
		out="MICROELLIPTICAL";
		break;
	default:
		out="MICROUNKNOWN";
		break;
	}
	return out;
}

ApertureArrayItem::MicroAptType ApertureArrayItem::stringToMicroAptType(QString in) const
{
	if (in.isNull())
		return ApertureArrayItem::MICROAPT_UNKNOWN;
	if (!in.compare("MICRORECTANGULAR"))
		return ApertureArrayItem::MICROAPT_RECTANGULAR;
	if (!in.compare("MICROELLIPTICAL"))
		return ApertureArrayItem::MICROAPT_ELLIPTICAL;

	return ApertureArrayItem::MICROAPT_UNKNOWN;
}

bool ApertureArrayItem::writeToXML(QDomDocument &document, QDomElement &root) const
{
	QDomElement node = document.createElement("geometry");

	// call base class method
	if (!GeometryItem::writeToXML(document, node))
		return false;

	node.setAttribute("geomType", "APERTUREARRAY");
	node.setAttribute("nrSurfacesSeq", "1");
	node.setAttribute("nrSurfacesNonSeq", "1");
	node.setAttribute("microAptRad.x", QString::number(m_microAptRad.X));
	node.setAttribute("microAptRad.y", QString::number(m_microAptRad.Y));
	node.setAttribute("microAptPitch.x", QString::number(m_microAptPitch.X));
	node.setAttribute("microAptPitch.y", QString::number(m_microAptPitch.Y));
	node.setAttribute("microAptType", microAptTypeToString(m_microAptType));

	root.appendChild(node);
	return true;
}

bool ApertureArrayItem::readFromXML(const QDomElement &node)
{
	// read base class from XML
	if (!GeometryItem::readFromXML(node))
		return false;

	m_microAptRad.X=node.attribute("microAptRad.x").toDouble();
	m_microAptRad.Y=node.attribute("microAptRad.y").toDouble();
	m_microAptPitch.X=node.attribute("microAptPitch.x").toDouble();
	m_microAptPitch.Y=node.attribute("microAptPitch.y").toDouble();
	m_microAptType=stringToMicroAptType(node.attribute("microAptType"));

	return true;
}

Vec3f ApertureArrayItem::calcNormal(Vec3f vertex, Vec3f* neighbours, int nr)
{
	return Vec3f(0,0,1);
}


Vec3f ApertureArrayItem::calcNormal(Vec3f vertex)
{
	return Vec3f(0,0,-1);
}

void ApertureArrayItem::renderVtk(vtkSmartPointer<vtkRenderer> renderer)
{
	renderer->AddActor(m_pActor);

	this->updateVtk();
}

void ApertureArrayItem::updateVtk()
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
	m_pActor->SetPosition(this->getRoot().X, this->getRoot().Y, this->getRoot().Z);
	//m_pActor->SetOrigin(this->getRoot().X, this->getRoot().Y, this->getRoot().Z);
	m_pActor->SetOrientation(this->getTilt().X, this->getTilt().Y, this->getTilt().Z);
	//m_pActor->GetProperty()->SetRepresentationToWireframe();

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
}
