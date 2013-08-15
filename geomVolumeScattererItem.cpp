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

#include "geomVolumeScattererItem.h"
#include "glut.h"

#include <vtkPoints.h>
#include <vtkVertex.h>
#include <vtkCellArray.h>
#include <vtkProperty.h>
#include <vtkDoubleArray.h>
#include <vtkPointData.h>

#include <vtkProperty.h>

using namespace macrosim;

VolumeScattererItem::VolumeScattererItem(QString name, QObject *parent) :
	GeometryItem(name, VOLUMESCATTERER, parent),
	m_meanFreePath(1),
	m_thickness(5),
	m_maxNrBounces(2),
	m_showRayPaths(false)
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
};

VolumeScattererItem::~VolumeScattererItem()
{
	m_childs.clear();
};

bool VolumeScattererItem::writeToXML(QDomDocument &document, QDomElement &root) const
{
	QDomElement node = document.createElement("geometry");

	// call base class method
	if (!GeometryItem::writeToXML(document, node))
		return false;

	node.setAttribute("geomType", "SPHERICALSURFACE");
	node.setAttribute("nrSurfacesSeq", "1");
	node.setAttribute("nrSurfacesNonSeq", "1");
	node.setAttribute("meanFreePath", QString::number(m_meanFreePath));
	node.setAttribute("depth", QString::number(m_thickness));
	node.setAttribute("maxNrBounces", QString::number(m_maxNrBounces));
	if (m_showRayPaths)
		node.setAttribute("showRayPaths", "TRUE");
	else
		node.setAttribute("showRayPaths", "FALSE");

	root.appendChild(node);
	return true;
};

bool VolumeScattererItem::readFromXML(const QDomElement &node)
{
	// read base class from XML
	if (!GeometryItem::readFromXML(node))
		return false;
	m_meanFreePath=node.attribute("meanFreePath").toDouble();
	m_thickness=node.attribute("depth").toDouble();
	m_maxNrBounces=node.attribute("maxNrBounces").toDouble();
	QString str=node.attribute("showRayPaths");
	if (!str.compare("TRUE"))
		m_showRayPaths=true;
	else
		m_showRayPaths=false;
	return true;
};

Vec3f VolumeScattererItem::calcNormal(Vec3f vertex, Vec3f* neighbours, int nr)
{
	return Vec3f(0,0,1);
};

Vec3f VolumeScattererItem::calcNormal(Vec3f vertex)
{
	return Vec3f(0,0,1);
};

void VolumeScattererItem::renderVtk(vtkSmartPointer<vtkRenderer> renderer)
{
	renderer->AddActor(m_pActor);

	this->updateVtk();
};

void VolumeScattererItem::updateVtk()
{
	vtkSmartPointer<vtkPoints> points =  vtkSmartPointer<vtkPoints>::New();
	vtkSmartPointer<vtkDoubleArray> pointNormalsArray =  vtkSmartPointer<vtkDoubleArray>::New();
	pointNormalsArray->SetNumberOfComponents(3); //3d normals (ie x,y,z)

	vtkSmartPointer<vtkVertex> vertex = vtkSmartPointer<vtkVertex>::New();
	// Create a cell array to store the vertices
	vtkSmartPointer<vtkCellArray> cells =  vtkSmartPointer<vtkCellArray>::New();

	unsigned long vertexIndex=0;
	
	// calc number of vertices 
	unsigned long numVert=17;

	pointNormalsArray->SetNumberOfTuples(numVert);

	vertex->GetPointIds()->SetNumberOfIds(numVert);

	vtkIdType pid;
	vertexIndex=0;

	Vec3d root=this->getRoot();
	Vec2d aptRadius=this->getApertureRadius();
	Vec3f normal=Vec3f(0,0,-1);

	// render front face
	double x=-aptRadius.X;
	double y=-aptRadius.Y;
	double z=0;
	pid=points->InsertNextPoint(x,y,z);
	pointNormalsArray->SetTuple(pid, &normal.X);
	vertex->GetPointIds()->SetId(vertexIndex,vertexIndex);
	vertexIndex++;

	x=-aptRadius.X;
	y=aptRadius.Y;
	z=0;
	pid=points->InsertNextPoint(x,y,z);
	pointNormalsArray->SetTuple(pid, &normal.X);
	vertex->GetPointIds()->SetId(vertexIndex,vertexIndex);
	vertexIndex++;

	x=aptRadius.X;
	y=-aptRadius.Y;
	z=0;		
	pid=points->InsertNextPoint(x,y,z);
	pointNormalsArray->SetTuple(pid, &normal.X);
	vertex->GetPointIds()->SetId(vertexIndex,vertexIndex);
	vertexIndex++;

	x=aptRadius.X;
	y=aptRadius.Y;
	z=0;
	pid=points->InsertNextPoint(x,y,z);
	pointNormalsArray->SetTuple(pid, &normal.X);
	vertex->GetPointIds()->SetId(vertexIndex,vertexIndex);
	vertexIndex++;

	// side face
	x=aptRadius.X;
	y=aptRadius.Y;
	z=this->getThickness();
	pid=points->InsertNextPoint(x,y,z);
	normal=Vec3f(-1,0,0);
	pointNormalsArray->SetTuple(pid, &normal.X);
	vertex->GetPointIds()->SetId(vertexIndex,vertexIndex);
	vertexIndex++;

	x=aptRadius.X;
	y=-aptRadius.Y;
	z=0;
	pid=points->InsertNextPoint(x,y,z);
	normal=Vec3f(0,1,0);
	pointNormalsArray->SetTuple(pid, &normal.X);
	vertex->GetPointIds()->SetId(vertexIndex,vertexIndex);
	vertexIndex++;

	x=aptRadius.X;
	y=-aptRadius.Y;
	z=this->getThickness();
	pid=points->InsertNextPoint(x,y,z);
	normal=Vec3f(-1,0,0);
	pointNormalsArray->SetTuple(pid, &normal.X);
	vertex->GetPointIds()->SetId(vertexIndex,vertexIndex);
	vertexIndex++;

	x=-aptRadius.X;
	y=-aptRadius.Y;
	z=0;
	pid=points->InsertNextPoint(x,y,z);
	normal=Vec3f(0,1,0);
	pointNormalsArray->SetTuple(pid, &normal.X);
	vertex->GetPointIds()->SetId(vertexIndex,vertexIndex);
	vertexIndex++;

	x=-aptRadius.X;
	y=-aptRadius.Y;
	z=this->getThickness();
	pid=points->InsertNextPoint(x,y,z);
	normal=Vec3f(0,1,0);
	pointNormalsArray->SetTuple(pid, &normal.X);
	vertex->GetPointIds()->SetId(vertexIndex,vertexIndex);
	vertexIndex++;

	x=-aptRadius.X;
	y=+aptRadius.Y;
	z=0;
	pid=points->InsertNextPoint(x,y,z);
	normal=Vec3f(1,0,0);
	pointNormalsArray->SetTuple(pid, &normal.X);
	vertex->GetPointIds()->SetId(vertexIndex,vertexIndex);
	vertexIndex++;

	x=-aptRadius.X;
	y=+aptRadius.Y;
	z=this->getThickness();
	pid=points->InsertNextPoint(x,y,z);
	normal=Vec3f(1,0,0);
	pointNormalsArray->SetTuple(pid, &normal.X);
	vertex->GetPointIds()->SetId(vertexIndex,vertexIndex);
	vertexIndex++;

	x=+aptRadius.X;
	y=+aptRadius.Y;
	z=0;
	pid=points->InsertNextPoint(x,y,z);
	normal=Vec3f(0,-1,0);
	pointNormalsArray->SetTuple(pid, &normal.X);
	vertex->GetPointIds()->SetId(vertexIndex,vertexIndex);
	vertexIndex++;

	x=+aptRadius.X;
	y=+aptRadius.Y;
	z=this->getThickness();
	pid=points->InsertNextPoint(x,y,z);
	normal=Vec3f(0,-1,0);
	pointNormalsArray->SetTuple(pid, &normal.X);
	vertex->GetPointIds()->SetId(vertexIndex,vertexIndex);
	vertexIndex++;

	// render back face
	x=+aptRadius.X;
	y=+aptRadius.Y;
	z=m_thickness;
	pid=points->InsertNextPoint(x,y,z);
	normal=Vec3f(0,0,-1);
	pointNormalsArray->SetTuple(pid, &normal.X);
	vertex->GetPointIds()->SetId(vertexIndex,vertexIndex);
	vertexIndex++;

	x=-aptRadius.X;
	y=aptRadius.Y;
	z=m_thickness;
	pid=points->InsertNextPoint(x,y,z);
	normal=Vec3f(0,0,-1);
	pointNormalsArray->SetTuple(pid, &normal.X);
	vertex->GetPointIds()->SetId(vertexIndex,vertexIndex);
	vertexIndex++;

	x=aptRadius.X;
	y=-aptRadius.Y;
	z=m_thickness;
	pid=points->InsertNextPoint(x,y,z);
	normal=Vec3f(0,0,-1);
	pointNormalsArray->SetTuple(pid, &normal.X);
	vertex->GetPointIds()->SetId(vertexIndex,vertexIndex);
	vertexIndex++;

	x=-aptRadius.X;
	y=-aptRadius.Y;
	z=m_thickness;
	pid=points->InsertNextPoint(x,y,z);
	normal=Vec3f(0,0,-1);
	pointNormalsArray->SetTuple(pid, &normal.X);
	vertex->GetPointIds()->SetId(vertexIndex,vertexIndex);
	vertexIndex++;



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
};