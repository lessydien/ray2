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
#include "materialVolumeScatterItem.h"
//#include "glut.h"

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
	m_thickness(5),
	m_showRayPaths(false)
{
	this->m_materialType=VOLUMESCATTER;
	this->setChild(new MaterialVolumeScatterItem());
	this->m_apertureType=RECTANGULAR;
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
	if (!AbstractItem::writeToXML(document, node))
		return false;
	node.setAttribute("root.x", QString::number(m_root.X));
	node.setAttribute("root.y", QString::number(m_root.Y));
	node.setAttribute("root.z", QString::number(m_root.Z));
	node.setAttribute("tilt.x", QString::number(m_tilt.X));
	node.setAttribute("tilt.y", QString::number(m_tilt.Y));
	node.setAttribute("tilt.z", QString::number(m_tilt.Z));
	node.setAttribute("apertureRadius.x", QString::number(m_apertureRadius.X));
	node.setAttribute("apertureRadius.y", QString::number(m_apertureRadius.Y));
	node.setAttribute("apertureType", apertureTypeToString(m_apertureType));
	//node.setAttribute("geometryID", QString::number(m_geometryID));
	node.setAttribute("geometryID", QString::number(m_index.row()));
	if (m_render)
		node.setAttribute("render", "true");
	else
		node.setAttribute("render", "false");

	node.setAttribute("geomType", "VOLUMESCATTERERBOX");

	if (m_showRayPaths)
	{
		this->getChild()->writeToXML(document, node);
	}
	else
	{
		MaterialVolumeScatterItem *l_materialItem;
		l_materialItem=reinterpret_cast<MaterialVolumeScatterItem*>(this->getChild());
		l_materialItem->writeBoxToXML(document, node, this->getApertureRadius(), this->m_thickness, this->m_root, this->m_tilt);
	}
	node.setAttribute("nrSurfacesSeq", "1");
	node.setAttribute("nrSurfacesNonSeq", "1");
	node.setAttribute("thickness", QString::number(m_thickness));
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
	m_thickness=node.attribute("thickness").toDouble();
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