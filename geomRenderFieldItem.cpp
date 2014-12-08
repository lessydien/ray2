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

#include "geomRenderFieldItem.h"
#include <vtkCellArray.h>
#include <vtkPoints.h>
#include <vtkDoubleArray.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkVertex.h>
#include <vtkPointData.h>

#include "materialItemLib.h"
#include "geometryItemLib.h"

using namespace macrosim;

GeomRenderFieldItem::GeomRenderFieldItem(QString name, QObject *parent) :
	FieldItem(name, GEOMRENDERFIELD, parent),
        m_coherence(0),
        m_width(100),
        m_height(100),
        m_widthLayout(10),
        m_heightLayout(10),
        m_raysPerPixel(5,5),
        m_fileName(QString("renderField.txt"))

{
	// Create a polydata to store everything in
	m_pPolydata = vtkSmartPointer<vtkPolyData>::New();
  
	// Setup actor and mapper
	m_pMapper =	vtkSmartPointer<vtkPolyDataMapper>::New();

	m_pActor->SetMapper(m_pMapper);

	this->setRender(true); //per default we dont render the ray field
}

GeomRenderFieldItem::~GeomRenderFieldItem()
{
	m_childs.clear();
}

bool GeomRenderFieldItem::signalDataChanged() 
{

	return true;
};

bool GeomRenderFieldItem::writeToXML(QDomDocument &document, QDomElement &root) const 
{
	QDomElement node = document.createElement("field");

	if (!FieldItem::writeToXML(document, node))
		return false;

	node.setAttribute("tilt.x", QString::number(m_tilt.X));
	node.setAttribute("tilt.y", QString::number(m_tilt.Y));
	node.setAttribute("tilt.z", QString::number(m_tilt.Z));
	node.setAttribute("rayDirection.x", QString::number(m_rayDirection.X));
	node.setAttribute("rayDirection.y", QString::number(m_rayDirection.Y));
	node.setAttribute("rayDirection.z", QString::number(m_rayDirection.Z));
	node.setAttribute("pupRoot.x", QString::number(m_pupRoot.X));
    node.setAttribute("pupRoot.y", QString::number(m_pupRoot.Y));
    node.setAttribute("pupRoot.z", QString::number(m_pupRoot.Z));
    node.setAttribute("pupTilt.x", QString::number(m_pupTilt.X));
    node.setAttribute("pupTilt.y", QString::number(m_pupTilt.Y));
    node.setAttribute("pupTilt.z", QString::number(m_pupTilt.Z));
    node.setAttribute("pupAptRad.x", QString::number(m_pupAptRad.X));
    node.setAttribute("pupAptRad.y", QString::number(m_pupAptRad.Y));
	node.setAttribute("coherence", QString::number(m_coherence));
	node.setAttribute("width", QString::number(m_width));
    node.setAttribute("height", QString::number(m_height));
	node.setAttribute("widthLayout", QString::number(m_widthLayout));
    node.setAttribute("heightLayout", QString::number(m_heightLayout));
    node.setAttribute("raysPerPixel.x", QString::number(m_raysPerPixel.X));
    node.setAttribute("raysPerPixel.y", QString::number(m_raysPerPixel.Y));

	// write material
	if (!this->getChild()->writeToXML(document,node))
		return false;

	root.appendChild(node);

    // we also write a detector here
    node = document.createElement("detector");

	node.setAttribute("objectType", "DETECTOR"); 
	node.setAttribute("name", m_name);

	node.setAttribute("root.x", m_root.X);
	node.setAttribute("root.y", m_root.Y);
	node.setAttribute("root.z", m_root.Z);
	node.setAttribute("tilt.x", QString::number(m_tilt.X));
	node.setAttribute("tilt.y", QString::number(m_tilt.Y));
	node.setAttribute("tilt.z", QString::number(m_tilt.Z));
	node.setAttribute("apertureHalfWidth.x", QString::number(m_apertureHalfWidth.X));
	node.setAttribute("apertureHalfWidth.y", QString::number(m_apertureHalfWidth.Y));

	node.setAttribute("detType", "INTENSITY");
	node.setAttribute("detOutFormat", "TEXT");
	node.setAttribute("fileName", m_fileName);

	node.setAttribute("detPixel.x", QString::number(m_width));
	node.setAttribute("detPixel.y", QString::number(m_height));
	node.setAttribute("ignoreDepth", QString::number(-1));

    root.appendChild(node);

	return true;
}

bool GeomRenderFieldItem::readFromXML(const QDomElement &node)
{
	if (!FieldItem::readFromXML(node))
		return false;

	m_tilt.X=node.attribute("tilt.x").toDouble();
	m_tilt.Y=node.attribute("tilt.y").toDouble();
	m_tilt.Z=node.attribute("tilt.z").toDouble();
	m_rayDirection.X=node.attribute("rayDirection.x").toDouble();
	m_rayDirection.Y=node.attribute("rayDirection.y").toDouble();
	m_rayDirection.Z=node.attribute("rayDirection.z").toDouble();
	m_pupRoot.X=node.attribute("pupRoot.x").toDouble();
    m_pupRoot.Y=node.attribute("pupRoot.y").toDouble();
    m_pupRoot.Z=node.attribute("pupRoot.z").toDouble();
    m_pupTilt.X=node.attribute("pupTilt.x").toDouble();
    m_pupTilt.Y=node.attribute("pupTilt.y").toDouble();
    m_pupTilt.Z=node.attribute("pupTilt.z").toDouble();
    m_pupAptRad.X=node.attribute("pupAptRad.x").toDouble();
    m_pupAptRad.Y=node.attribute("pupAptRad.y").toDouble();
	m_coherence=node.attribute("coherence").toDouble();
	m_width=node.attribute("width").toDouble();
	m_height=node.attribute("height").toDouble();
	m_widthLayout=node.attribute("widthLayout").toDouble();
	m_heightLayout=node.attribute("heightLayout").toDouble();
    m_raysPerPixel.X=node.attribute("raysPerPixel.x").toDouble();
    m_raysPerPixel.Y=node.attribute("raysPerPixel.y").toDouble();

	// read material
	// look for material
	QDomNodeList l_matNodeList=node.elementsByTagName("material");
	if (l_matNodeList.count()==0)
		return false;
	QDomElement l_matElementXML=l_matNodeList.at(0).toElement();
	MaterialItemLib l_materialLib;
	MaterialItem l_materialItem;
	QString l_matTypeStr=l_matElementXML.attribute("materialType");
	MaterialItem* l_pMaterialItem = l_materialLib.createMaterial(l_materialLib.stringToMaterialType(l_matTypeStr));
	if (!l_pMaterialItem->readFromXML(l_matElementXML))
		return false;

	GeometryItemLib l_geomItemLib;
	m_materialType=l_geomItemLib.stringToGeomMatType(l_matTypeStr);

	this->setChild(l_pMaterialItem);

	return true;
}

void GeomRenderFieldItem::render(QMatrix4x4 &m, RenderOptions &options)
{

}

void GeomRenderFieldItem::renderVtk(vtkSmartPointer<vtkRenderer> renderer)
{
	renderer->AddActor(m_pActor);

	this->updateVtk();
}

void GeomRenderFieldItem::updateVtk()
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
        float p0[3] = {float(-this->m_apertureHalfWidth.X), float(-this->m_apertureHalfWidth.Y), 0.0f};
		float p1[3] = {float(-this->m_apertureHalfWidth.X), float(this->m_apertureHalfWidth.Y), 0.0f};
		float p3[3] = {float(this->m_apertureHalfWidth.X), float(this->m_apertureHalfWidth.Y), 0.0f};
		float p2[3] = {float(this->m_apertureHalfWidth.X), float(-this->m_apertureHalfWidth.Y), 0.0f};
 
		// Add the points to a vtkPoints object
		pid=points->InsertNextPoint(p0);
		Vec3f normal=Vec3f(0,0,1);
		pointNormalsArray->SetTuple(pid, &normal.X);
		vertex->GetPointIds()->SetId(0,0);

		pid=points->InsertNextPoint(p1);
		normal=Vec3f(0,0,1);
		pointNormalsArray->SetTuple(pid, &normal.X);
		vertex->GetPointIds()->SetId(1,1);

		pid=points->InsertNextPoint(p2);
		normal=Vec3f(0,0,1);
		pointNormalsArray->SetTuple(pid, &normal.X);
		vertex->GetPointIds()->SetId(2,2);

		pid=points->InsertNextPoint(p3);
		normal=Vec3f(0,0,1);
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