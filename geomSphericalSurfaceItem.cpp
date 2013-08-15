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

#include "geomSphericalSurfaceItem.h"
//#include "glut.h"

#include <vtkPoints.h>
#include <vtkVertex.h>
#include <vtkCellArray.h>
#include <vtkProperty.h>
#include <vtkDoubleArray.h>
#include <vtkPointData.h>

#include <vtkProperty.h>

using namespace macrosim;

SphericalSurfaceItem::SphericalSurfaceItem(QString name, QObject *parent) :
	GeometryItem(name, SPHERICALSURFACE, parent),
	m_radius(0)
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

SphericalSurfaceItem::~SphericalSurfaceItem()
{
	m_childs.clear();
};

bool SphericalSurfaceItem::writeToXML(QDomDocument &document, QDomElement &root) const
{
	QDomElement node = document.createElement("geometry");

	// call base class method
	if (!GeometryItem::writeToXML(document, node))
		return false;

	node.setAttribute("geomType", "SPHERICALSURFACE");
	node.setAttribute("nrSurfacesSeq", "1");
	node.setAttribute("nrSurfacesNonSeq", "1");
	node.setAttribute("radius", QString::number(m_radius));

	root.appendChild(node);
	return true;
};

bool SphericalSurfaceItem::readFromXML(const QDomElement &node)
{
	// read base class from XML
	if (!GeometryItem::readFromXML(node))
		return false;
	m_radius=node.attribute("radius").toDouble();
	return true;
};

void SphericalSurfaceItem::render(QMatrix4x4 &m, RenderOptions &options)
{
	if (this->getRender())
	{
		loadGlMatrix(m);

		glPushMatrix();

		if (this->m_focus)
			glColor3f(0.0f,1.0f,0.0f); //green
		else
			glColor3f(0.0f,0.0f,1.0f); //blue

		glTranslatef(this->getRoot().X,this->getRoot().Y,this->getRoot().Z);
		glRotatef(this->getTilt().X,1.0f,0.0f,0.0f);
		glRotatef(this->getTilt().Y,0.0f,1.0f,0.0f);
		glRotatef(this->getTilt().Z,0.0f,0.0f,1.0f);

		if (this->getApertureType()==RECTANGULAR)
		{
			// ?????????????
		}
		else
		{
			renderSemiSphere(this->getApertureRadius().X, this->getRadius(), 1, options);
		}

		glPopMatrix();
	}
};

Vec3f SphericalSurfaceItem::calcNormal(Vec3f vertex, Vec3f* neighbours, int nr)
{
	// calc centre of sphere
	Vec3f orientation=Vec3f(0,0,1);
	Vec3f centre=Vec3f(0,0,0)+orientation*this->getRadius();
	Vec3f normal=vertex-centre;
	// normalize
	normal=normal/(sqrt(normal*normal));
	if (this->getRadius() <0)
		normal=normal*-1;
	return normal;
};

Vec3f SphericalSurfaceItem::calcNormal(Vec3f vertex)
{
	// calc centre of sphere
	Vec3f orientation=Vec3f(0,0,1);
	Vec3f centre=Vec3f(0,0,0)+orientation*this->getRadius();
	Vec3f normal=vertex-centre;
	// normalize
	normal=normal/(sqrt(normal*normal));
	if (this->getRadius() <0)
		normal=normal*-1;
	return normal*-1;
};

void SphericalSurfaceItem::renderVtk(vtkSmartPointer<vtkRenderer> renderer)
{
	renderer->AddActor(m_pActor);

	this->updateVtk();
};

void SphericalSurfaceItem::updateVtk()
{

	vtkSmartPointer<vtkPoints> points =  vtkSmartPointer<vtkPoints>::New();
	vtkSmartPointer<vtkDoubleArray> pointNormalsArray =  vtkSmartPointer<vtkDoubleArray>::New();
	pointNormalsArray->SetNumberOfComponents(3); //3d normals (ie x,y,z)

	vtkSmartPointer<vtkVertex> vertex = vtkSmartPointer<vtkVertex>::New();
	// Create a cell array to store the vertices
	vtkSmartPointer<vtkCellArray> cells =  vtkSmartPointer<vtkCellArray>::New();

	double ar=this->getApertureRadius().X;
	double r=-this->m_radius;

	double deltaU=2*PI/m_renderOptions.m_slicesWidth;
	double deltaV;

	if (ar>=abs(r) )
		// if aperture is bigger than radius, we need to draw the full semi-sphere
		deltaV=PI/2/m_renderOptions.m_slicesHeight;
	else
	{
		if (ar < abs(r))
		{
			// if not, we only draw part of the hemi-sphere
			deltaV=std::min(asin(ar/r), std::min(asin(1.0), PI/2))/m_renderOptions.m_slicesHeight;
		}
		else
			deltaV=std::min(asin(1.0), PI/2)/m_renderOptions.m_slicesHeight;
	}

	// calc number of vertices
	unsigned long numVert=m_renderOptions.m_slicesHeight*4+m_renderOptions.m_slicesHeight*(m_renderOptions.m_slicesWidth)*2;
	pointNormalsArray->SetNumberOfTuples(numVert);
	vertex->GetPointIds()->SetNumberOfIds(numVert);
	vtkIdType pid;

	unsigned long vertexIndex=0;

	for (int iv=0; iv<m_renderOptions.m_slicesHeight; iv++)
	{
		double phi=0+iv*deltaV;

		double x=r*sin(phi)*cos(0.0);
		double y=r*sin(phi)*sin(0.0);
		double z=r*cos(phi)-r;
		Vec3f normal;
		normal=calcNormal(Vec3f(x,y,z));
		pid=points->InsertNextPoint(x,y,z);
		pointNormalsArray->SetTuple(pid, &normal.X);
		vertex->GetPointIds()->SetId(vertexIndex,vertexIndex);
//		cells->InsertNextCell(1, pid);
		vertexIndex++;

		x=r*sin(phi+deltaV)*cos(0.0f);
		y=r*sin(phi+deltaV)*sin(0.0f);
		z=r*cos(phi+deltaV)-r;
		pid=points->InsertNextPoint(x,y,z);
		normal=calcNormal(Vec3f(x,y,z));
		pointNormalsArray->SetTuple(pid, &normal.X);
		vertex->GetPointIds()->SetId(vertexIndex,vertexIndex);
//		cells->InsertNextCell(1, pid);
		vertexIndex++;
	
		x=r*sin(phi)*cos(deltaU);
		y=r*sin(phi)*sin(deltaU);
		z=r*cos(phi)-r;
		pid=points->InsertNextPoint(x,y,z);
		normal=calcNormal(Vec3f(x,y,z));
		pointNormalsArray->SetTuple(pid, &normal.X);
		vertex->GetPointIds()->SetId(vertexIndex,vertexIndex);
//		cells->InsertNextCell(1, pid);
		vertexIndex++;
	
		x=r*sin(phi+deltaV)*cos(deltaU);
		y=r*sin(phi+deltaV)*sin(deltaU);
		z=r*cos(phi+deltaV)-r;
		pid=points->InsertNextPoint(x,y,z);
		normal=calcNormal(Vec3f(x,y,z));
		pointNormalsArray->SetTuple(pid, &normal.X);
		vertex->GetPointIds()->SetId(vertexIndex,vertexIndex);
//		cells->InsertNextCell(1, pid);
		vertexIndex++;

		for (int iu=1; iu<=m_renderOptions.m_slicesWidth; iu++)
		{
			double theta=0+iu*deltaU;
			x=r*sin(phi)*cos(theta);
			y=r*sin(phi)*sin(theta);
			z=r*cos(phi)-r;
			pid=points->InsertNextPoint(x,y,z);
			normal=calcNormal(Vec3f(x,y,z));
			pointNormalsArray->SetTuple(pid, &normal.X);
			vertex->GetPointIds()->SetId(vertexIndex,vertexIndex);
//			cells->InsertNextCell(1, pid);
			vertexIndex++;

			x=r*sin(phi+deltaV)*cos(theta);
			y=r*sin(phi+deltaV)*sin(theta);
			z=r*cos(phi+deltaV)-r;
			pid=points->InsertNextPoint(x,y,z);
			normal=calcNormal(Vec3f(x,y,z));
			pointNormalsArray->SetTuple(pid, &normal.X);
			vertex->GetPointIds()->SetId(vertexIndex,vertexIndex);
//			cells->InsertNextCell(1, pid);
			vertexIndex++;
		}
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