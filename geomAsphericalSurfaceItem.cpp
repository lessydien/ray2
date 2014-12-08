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

#include "geomAsphericalSurfaceItem.h"
//#include "glut.h"
#include <vtkCellArray.h>
#include <vtkPoints.h>
#include <vtkDoubleArray.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkVertex.h>
#include <vtkPointData.h>


using namespace macrosim;

AsphericalSurfaceItem::AsphericalSurfaceItem(QString name, QObject *parent) :
	GeometryItem(name, ASPHERICALSURF, parent),
	m_k(0), m_c(0), m_c2(0), m_c4(0), m_c6(0), m_c8(0), m_c10(0), m_c12(0), m_c14(0), m_c16(0)
{

}

AsphericalSurfaceItem::~AsphericalSurfaceItem()
{
	m_childs.clear();
}

bool AsphericalSurfaceItem::writeToXML(QDomDocument &document, QDomElement &root) const
{
	QDomElement node = document.createElement("geometry");

	// call base class method
	if (!GeometryItem::writeToXML(document, node))
		return false;

	node.setAttribute("geomType", "ASPHERICALSURF");
	node.setAttribute("nrSurfacesSeq", "1");
	node.setAttribute("nrSurfacesNonSeq", "1");
	node.setAttribute("k", QString::number(m_k));
	node.setAttribute("c", QString::number(m_c));
	node.setAttribute("c2", QString::number(m_c2));
	node.setAttribute("c4", QString::number(m_c4));
	node.setAttribute("c6", QString::number(m_c6));
	node.setAttribute("c8", QString::number(m_c8));
	node.setAttribute("c10", QString::number(m_c10));
	node.setAttribute("c12", QString::number(m_c12));
	node.setAttribute("c14", QString::number(m_c14));
	node.setAttribute("c16", QString::number(m_c16));
	root.appendChild(node);
	return true;
}

bool AsphericalSurfaceItem::readFromXML(const QDomElement &node)
{
	// read base class from XML
	if (!GeometryItem::readFromXML(node))
		return false;
	m_k=node.attribute("k").toDouble();
	m_c=node.attribute("c").toDouble();
	m_c2=node.attribute("c2").toDouble();
	m_c4=node.attribute("c4").toDouble();
	m_c6=node.attribute("c6").toDouble();
	m_c8=node.attribute("c8").toDouble();
	m_c10=node.attribute("c10").toDouble();
	m_c12=node.attribute("c12").toDouble();
	m_c14=node.attribute("c14").toDouble();
	m_c16=node.attribute("c16").toDouble();
	return true;
}

void AsphericalSurfaceItem::render(QMatrix4x4 &m, RenderOptions &options)
{
	if (this->getRender())
	{

	}

}

void AsphericalSurfaceItem::renderVtk(vtkSmartPointer<vtkRenderer> renderer)
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

void AsphericalSurfaceItem::updateVtk()
{
	vtkSmartPointer<vtkPoints> points =  vtkSmartPointer<vtkPoints>::New();
	vtkSmartPointer<vtkDoubleArray> pointNormalsArray =  vtkSmartPointer<vtkDoubleArray>::New();
	pointNormalsArray->SetNumberOfComponents(3); //3d normals (ie x,y,z)

	vtkSmartPointer<vtkVertex> vertex = vtkSmartPointer<vtkVertex>::New();
	// Create a cell array to store the vertices
	vtkSmartPointer<vtkCellArray> cells =  vtkSmartPointer<vtkCellArray>::New();

	double ar=this->getApertureRadius().X;
	double r=-this->m_c;

	double deltaU=2*PI/m_renderOptions.m_slicesWidth;
	double deltaV;

	deltaV=ar/m_renderOptions.m_slicesHeight;

	// calc number of vertices
	unsigned long numVert=m_renderOptions.m_slicesHeight*4+m_renderOptions.m_slicesHeight*(m_renderOptions.m_slicesWidth)*2;
	pointNormalsArray->SetNumberOfTuples(numVert);
	vertex->GetPointIds()->SetNumberOfIds(numVert);
	vtkIdType pid;

	unsigned long vertexIndex=0;

	for (int iv=0; iv<m_renderOptions.m_slicesHeight; iv++)
	{
		double l_r=iv*deltaV;

		double x=l_r*cos(0.0);
		double y=l_r*sin(0.0);
		double z=this->calcZ(sqrt(x*x+y*y));
		Vec3f normal;
		normal=calcNormal(Vec3f(x,y,z));
		pid=points->InsertNextPoint(x,y,z);
		pointNormalsArray->SetTuple(pid, &normal.X);
		vertex->GetPointIds()->SetId(vertexIndex,vertexIndex);
//		cells->InsertNextCell(1, pid);
		vertexIndex++;

		x=(l_r+deltaV)*cos(0.0f);
		y=(l_r+deltaV)*sin(0.0f);
		z=this->calcZ(sqrt(x*x+y*y));
		pid=points->InsertNextPoint(x,y,z);
		normal=calcNormal(Vec3f(x,y,z));
		pointNormalsArray->SetTuple(pid, &normal.X);
		vertex->GetPointIds()->SetId(vertexIndex,vertexIndex);
//		cells->InsertNextCell(1, pid);
		vertexIndex++;
	
		x=l_r*cos(deltaU);
		y=l_r*sin(deltaU);
		z=this->calcZ(sqrt(x*x+y*y));
		pid=points->InsertNextPoint(x,y,z);
		normal=calcNormal(Vec3f(x,y,z));
		pointNormalsArray->SetTuple(pid, &normal.X);
		vertex->GetPointIds()->SetId(vertexIndex,vertexIndex);
//		cells->InsertNextCell(1, pid);
		vertexIndex++;
	
		x=(l_r+deltaV)*cos(deltaU);
		y=(l_r+deltaV)*sin(deltaU);
		z=this->calcZ(sqrt(x*x+y*y));
		pid=points->InsertNextPoint(x,y,z);
		normal=calcNormal(Vec3f(x,y,z));
		pointNormalsArray->SetTuple(pid, &normal.X);
		vertex->GetPointIds()->SetId(vertexIndex,vertexIndex);
//		cells->InsertNextCell(1, pid);
		vertexIndex++;

		for (int iu=1; iu<=m_renderOptions.m_slicesWidth; iu++)
		{
			double theta=0+iu*deltaU;
			x=l_r*cos(theta);
			y=l_r*sin(theta);
			z=this->calcZ(sqrt(x*x+y*y));
			pid=points->InsertNextPoint(x,y,z);
			normal=calcNormal(Vec3f(x,y,z));
			pointNormalsArray->SetTuple(pid, &normal.X);
			vertex->GetPointIds()->SetId(vertexIndex,vertexIndex);
//			cells->InsertNextCell(1, pid);
			vertexIndex++;

			x=(l_r+deltaV)*cos(theta);
			y=(l_r+deltaV)*sin(theta);
			z=this->calcZ(sqrt(x*x+y*y));
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

#if  (VTK_MAJOR_VERSION <= 5)
	// request the update
	m_pPolydata->Update();
#else
    m_pMapper->Update();
#endif
};

Vec3f AsphericalSurfaceItem::calcNormal(Vec3f vertex)
{
	float r=sqrt(vertex.X*vertex.X+vertex.Y*vertex.Y);

	// calc derivative of z at given radial distance
	// see H. Gross, Handbook of optical systems, Vol 1, pp. 198
	float dzdr=this->m_c*r/(sqrt(1-(1+this->m_k)*pow(this->m_c,2)*pow(r,2)))+2*this->m_c2*pow(r,1)+4*this->m_c4*pow(r,3)+6*this->m_c6*pow(r,5)+8*this->m_c8*pow(r,7)+10*this->m_c10*pow(r,9)+12*this->m_c12*pow(r,11)+14*this->m_c14*pow(r,13)+16*this->m_c16*pow(r,15);

	// calc aberration of normal
	// see Malacara, Handbook of OpticalDesign, 2nd ed, A.2.1.1
	float Ln=r/dzdr+(this->m_c*r*r/(1+sqrt(1-(1+this->m_k)*this->m_c*this->m_c*r*r))+ this->m_c2*pow(r,2)+this->m_c4*pow(r,4)+this->m_c6*pow(r,6)+this->m_c8*pow(r,8)+this->m_c10*pow(r,10)+this->m_c12*pow(r,12)+this->m_c14*pow(r,14)+this->m_c16*pow(r,16));
	// calc normal
	Vec3f normal=Vec3f(this->m_root.X,this->m_root.Y,this->m_root.Z)+Vec3f(0.0,0.0,1.0)*Ln-vertex;
	// normalize
	return normal/sqrt(normal*normal);
};

double AsphericalSurfaceItem::calcZ(double r)
{
	return this->m_c*pow(r,2)/(1+sqrt(1-(1+this->m_k)*pow(this->m_c,2)*pow(r,2)))+this->m_c2*pow(r,2)+this->m_c4*pow(r,4)+this->m_c6*pow(r,6)+this->m_c8*pow(r,8)+this->m_c10*pow(r,10)+this->m_c12*pow(r,12)+this->m_c14*pow(r,14)+this->m_c16*pow(r,16);
};