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

#include "geomSubstrateItem.h"
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

SubstrateItem::SubstrateItem(QString name, QObject *parent) :
	GeometryItem(name, SUBSTRATE, parent),
		m_thickness(2)
{
	//this->setApertureType(RECTANGULAR);
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

SubstrateItem::~SubstrateItem()
{
	m_childs.clear();
}

bool SubstrateItem::writeToXML(QDomDocument &document, QDomElement &root) const
{
	// micro lens arrays consist of three surface. front face: microstructured surface. back face: plane surface. side face: clyPipe
	QDomElement baseNode = document.createElement("geometry");
	QDomElement frontNode = document.createElement("surface");
	QDomElement backNode = document.createElement("surface");
	QDomElement sideNode1 = document.createElement("surface");

	// back face
	backNode.setAttribute("faceType", "BACKFACE");
	backNode.setAttribute("objectType", "GEOMETRY");
	backNode.setAttribute("geomType", "PLANESURFACE");
	backNode.setAttribute("nrSurfacesSeq", "1");

	Vec3d orientation(0,0,1);
	rotateVec3d(&orientation, this->getTilt());
	Vec3d root2=this->getRoot()+orientation*this->getThickness();

	backNode.setAttribute("root.x", QString::number(root2.X));
	backNode.setAttribute("root.y", QString::number(root2.Y));
	backNode.setAttribute("root.z", QString::number(root2.Z));
	backNode.setAttribute("tilt.x", QString::number(this->getTilt().X));
	backNode.setAttribute("tilt.y", QString::number(this->getTilt().Y));
	backNode.setAttribute("tilt.z", QString::number(this->getTilt().Z));
	backNode.setAttribute("apertureRadius.x", QString::number(this->getApertureRadius().X));
	backNode.setAttribute("apertureRadius.y", QString::number(this->getApertureRadius().Y));
	backNode.setAttribute("apertureType", apertureTypeToString(this->getApertureType()));
	//node.setAttribute("geometryID", QString::number(m_geometryID));
	backNode.setAttribute("geometryID", QString::number(m_index.row()));

	// add material
	// we must have exactly one material
	if (m_childs.count() != 1)
		return false;
	if (!this->getChild()->writeToXML(document, backNode))
		return false;

	// front face

	frontNode.setAttribute("faceType", "FRONTFACE");
	frontNode.setAttribute("objectType", "GEOMETRY");
	frontNode.setAttribute("geomType", "PLANESURFACE");
	frontNode.setAttribute("root.x", QString::number(this->getRoot().X));
	frontNode.setAttribute("root.y", QString::number(this->getRoot().Y));
	frontNode.setAttribute("root.z", QString::number(this->getRoot().Z));
	frontNode.setAttribute("nrSurfacesSeq", "1");
	frontNode.setAttribute("nrSurfacesNonSeq", "1");
	frontNode.setAttribute("tilt.x", QString::number(this->getTilt().X));
	frontNode.setAttribute("tilt.y", QString::number(this->getTilt().Y));
	frontNode.setAttribute("tilt.z", QString::number(this->getTilt().Z));
	frontNode.setAttribute("geometryID", QString::number(this->getGeometryID()));
	frontNode.setAttribute("apertureType", apertureTypeToString(this->getApertureType()));
	frontNode.setAttribute("apertureRadius.x", QString::number(this->getApertureRadius().X));
	frontNode.setAttribute("apertureRadius.y", QString::number(this->getApertureRadius().Y));

	// add material
	if (!this->getChild()->writeToXML(document, frontNode))
		return false;


	// side face
	if (this->getApertureType() == RECTANGULAR)
	{
		QDomElement sideNode2 = document.createElement("surface");
		QDomElement sideNode3 = document.createElement("surface");
		QDomElement sideNode4 = document.createElement("surface");

		sideNode1.setAttribute("faceType", "SIDEFACE");
		sideNode1.setAttribute("objectType", "GEOMETRY");
		sideNode1.setAttribute("geomType", "PLANESURFACE");
		sideNode1.setAttribute("nrSurfacesSeq", "1");
		sideNode1.setAttribute("nrSurfacesNonSeq", "1");
		Vec3d l_tilt=concatenateTilts(this->getTilt(), Vec3d(90,0,0));
		Vec3d l_orientation=Vec3d(0,0,1);
		rotateVec3d(&l_orientation, l_tilt);
		Vec3d l_root=this->getRoot()-l_orientation*this->getApertureRadius().Y+orientation*m_thickness/2;
		sideNode1.setAttribute("root.x", QString::number(l_root.X));
		sideNode1.setAttribute("root.y", QString::number(l_root.Y));
		sideNode1.setAttribute("root.z", QString::number(l_root.Z));
		sideNode1.setAttribute("tilt.x", QString::number(l_tilt.X));
		sideNode1.setAttribute("tilt.y", QString::number(l_tilt.Y));
		sideNode1.setAttribute("tilt.z", QString::number(l_tilt.Z));
		sideNode1.setAttribute("apertureRadius.x", QString::number(this->getApertureRadius().X));
		sideNode1.setAttribute("apertureRadius.y", QString::number(m_thickness/2));
		sideNode1.setAttribute("apertureType", "RECTANGULAR");
		sideNode1.setAttribute("geometryID", QString::number(m_index.row()));
		if (m_render)
			sideNode1.setAttribute("render", "true");
		else
			sideNode1.setAttribute("render", "false");
	
		// add material
		// geometries must have exactly one child
		if (m_childs.count() != 1)
			return false;
		if (!this->getChild()->writeToXML(document, sideNode1))
			return false;


		sideNode2.setAttribute("faceType", "SIDEFACE");
		sideNode2.setAttribute("objectType", "GEOMETRY");
		sideNode2.setAttribute("geomType", "PLANESURFACE");
		sideNode2.setAttribute("nrSurfacesSeq", "1");
		sideNode2.setAttribute("nrSurfacesNonSeq", "1");
		l_tilt=concatenateTilts(this->getTilt(), Vec3d(0,90,0));
		l_orientation=Vec3d(0,0,1);
		rotateVec3d(&l_orientation, l_tilt);
		l_root=this->getRoot()+l_orientation*this->getApertureRadius().X+orientation*m_thickness/2;
		sideNode2.setAttribute("root.x", QString::number(l_root.X));
		sideNode2.setAttribute("root.y", QString::number(l_root.Y));
		sideNode2.setAttribute("root.z", QString::number(l_root.Z));
		sideNode2.setAttribute("tilt.x", QString::number(l_tilt.X));
		sideNode2.setAttribute("tilt.y", QString::number(l_tilt.Y));
		sideNode2.setAttribute("tilt.z", QString::number(l_tilt.Z));
		sideNode2.setAttribute("apertureRadius.x", QString::number(m_thickness/2));
		sideNode2.setAttribute("apertureRadius.y", QString::number(this->getApertureRadius().Y));
		sideNode2.setAttribute("apertureType", "RECTANGULAR");
		sideNode2.setAttribute("geometryID", QString::number(m_index.row()));
		if (m_render)
			sideNode2.setAttribute("render", "true");
		else
			sideNode2.setAttribute("render", "false");
		// add material
		// geometries must have exactly one child
		if (m_childs.count() != 1)
			return false;
		if (!this->getChild()->writeToXML(document, sideNode2))
			return false;

		sideNode3.setAttribute("faceType", "SIDEFACE");
		sideNode3.setAttribute("objectType", "GEOMETRY");
		sideNode3.setAttribute("geomType", "PLANESURFACE");
		sideNode3.setAttribute("nrSurfacesSeq", "1");
		sideNode3.setAttribute("nrSurfacesNonSeq", "1");
		l_tilt=concatenateTilts(this->getTilt(), Vec3d(90,0,0));
		l_orientation=Vec3d(0,0,1);
		rotateVec3d(&l_orientation, l_tilt);
		l_root=this->getRoot()+l_orientation*this->getApertureRadius().Y+orientation*m_thickness/2;
		sideNode3.setAttribute("root.x", QString::number(l_root.X));
		sideNode3.setAttribute("root.y", QString::number(l_root.Y));
		sideNode3.setAttribute("root.z", QString::number(l_root.Z));
		sideNode3.setAttribute("tilt.x", QString::number(l_tilt.X));
		sideNode3.setAttribute("tilt.y", QString::number(l_tilt.Y));
		sideNode3.setAttribute("tilt.z", QString::number(l_tilt.Z));
		sideNode3.setAttribute("apertureRadius.x", QString::number(this->getApertureRadius().X));
		sideNode3.setAttribute("apertureRadius.y", QString::number(m_thickness/2));
		sideNode3.setAttribute("apertureType", "RECTANGULAR");
		sideNode3.setAttribute("geometryID", QString::number(m_index.row()));
		if (m_render)
			sideNode3.setAttribute("render", "true");
		else
			sideNode3.setAttribute("render", "false");
		// add material
		// geometries must have exactly one child
		if (m_childs.count() != 1)
			return false;
		if (!this->getChild()->writeToXML(document, sideNode3))
			return false;

		sideNode4.setAttribute("faceType", "SIDEFACE");
		sideNode4.setAttribute("objectType", "GEOMETRY");
		sideNode4.setAttribute("geomType", "PLANESURFACE");
		sideNode4.setAttribute("nrSurfacesSeq", "1");
		sideNode4.setAttribute("nrSurfacesNonSeq", "1");
		l_tilt=concatenateTilts(this->getTilt(), Vec3d(0,90,0));
		l_orientation=Vec3d(0,0,1);
		rotateVec3d(&l_orientation, l_tilt);
		l_root=this->getRoot()-l_orientation*this->getApertureRadius().X+orientation*m_thickness/2;
		sideNode4.setAttribute("root.x", QString::number(l_root.X));
		sideNode4.setAttribute("root.y", QString::number(l_root.Y));
		sideNode4.setAttribute("root.z", QString::number(l_root.Z));
		sideNode4.setAttribute("tilt.x", QString::number(l_tilt.X));
		sideNode4.setAttribute("tilt.y", QString::number(l_tilt.Y));
		sideNode4.setAttribute("tilt.z", QString::number(l_tilt.Z));
		sideNode4.setAttribute("apertureRadius.x", QString::number(m_thickness/2));
		sideNode4.setAttribute("apertureRadius.y", QString::number(this->getApertureRadius().Y));
		sideNode4.setAttribute("apertureType", "RECTANGULAR");
		sideNode4.setAttribute("geometryID", QString::number(m_index.row()));
		if (m_render)
			sideNode4.setAttribute("render", "true");
		else
			sideNode4.setAttribute("render", "false");	
		// add material
		// geometries must have exactly one child
		if (m_childs.count() != 1)
			return false;
		if (!this->getChild()->writeToXML(document, sideNode4))
			return false;


		baseNode.setAttribute("nrSurfacesNonSeq", "6");
		baseNode.appendChild(sideNode1);
		baseNode.appendChild(sideNode2);
		baseNode.appendChild(sideNode3);
		baseNode.appendChild(sideNode4);
	}
	else
	{
		baseNode.setAttribute("nrSurfacesNonSeq", "3");

		sideNode1.setAttribute("faceType", "SIDEFACE");
		sideNode1.setAttribute("objectType", "GEOMETRY");
		sideNode1.setAttribute("geomType", "CYLPIPE");
		sideNode1.setAttribute("root.x", QString::number(this->getRoot().X));
		sideNode1.setAttribute("root.y", QString::number(this->getRoot().Y));
		sideNode1.setAttribute("root.z", QString::number(this->getRoot().Z));
		sideNode1.setAttribute("tilt.x", QString::number(this->getTilt().X));
		sideNode1.setAttribute("tilt.y", QString::number(this->getTilt().Y));
		sideNode1.setAttribute("tilt.z", QString::number(this->getTilt().Z));
		sideNode1.setAttribute("thickness", QString::number(m_thickness));
		sideNode1.setAttribute("apertureRadius.x", QString::number(this->getApertureRadius().X));
		sideNode1.setAttribute("apertureRadius.y", QString::number(this->getApertureRadius().Y));
		if (this->getApertureRadius().X != this->getApertureRadius().Y)
		{
			cout << "warning in SubstrateItem.writeToXML(): different aperture radii in x and y for elliptical apertures are not implmented yet. An circular aperture with radius x will be assuemed." << endl;
		}
		sideNode1.setAttribute("radius.x", QString::number(this->getApertureRadius().X));
		sideNode1.setAttribute("radius.y", QString::number(this->getApertureRadius().X));
		sideNode1.setAttribute("nrSurfacesSeq", "1");
		sideNode1.setAttribute("nrSurfacesNonSeq", "1");
		sideNode1.setAttribute("geometryID", QString::number(this->getGeometryID()));
		sideNode1.setAttribute("apertureType", apertureTypeToString(this->getApertureType()));

		// add material
		if (!this->getChild()->writeToXML(document, sideNode1))
			return false;

		baseNode.appendChild(sideNode1);
	}

	baseNode.appendChild(backNode);
	baseNode.appendChild(frontNode);

	baseNode.setAttribute("name", this->getName());
	baseNode.setAttribute("objectType", "GEOMETRY");
	baseNode.setAttribute("geomType", "SUBSTRATE");
	if (this->getApertureType() == RECTANGULAR)
		baseNode.setAttribute("nrSurfacesNonSeq", "6");
	else
		baseNode.setAttribute("nrSurfacesNonSeq", "3");
	baseNode.setAttribute("nrSurfacesSeq", "2");
	baseNode.setAttribute("name", this->getName());
	if (m_render)
		baseNode.setAttribute("render", "true");
	else
		baseNode.setAttribute("render", "false");


	root.appendChild(baseNode);
	return true;
}

bool SubstrateItem::readFromXML(const QDomElement &node)
{
	// read base class from XML
	QDomNodeList l_nodeList=node.elementsByTagName("surface");
	// spherical lense items consists of three surfaces
	if ( !((l_nodeList.count() == 3) || (l_nodeList.count() == 6)) )
	{
		cout << "error in SubstrateItem.readFromXML(): item has neither three nor six child items" << endl;
		return false;
	}
	this->setName(node.attribute("name"));

	if (!node.attribute("render").compare("true"))
		this->setRender(true);
	else
		this->setRender(false);

	QDomElement l_ele;
	Vec3d l_root;

	for (unsigned int i=0; i<l_nodeList.count(); i++)
	{
		l_ele=l_nodeList.at(i).toElement();
		if (!l_ele.attribute("faceType").compare("BACKFACE"))
		{
			QString l_geomType=l_ele.attribute("geomType");
			if (l_geomType.compare("PLANESURFACE"))
			{
				cout << "error in SubstrateItem.readFromXML(): front face is not a PLANESURF" << endl;
				return false;
			}
			Vec3d l_tilt;
			l_tilt.X=l_ele.attribute("tilt.x").toDouble();
			l_tilt.Y=l_ele.attribute("tilt.y").toDouble();
			l_tilt.Z=l_ele.attribute("tilt.z").toDouble();
			this->setTilt(l_tilt);
			l_root.X=l_ele.attribute("root.x").toDouble();
			l_root.Y=l_ele.attribute("root.y").toDouble();
			l_root.Z=l_ele.attribute("root.z").toDouble();
			int l_geometryID=l_ele.attribute("geometryID").toDouble();
			this->setGeometryID(l_geometryID);
			ApertureType l_at;
			l_at=stringToApertureType(l_ele.attribute("apertureType"));
			this->setApertureType(l_at);
			Vec2d l_apertureRadius;
			l_apertureRadius.X=l_ele.attribute("apertureRadius.x").toDouble();
			l_apertureRadius.Y=l_ele.attribute("apertureRadius.y").toDouble();
			this->setApertureRadius(l_apertureRadius);
		}
		if (!l_ele.attribute("faceType").compare("FRONTFACE"))
		{
			QString l_geomType=l_ele.attribute("geomType");
			if (l_geomType.compare("PLANESURFACE"))
			{
				cout << "error in SubstrateItem.readFromXML(): back face is not a plane surface" << endl;
				return false;
			}
			Vec3d t_root;
			t_root.X=l_ele.attribute("root.x").toDouble();
			t_root.Y=l_ele.attribute("root.y").toDouble();
			t_root.Z=l_ele.attribute("root.z").toDouble();
			this->setRoot(t_root);
			m_thickness=sqrt((t_root-l_root)*(t_root-l_root));
		}
		if (!l_ele.attribute("faceType").compare("SIDEFACE"))
		{
			// nothing needs to be read here....
		}
	}

	// look for material
	QDomNodeList l_matNodeList=node.elementsByTagName("material");
	if (l_matNodeList.count()==0)
		return true;
	QDomElement l_matElementXML=l_matNodeList.at(0).toElement();
	MaterialItemLib l_materialLib;
	MaterialItem l_materialItem;
	QString l_matTypeStr=l_matElementXML.attribute("materialType");
	MaterialItem* l_pMaterialItem = l_materialLib.createMaterial(l_materialLib.stringToMaterialType(l_matTypeStr));
	if (!l_pMaterialItem->readFromXML(l_matElementXML))
		return false;

	GeometryItemLib l_geomItemLib;
	this->setMaterialType(l_geomItemLib.stringToGeomMatType(l_matTypeStr));

	this->setChild(l_pMaterialItem);




	return true;
}

Vec3f SubstrateItem::calcNormal(Vec3f vertex)
{
	Vec3f normal=vertex-Vec3f(0, 0, vertex.Z);
	return normal/(sqrt(normal*normal));
}

void SubstrateItem::renderVtk(vtkSmartPointer<vtkRenderer> renderer)
{
	renderer->AddActor(m_pActor);

	this->updateVtk();
}

void SubstrateItem::updateVtk()
{
	vtkSmartPointer<vtkPoints> points =  vtkSmartPointer<vtkPoints>::New();
	vtkSmartPointer<vtkDoubleArray> pointNormalsArray =  vtkSmartPointer<vtkDoubleArray>::New();
	pointNormalsArray->SetNumberOfComponents(3); //3d normals (ie x,y,z)

	vtkSmartPointer<vtkVertex> vertex = vtkSmartPointer<vtkVertex>::New();
	// Create a cell array to store the vertices
	vtkSmartPointer<vtkCellArray> cells =  vtkSmartPointer<vtkCellArray>::New();

	unsigned long vertexIndex=0;
	
	if (this->getApertureType()==RECTANGULAR)
	{
		// calc number of vertices 
		unsigned long numVert=26;

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
		z=0;
		pid=points->InsertNextPoint(x,y,z);
		normal=Vec3f(0,-1,0);
		pointNormalsArray->SetTuple(pid, &normal.X);
		vertex->GetPointIds()->SetId(vertexIndex,vertexIndex);
		vertexIndex++;

		x=aptRadius.X;
		y=aptRadius.Y;
		z=this->getThickness();
		//z=this->getThickness();
		pid=points->InsertNextPoint(x,y,z);
		normal=Vec3f(0,-1,0);
		pointNormalsArray->SetTuple(pid, &normal.X);
		vertex->GetPointIds()->SetId(vertexIndex,vertexIndex);
		vertexIndex++;

		x=-aptRadius.X;
		y=aptRadius.Y;
		z=0;
		pid=points->InsertNextPoint(x,y,z);
		normal=Vec3f(0,-1,0);
		pointNormalsArray->SetTuple(pid, &normal.X);
		vertex->GetPointIds()->SetId(vertexIndex,vertexIndex);
		vertexIndex++;

		x=-aptRadius.X;
		y=aptRadius.Y;
		z=this->getThickness();
		pid=points->InsertNextPoint(x,y,z);
		normal=Vec3f(0,-1,0);
		pointNormalsArray->SetTuple(pid, &normal.X);
		vertex->GetPointIds()->SetId(vertexIndex,vertexIndex);
		vertexIndex++;

		x=-aptRadius.X;
		y=aptRadius.Y;
		z=this->getThickness();
		pid=points->InsertNextPoint(x,y,z);
		normal=Vec3f(-1,0,0);
		pointNormalsArray->SetTuple(pid, &normal.X);
		vertex->GetPointIds()->SetId(vertexIndex,vertexIndex);
		vertexIndex++;

		x=-aptRadius.X;
		y=aptRadius.Y;
		z=0;
		pid=points->InsertNextPoint(x,y,z);
		normal=Vec3f(-1,0,0);
		pointNormalsArray->SetTuple(pid, &normal.X);
		vertex->GetPointIds()->SetId(vertexIndex,vertexIndex);
		vertexIndex++;

		x=-aptRadius.X;
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
		normal=Vec3f(0,1,0);
		pointNormalsArray->SetTuple(pid, &normal.X);
		vertex->GetPointIds()->SetId(vertexIndex,vertexIndex);
		vertexIndex++;

		x=aptRadius.X;
		y=-aptRadius.Y;
		z=this->getThickness();
		pid=points->InsertNextPoint(x,y,z);
		normal=Vec3f(1,0,0);
		pointNormalsArray->SetTuple(pid, &normal.X);
		vertex->GetPointIds()->SetId(vertexIndex,vertexIndex);
		vertexIndex++;

		x=aptRadius.X;
		y=-aptRadius.Y;
		z=0;
		pid=points->InsertNextPoint(x,y,z);
		normal=Vec3f(1,0,0);
		pointNormalsArray->SetTuple(pid, &normal.X);
		vertex->GetPointIds()->SetId(vertexIndex,vertexIndex);
		vertexIndex++;

		x=aptRadius.X;
		y=aptRadius.Y;
		z=this->getThickness();
		pid=points->InsertNextPoint(x,y,z);
		normal=Vec3f(1,0,0);
		pointNormalsArray->SetTuple(pid, &normal.X);
		vertex->GetPointIds()->SetId(vertexIndex,vertexIndex);
		vertexIndex++;

		x=aptRadius.X;
		y=aptRadius.Y;
		z=0;
		pid=points->InsertNextPoint(x,y,z);
		normal=Vec3f(1,0,0);
		pointNormalsArray->SetTuple(pid, &normal.X);
		vertex->GetPointIds()->SetId(vertexIndex,vertexIndex);
		vertexIndex++;

		x=aptRadius.X;
		y=aptRadius.Y;
		z=0;
		pid=points->InsertNextPoint(x,y,z);
		normal=Vec3f(1,0,0);
		pointNormalsArray->SetTuple(pid, &normal.X);
		vertex->GetPointIds()->SetId(vertexIndex,vertexIndex);
		vertexIndex++;

		x=aptRadius.X;
		y=aptRadius.Y;
		z=this->getThickness();
		pid=points->InsertNextPoint(x,y,z);
		normal=Vec3f(1,0,0);
		pointNormalsArray->SetTuple(pid, &normal.X);
		vertex->GetPointIds()->SetId(vertexIndex,vertexIndex);
		vertexIndex++;

		// render back face
		x=aptRadius.X;
		y=aptRadius.Y;
		z=this->getThickness();
		pid=points->InsertNextPoint(x,y,z);
		normal=Vec3f(0,0,1);
		pointNormalsArray->SetTuple(pid, &normal.X);
		vertex->GetPointIds()->SetId(vertexIndex,vertexIndex);
		vertexIndex++;

		x=-aptRadius.X;
		y=aptRadius.Y;
		z=this->getThickness();
		pid=points->InsertNextPoint(x,y,z);
		normal=Vec3f(0,0,1);
		pointNormalsArray->SetTuple(pid, &normal.X);
		vertex->GetPointIds()->SetId(vertexIndex,vertexIndex);
		vertexIndex++;

		x=aptRadius.X;
		y=-aptRadius.Y;
		z=this->getThickness();
		pid=points->InsertNextPoint(x,y,z);
		normal=Vec3f(0,0,1);
		pointNormalsArray->SetTuple(pid, &normal.X);
		vertex->GetPointIds()->SetId(vertexIndex,vertexIndex);
		vertexIndex++;

		x=-aptRadius.X;
		y=-aptRadius.Y;
		z=this->getThickness();
		pid=points->InsertNextPoint(x,y,z);
		normal=Vec3f(0,0,1);
		pointNormalsArray->SetTuple(pid, &normal.X);
		vertex->GetPointIds()->SetId(vertexIndex,vertexIndex);
		vertexIndex++;

	}
	else
	{

		unsigned long numVertices=2*((m_renderOptions.m_slicesWidth+1)*2)+(4+2*(m_renderOptions.m_slicesWidth-1))+4;

		vtkIdType pid;
		pointNormalsArray->SetNumberOfTuples(numVertices);
		vertex->GetPointIds()->SetNumberOfIds(numVertices);


		vertexIndex=0;

		double deltaU=2*PI/m_renderOptions.m_slicesWidth;
		double a=this->getApertureRadius().X;
		double b=this->getApertureRadius().Y;

		// front face
		double x=a*cos(0*deltaU);
		double y=b*sin(0*deltaU);
		double z=0;
		pid=points->InsertNextPoint(x,y,z);
		Vec3f normal=Vec3f(0,0,-1);
		pointNormalsArray->SetTuple(pid, &normal.X);
		vertex->GetPointIds()->SetId(vertexIndex,vertexIndex);
		vertexIndex++;

		x=0;
		y=0;
		z=0;
		pid=points->InsertNextPoint(x,y,z);
		normal=Vec3f(0,0,-1);
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
			normal=Vec3f(0,0,-1);
			pointNormalsArray->SetTuple(pid, &normal.X);
			vertex->GetPointIds()->SetId(vertexIndex,vertexIndex);
			vertexIndex++;

			x=0;
			y=0;
			z=0;
			pid=points->InsertNextPoint(x,y,z);
			normal=Vec3f(0,0,-1);
			pointNormalsArray->SetTuple(pid, &normal.X);
			vertex->GetPointIds()->SetId(vertexIndex,vertexIndex);
			vertexIndex++;
		}
		x=0;
		y=0;
		z=0;
		pid=points->InsertNextPoint(x,y,z);
		normal=Vec3f(0,0,-1);
		pointNormalsArray->SetTuple(pid, &normal.X);
		vertex->GetPointIds()->SetId(vertexIndex,vertexIndex);
		vertexIndex++;

		// side face
		x=a*cos(0*deltaU);
		y=b*sin(0*deltaU);
		z=m_thickness;
		normal=calcNormal(Vec3f(x,y,z));
		pid=points->InsertNextPoint(x,y,z);
		pointNormalsArray->SetTuple(pid, &normal.X);
		vertex->GetPointIds()->SetId(vertexIndex,vertexIndex);
		vertexIndex++;

		x=a*cos(0*deltaU);
		y=b*sin(0*deltaU);
		z=m_thickness;
		normal=calcNormal(Vec3f(x,y,z));
		pid=points->InsertNextPoint(x,y,z);
		pointNormalsArray->SetTuple(pid, &normal.X);
		vertex->GetPointIds()->SetId(vertexIndex,vertexIndex);
		vertexIndex++;

		x=a*cos(0*deltaU);
		y=b*sin(0*deltaU);
		z=0;
		normal=calcNormal(Vec3f(x,y,z));
		pid=points->InsertNextPoint(x,y,z);
		pointNormalsArray->SetTuple(pid, &normal.X);
		vertex->GetPointIds()->SetId(vertexIndex,vertexIndex);
		vertexIndex++;

		x=a*cos((0+1)*deltaU);
		y=b*sin((0+1)*deltaU);
		z=m_thickness;
		normal=calcNormal(Vec3f(x,y,z));
		pid=points->InsertNextPoint(x,y,z);
		pointNormalsArray->SetTuple(pid, &normal.X);
		vertex->GetPointIds()->SetId(vertexIndex,vertexIndex);
		vertexIndex++;

		x=a*cos((0+1)*deltaU);
		y=b*sin((0+1)*deltaU);
		z=0;
		normal=calcNormal(Vec3f(x,y,z));
		pid=points->InsertNextPoint(x,y,z);
		pointNormalsArray->SetTuple(pid, &normal.X);
		vertex->GetPointIds()->SetId(vertexIndex,vertexIndex);
		vertexIndex++;

		for (int iu=1; iu<m_renderOptions.m_slicesWidth; iu++)
		{
			x=a*cos((iu+1)*deltaU);
			y=b*sin((iu+1)*deltaU);
			z=m_thickness;
			normal=calcNormal(Vec3f(x,y,z));
			pid=points->InsertNextPoint(x,y,z);
			pointNormalsArray->SetTuple(pid, &normal.X);
			vertex->GetPointIds()->SetId(vertexIndex,vertexIndex);
			vertexIndex++;

			x=a*cos((iu+1)*deltaU);
			y=b*sin((iu+1)*deltaU);
			z=0;
			normal=calcNormal(Vec3f(x,y,z));
			pid=points->InsertNextPoint(x,y,z);
			pointNormalsArray->SetTuple(pid, &normal.X);
			vertex->GetPointIds()->SetId(vertexIndex,vertexIndex);
			vertexIndex++;
		}
		x=a;
		y=0;
		z=0;
		normal=calcNormal(Vec3f(x,y,z));
		pid=points->InsertNextPoint(x,y,z);
		pointNormalsArray->SetTuple(pid, &normal.X);
		vertex->GetPointIds()->SetId(vertexIndex,vertexIndex);
		vertexIndex++;

		// back face
		x=a*cos(0*deltaU);
		y=b*sin(0*deltaU);
		z=m_thickness;
		pid=points->InsertNextPoint(x,y,z);
		normal=Vec3f(0,0,-1);
		pointNormalsArray->SetTuple(pid, &normal.X);
		vertex->GetPointIds()->SetId(vertexIndex,vertexIndex);
		vertexIndex++;

		x=a*cos(0*deltaU);
		y=b*sin(0*deltaU);
		z=m_thickness;
		pid=points->InsertNextPoint(x,y,z);
		normal=Vec3f(0,0,-1);
		pointNormalsArray->SetTuple(pid, &normal.X);
		vertex->GetPointIds()->SetId(vertexIndex,vertexIndex);
		vertexIndex++;

		x=0;
		y=0;
		z=m_thickness;
		pid=points->InsertNextPoint(x,y,z);
		normal=Vec3f(0,0,-1);
		pointNormalsArray->SetTuple(pid, &normal.X);
		vertex->GetPointIds()->SetId(vertexIndex,vertexIndex);
		vertexIndex++;

		for (unsigned int i=1; i<=m_renderOptions.m_slicesWidth; i++)
		//for (unsigned int i=1; i<=1; i++)
		{
			x=a*cos(double(i)*deltaU);
			y=b*sin(double(i)*deltaU);
			z=m_thickness;
			pid=points->InsertNextPoint(x,y,z);
			normal=Vec3f(0,0,-1);
			pointNormalsArray->SetTuple(pid, &normal.X);
			vertex->GetPointIds()->SetId(vertexIndex,vertexIndex);
			vertexIndex++;

			x=0;
			y=0;
			z=m_thickness;
			pid=points->InsertNextPoint(x,y,z);
			normal=Vec3f(0,0,-1);
			pointNormalsArray->SetTuple(pid, &normal.X);
			vertex->GetPointIds()->SetId(vertexIndex,vertexIndex);
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