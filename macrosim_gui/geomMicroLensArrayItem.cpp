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

#include "geomMicroLensArrayItem.h"
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

MicroLensArrayItem::MicroLensArrayItem(QString name, QObject *parent) :
	GeometryItem(name, MICROLENSARRAY, parent),
		m_thickness(2),
		m_microLensPitch(2),
		m_microLensAptRad(2),
		m_microLensRadius(-2),
		m_microLensAptType(MICRORECTANGULAR)
{
	//this->setApertureType(RECTANGULAR);
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

	this->setApertureType(RECTANGULAR);
}

MicroLensArrayItem::~MicroLensArrayItem()
{
	m_childs.clear();
}

QString MicroLensArrayItem::microLensAptTypeToString(MicroLensAptType in) const
{
	QString out;
	switch (in)
	{
	case MICRORECTANGULAR:
		out="MICRORECTANGULAR";
		break;
	case MICROELLIPTICAL:
		out="MICROELLIPTICAL";
		break;
	default:
		out="MICROUNKNOWN";
		break;
	}
	return out;
}

MicroLensArrayItem::MicroLensAptType MicroLensArrayItem::stringToMicroLensAptType(QString in) const
{
	if (in.isNull())
		return MicroLensArrayItem::MICROUNKNOWN;
	if (!in.compare("MICRORECTANGULAR"))
		return MicroLensArrayItem::MICRORECTANGULAR;
	if (!in.compare("MICROELLIPTICAL"))
		return MicroLensArrayItem::MICROELLIPTICAL;

	return MicroLensArrayItem::MICROUNKNOWN;
}

bool MicroLensArrayItem::writeToXML(QDomDocument &document, QDomElement &root) const
{
	// micro lens arrays consist of three surface. front face: microstructured surface. back face: plane surface. side face: clyPipe
	QDomElement baseNode = document.createElement("geometry");
	QDomElement frontNode = document.createElement("surface");
	QDomElement backNode = document.createElement("surface");
	QDomElement sideNode1 = document.createElement("surface");

	// back face
	backNode.setAttribute("faceType", "BACKFACE");
	backNode.setAttribute("objectType", "GEOMETRY");
	backNode.setAttribute("geomType", "MICROLENSARRAYSURF");
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

	backNode.setAttribute("microLensRadius", QString::number(m_microLensRadius));
	backNode.setAttribute("microLensAptRad", QString::number(m_microLensAptRad));
	backNode.setAttribute("microLensPitch", QString::number(m_microLensPitch));
	backNode.setAttribute("microLensAptType", microLensAptTypeToString(m_microLensAptType));

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
	baseNode.setAttribute("geomType", "MICROLENSARRAY");
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

bool MicroLensArrayItem::readFromXML(const QDomElement &node)
{
	// read base class from XML
	QDomNodeList l_nodeList=node.elementsByTagName("surface");
	// spherical lense items consists of three surfaces
	if ( !((l_nodeList.count() == 3) || (l_nodeList.count() == 6)) )
	{
		cout << "error in MicroLensArrayItem.readFromXML(): item has neither three nor six child items" << endl;
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
			if (l_geomType.compare("MICROLENSARRAYSURF"))
			{
				cout << "error in MicroLensArrayItem.readFromXML(): front face is not a microLensArraySurf" << endl;
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
			m_microLensRadius=l_ele.attribute("microLensRadius").toDouble();
			m_microLensAptRad=l_ele.attribute("microLensAptRad").toDouble();
			m_microLensPitch=l_ele.attribute("microLensPitch").toDouble();
			m_microLensAptType=stringToMicroLensAptType(l_ele.attribute("microLensAptType"));
		}
		if (!l_ele.attribute("faceType").compare("FRONTFACE"))
		{
			QString l_geomType=l_ele.attribute("geomType");
			if (l_geomType.compare("PLANESURFACE"))
			{
				cout << "error in MicroLensArrayItem.readFromXML(): back face is not a plane surface" << endl;
				false;
			}
			Vec3d t_root;
			t_root.X=l_ele.attribute("root.x").toDouble();
			t_root.Y=l_ele.attribute("root.y").toDouble();
			t_root.Z=l_ele.attribute("root.z").toDouble();
			this->setRoot(t_root);
		}
		if (!l_ele.attribute("faceType").compare("SIDEFACE"))
		{
			// nothing needs to be read here....
		}
	}
	m_thickness=sqrt((l_root-this->getRoot())*(l_root-this->getRoot()));

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

void MicroLensArrayItem::render(QMatrix4x4 &m, RenderOptions &options)
{
//	if (this->getRender())
//	{
//		// apply current global transformations
//		loadGlMatrix(m);
//
//		glPushMatrix();
//
//		if (this->m_focus)
//			glColor3f(0.0f,1.0f,0.0f); //green
//		else
//			glColor3f(0.0f,0.0f,1.0f); //blue
//
//		// apply current global transform
//		Vec3d root=this->getRoot();
//		Vec2d aptRadius=this->getApertureRadius();
//		glTranslatef(root.X,root.Y,root.Z);
//		glRotatef(this->getTilt().X,1.0f,0.0f,0.0f);
//		glRotatef(this->getTilt().Y,0.0f,1.0f,0.0f);
//		glRotatef(this->getTilt().Z,0.0f,0.0f,1.0f);
//
//		Vec3f neighbours[8];
//		Vec3f normal=calcNormal(Vec3f(root.X,root.Y,root.Z),&neighbours[0],0);
//
//		if (this->getApertureType()==RECTANGULAR)
//		{
//			// render front face
//			glBegin(GL_QUADS);
//			glNormal3f(normal.X, normal.Y, normal.Z); // this normal holds to all vertices
//			float x=-aptRadius.X;
//			float y=-aptRadius.Y;
//			float z=0;
//			glVertex3f(x,y,z);
//
//			x=-aptRadius.X;
//			y=aptRadius.Y;
//			z=0;
//			glNormal3f(normal.X, normal.Y, normal.Z);
//			glVertex3f(x,y,z);
//
//			x=aptRadius.X;
//			y=aptRadius.Y;
//			z=0;
//			glNormal3f(normal.X, normal.Y, normal.Z);
//			glVertex3f(x,y,z);
//
//			x=aptRadius.X;
//			y=-aptRadius.Y;
//			z=0;
//			glNormal3f(normal.X, normal.Y, normal.Z);
//			glVertex3f(x,y,z);
//
//			glEnd();
//
//			// side face
//			//glBegin(GL_QUAD_STRIP);
//			//glNormal3f(0.0, -1.0, 0.0);
//			//x=-aptRadius.X;
//			//y=-aptRadius.Y;
//			//z=0;
//			//glVertex3f(x,y,z);
//
//			//x=-aptRadius.X;
//			//y=-aptRadius.Y;
//			//z=m_thickness;
//			//glNormal3f(0.0, -1.0, 0.0);
//			//glVertex3f(x,y,z);
//
//			//x=aptRadius.X;
//			//y=-aptRadius.Y;
//			//z=0;
//			//glNormal3f(0.0, -1.0, 0.0);
//			//glVertex3f(x,y,z);
//
//			//x=aptRadius.X;
//			//y=-aptRadius.Y;
//			//z=m_thickness;
//			//glNormal3f(0.0, -1.0, 0.0);
//			//glVertex3f(x,y,z);
//
//			//x=aptRadius.X;
//			//y=aptRadius.Y;
//			//z=0;
//			//glNormal3f(1.0, 0.0, 0.0);
//			//glVertex3f(x,y,z);
//
//			//x=aptRadius.X;
//			//y=aptRadius.Y;
//			//z=m_thickness;
//			//glNormal3f(1.0, 0.0, 0.0);
//			//glVertex3f(x,y,z);
//
//			//x=-aptRadius.X;
//			//y=aptRadius.Y;
//			//z=0;
//			//glNormal3f(0.0, 1.0, 0.0);
//			//glVertex3f(x,y,z);
//
//			//x=-aptRadius.X;
//			//y=aptRadius.Y;
//			//z=m_thickness;
//			//glNormal3f(0.0, 1.0, 0.0);
//			//glVertex3f(x,y,z);
//
//			//x=-aptRadius.X;
//			//y=-aptRadius.Y;
//			//z=0;
//			//glNormal3f(-1.0, 0.0, 0.0);
//			//glVertex3f(x,y,z);
//
//			//x=-aptRadius.X;
//			//y=-aptRadius.Y;
//			//z=m_thickness;
//			//glNormal3f(-1.0, 0.0, 0.0);
//			//glVertex3f(x,y,z);
//
//			//glEnd();
//
//			// back face
//			float sizeX=2*this->getApertureRadius().X;
//			float sizeY=2*this->getApertureRadius().Y;
//
//			float dx=sizeX/(options.m_slicesWidth);
//			float dy=sizeY/(options.m_slicesHeight);
//
//			double x0=-this->getApertureRadius().X;
//			double y0=-this->getApertureRadius().Y;
//
//			unsigned long nrOfQuads=options.m_slicesWidth*options.m_slicesHeight;
//			unsigned long nrOfVertices=(options.m_slicesWidth+1)*(options.m_slicesHeight+1);
//			unsigned long nrOfIndices=4*nrOfQuads;
//
//			GLfloat *vertices=(GLfloat*)malloc(nrOfVertices*3*sizeof(GLfloat));
//			GLfloat *normals=(GLfloat*)malloc(nrOfVertices*3*sizeof(GLfloat));
//			GLuint *indices=(GLuint*)malloc(nrOfIndices*sizeof(GLuint));
//
//			float lensHeightMax;
//			float effectiveAptRadius=min(m_microLensPitch/2,m_microLensAptRad);
//			//if (m_microLensAptType==MICRORECTANGULAR)
//			//{
//			//	float rmax=sqrt(effectiveAptRadius*effectiveAptRadius+effectiveAptRadius*effectiveAptRadius);
//			//	if (rmax>abs(m_microLensRadius))
//			//		lensHeightMax=sqrt(m_microLensRadius*m_microLensRadius-effectiveAptRadius*effectiveAptRadius);
//			//	else
//			//		lensHeightMax=sqrt(m_microLensRadius*m_microLensRadius-rmax*rmax);
//			//}
//			//else
//			//{
//				float rmax=sqrt(effectiveAptRadius*effectiveAptRadius+effectiveAptRadius*effectiveAptRadius);
//				if (rmax>abs(m_microLensRadius))
//					lensHeightMax=sqrt(m_microLensRadius*m_microLensRadius-effectiveAptRadius*effectiveAptRadius);
//				else
//					lensHeightMax=sqrt(m_microLensRadius*m_microLensRadius-rmax*rmax);
////			}
//			for (unsigned int iy=0; iy<options.m_slicesHeight+1;iy++)
//			{
//				for (unsigned int ix=0; ix<options.m_slicesWidth+1;ix++)
//				{
//					//..vertices...
//					// x-coordinate
//					x=x0+ix*dx;
//					vertices[3*ix+iy*(options.m_slicesWidth+1)*3]=x;
//					// y-coordinate
//					y=y0+iy*dy;
//					vertices[3*ix+iy*(options.m_slicesWidth+1)*3+1]=y;
//					// z-coordinate
//					float z;
//					float fac=floorf(x/m_microLensPitch+0.5);
//					float xloc=x-fac*m_microLensPitch;
//					fac=floorf(y/m_microLensPitch+0.5);
//					float yloc=y-fac*m_microLensPitch;
//					float r; // lateral distance to local centre
//					if (m_microLensAptType==MICRORECTANGULAR)
//						r=max(abs(xloc),abs(yloc));
//					else
//						r=sqrt(xloc*xloc+yloc*yloc);
//					//if ( (r>=effectiveAptRadius) || (sqrt(xloc*xloc+yloc*yloc)>=abs(m_microLensRadius)) )
//					if ( (r>=m_microLensAptRad) || (sqrt(xloc*xloc+yloc*yloc)>=abs(m_microLensRadius)) )
//					{
//						vertices[3*ix+iy*(options.m_slicesWidth+1)*3+2]=m_thickness;
//						normals[3*ix+(options.m_slicesWidth+1)*iy*3]=GLfloat(0.0f);
//						normals[3*ix+(options.m_slicesWidth+1)*iy*3+1]=GLfloat(0.0f);
//						normals[3*ix+(options.m_slicesWidth+1)*iy*3+2]=GLfloat(-1.0f);
//					}
//					else
//					{
//						z=sqrt(m_microLensRadius*m_microLensRadius-xloc*xloc-yloc*yloc);
//						float l_vecLength=sqrt(xloc*xloc+yloc*yloc+z*z);
//						normals[3*ix+(options.m_slicesWidth+1)*iy*3]=GLfloat(-xloc/l_vecLength);
//						normals[3*ix+(options.m_slicesWidth+1)*iy*3+1]=GLfloat(-yloc/l_vecLength);
//						
//						if (m_microLensRadius<0)
//						{
//							vertices[3*ix+iy*(options.m_slicesWidth+1)*3+2]=z-lensHeightMax+m_thickness;//lensHeightMax;
//							normals[3*ix+(options.m_slicesWidth+1)*iy*3+2]=GLfloat(-z/l_vecLength);
//						}
//						else
//						{
//							vertices[3*ix+iy*(options.m_slicesWidth+1)*3+2]=-z+lensHeightMax+m_thickness;//lensHeightMax;
//							normals[3*ix+(options.m_slicesWidth+1)*iy*3]=GLfloat(xloc/l_vecLength);
//							normals[3*ix+(options.m_slicesWidth+1)*iy*3+1]=GLfloat(yloc/l_vecLength);
//
//							normals[3*ix+(options.m_slicesWidth+1)*iy*3+2]=GLfloat(-z/l_vecLength);
//						}
//					}
//				}
//			}
//
//			// create indices
//			unsigned int yIdx=0;
//			unsigned int xIdx=0;
//			indices[0]=0;
//			unsigned long iQuadY=0;
//			for (unsigned long i=1; i<nrOfIndices; i++)
//			{
//				unsigned long iQuad=i%4;
//				if (iQuad<2)
//					indices[i]=xIdx+iQuad+(options.m_slicesWidth+1)*yIdx;
//				else
//					indices[i]=xIdx+(options.m_slicesWidth+1)*(yIdx+1)-iQuad+3;
//				if (iQuad==3)
//					xIdx++;
//				if ( ((i+1)%((options.m_slicesWidth)*4))==0 )
//				{
//					xIdx=0;
//					yIdx++;
//				}
//
//			}
//
//			glEnableClientState(GL_VERTEX_ARRAY);
//			glEnableClientState(GL_NORMAL_ARRAY);
//
//			glVertexPointer(3, GL_FLOAT, 0, vertices);
//			glNormalPointer(GL_FLOAT, 0, normals);
//
//			glDrawElements(GL_QUADS, nrOfIndices, GL_UNSIGNED_INT, indices);
//
//
//			delete vertices;
//			delete normals;
//			delete indices;
//		}
//		else
//		{
//			//glBegin(GL_TRIANGLE_FAN);
//			//glNormal3f(normal.X, normal.Y, normal.Z); // this normal holds to all vertices
//			//float deltaU=2*PI/options.m_slicesWidth;
//			//double a=this->getApertureRadius().X;
//			//double b=this->getApertureRadius().Y;
//			//glVertex3f(0, 0, 0);
//			//for (int i=0; i<=options.m_slicesWidth; i++)
//			//{
//			//	glNormal3f(normal.X, normal.Y, normal.Z);
//			//	glVertex3f(a*cos(-i*deltaU), b*sin(-i*deltaU), m_thickness);
//			//}
//			//glEnd();
//		}
//
//		glPopMatrix();
//	}
}

Vec3f MicroLensArrayItem::calcNormal(Vec3f vertex, Vec3f* neighbours, int nr)
{
	return Vec3f(0,0,1);
}


Vec3f MicroLensArrayItem::calcNormal(Vec3f vertex)
{
	return Vec3f(0,0,-1);
}

void MicroLensArrayItem::renderVtk(vtkSmartPointer<vtkRenderer> renderer)
{
	renderer->AddActor(m_pActor);

	this->updateVtk();
}

void MicroLensArrayItem::updateVtk()
{
	vtkSmartPointer<vtkPoints> points =  vtkSmartPointer<vtkPoints>::New();
	vtkSmartPointer<vtkDoubleArray> pointNormalsArray =  vtkSmartPointer<vtkDoubleArray>::New();
	pointNormalsArray->SetNumberOfComponents(3); //3d normals (ie x,y,z)

	vtkSmartPointer<vtkVertex> vertex = vtkSmartPointer<vtkVertex>::New();
	// Create a cell array to store the vertices
	vtkSmartPointer<vtkCellArray> cells =  vtkSmartPointer<vtkCellArray>::New();

	// calc number of vertices 
	//unsigned long numVert=13+4*4+1;//((m_renderOptions.m_slicesWidth+1)*(m_renderOptions.m_slicesHeight+1));
	//unsigned long numVert=13+4*((m_renderOptions.m_slicesWidth)*(m_renderOptions.m_slicesHeight))+1;
	unsigned long numVert=23+4*((m_renderOptions.m_slicesWidth)*(m_renderOptions.m_slicesHeight))+1;

	pointNormalsArray->SetNumberOfTuples(numVert);

	vertex->GetPointIds()->SetNumberOfIds(numVert);

	vtkIdType pid;
	unsigned long vertexIndex=0;

	Vec3d root=this->getRoot();
	Vec2d aptRadius=this->getApertureRadius();
	Vec3f normal=this->calcNormal(Vec3f());
	
	if (1)//this->getApertureType()==RECTANGULAR)
	{
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
		float lensHeightMax;
		float effectiveAptRadius=min(m_microLensPitch/2,m_microLensAptRad);
		float rmax;
		if ((this->m_microLensAptType == MICRORECTANGULAR) || (m_microLensPitch/2<m_microLensAptRad))
			rmax=sqrt(effectiveAptRadius*effectiveAptRadius+effectiveAptRadius*effectiveAptRadius);
		else
			rmax=effectiveAptRadius;	
		if (rmax>abs(m_microLensRadius))
			lensHeightMax=sqrt(m_microLensRadius*m_microLensRadius-effectiveAptRadius*effectiveAptRadius);
		else
			lensHeightMax=sqrt(m_microLensRadius*m_microLensRadius-rmax*rmax);

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
			z=this->calcZCoordinate(x, y, lensHeightMax, &normal);
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
			z=this->calcZCoordinate(x, y, lensHeightMax, &normal);
			pid=points->InsertNextPoint(x,y,z);
			normal=Vec3f(0,-1,0);
			pointNormalsArray->SetTuple(pid, &normal.X);
			vertex->GetPointIds()->SetId(vertexIndex,vertexIndex);
			vertexIndex++;

			x=-aptRadius.X;
			y=aptRadius.Y;
			z=this->calcZCoordinate(x, y, lensHeightMax, &normal);
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
			z=this->calcZCoordinate(x, y, lensHeightMax, &normal);
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
			z=this->calcZCoordinate(x, y, lensHeightMax, &normal);
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
			z=this->calcZCoordinate(x, y, lensHeightMax, &normal);
			pid=points->InsertNextPoint(x,y,z);
			normal=Vec3f(0,1,0);
			pointNormalsArray->SetTuple(pid, &normal.X);
			vertex->GetPointIds()->SetId(vertexIndex,vertexIndex);
			vertexIndex++;

			x=aptRadius.X;
			y=-aptRadius.Y;
			z=this->calcZCoordinate(x, y, lensHeightMax, &normal);
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
			z=this->calcZCoordinate(x, y, lensHeightMax, &normal);
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
			z=this->calcZCoordinate(x, y, lensHeightMax, &normal);
			pid=points->InsertNextPoint(x,y,z);
			normal=Vec3f(1,0,0);
			pointNormalsArray->SetTuple(pid, &normal.X);
			vertex->GetPointIds()->SetId(vertexIndex,vertexIndex);
			vertexIndex++;

			x=aptRadius.X;
			y=aptRadius.Y;
			z=this->calcZCoordinate(x, y, lensHeightMax, &normal);
			pid=points->InsertNextPoint(x,y,z);
			normal=Vec3f(0,-1,0);
			pointNormalsArray->SetTuple(pid, &normal.X);
			vertex->GetPointIds()->SetId(vertexIndex,vertexIndex);
			vertexIndex++;


		// back face
		x=+aptRadius.X;
		y=+aptRadius.Y;
		z=this->calcZCoordinate(x, y, lensHeightMax, &normal);
		//z=this->getThickness();
		pid=points->InsertNextPoint(x,y,z);
		normal=Vec3f(0,-1,0);
		pointNormalsArray->SetTuple(pid, &normal.X);
		vertex->GetPointIds()->SetId(vertexIndex,vertexIndex);
		vertexIndex++;

		float sizeX=2*this->getApertureRadius().X;
		float sizeY=2*this->getApertureRadius().Y;

//		m_renderOptions.m_slicesWidth=2;
//		m_renderOptions.m_slicesHeight=2;

		float dx=sizeX/(m_renderOptions.m_slicesWidth);
		float dy=sizeY/(m_renderOptions.m_slicesHeight);

		float x0=this->getApertureRadius().X;
		float y0=this->getApertureRadius().Y;

		for (unsigned int iy=0; iy<m_renderOptions.m_slicesHeight;iy++)
		{
			for (unsigned int ix=0; ix<m_renderOptions.m_slicesWidth;ix++)
			{
				if ( (iy % 2) == 0)
				{
					// we need to create a square around each of the points we created in our direct OpenGL view to get the ordering right
					// first point of the square
					float xC=this->getApertureRadius().X-dx/2-ix*dx;
					float yC=this->getApertureRadius().X-dy/2-iy*dy;
					float z;
					Vec3f normal;
					x=xC+dx/2;
					y=yC+dy/2;
					z=this->calcZCoordinate(x, y, lensHeightMax, &normal);
					pid=points->InsertNextPoint(x,y,z);
					pointNormalsArray->SetTuple(pid, &normal.X);
					vertex->GetPointIds()->SetId(vertexIndex,vertexIndex);
					vertexIndex++;

					x=xC+dx/2;
					y=yC-dy/2;
					z=this->calcZCoordinate(x, y, lensHeightMax, &normal);
					pid=points->InsertNextPoint(x,y,z);
					pointNormalsArray->SetTuple(pid, &normal.X);
					vertex->GetPointIds()->SetId(vertexIndex,vertexIndex);
					vertexIndex++;

					x=xC-dx/2;
					y=yC+dy/2;
					z=this->calcZCoordinate(x, y, lensHeightMax, &normal);
					pid=points->InsertNextPoint(x,y,z);
					pointNormalsArray->SetTuple(pid, &normal.X);
					vertex->GetPointIds()->SetId(vertexIndex,vertexIndex);
					vertexIndex++;

					x=xC-dx/2;
					y=yC-dy/2;
					z=this->calcZCoordinate(x, y, lensHeightMax, &normal);
					pid=points->InsertNextPoint(x,y,z);
					pointNormalsArray->SetTuple(pid, &normal.X);
					vertex->GetPointIds()->SetId(vertexIndex,vertexIndex);
					vertexIndex++;
				}
				else
				{
					// we need to create a square around each of the points we created in our direct OpenGL view to get the ordering right
					// first point of the square
					float xC=-this->getApertureRadius().X+dx/2+ix*dx;
					float yC=this->getApertureRadius().X-dy/2-iy*dy;
					float z;
					Vec3f normal;
					x=xC-dx/2;
					y=yC+dy/2;
					z=this->calcZCoordinate(x, y, lensHeightMax, &normal);
					normal=normal*-1;
					pid=points->InsertNextPoint(x,y,z);
					pointNormalsArray->SetTuple(pid, &normal.X);
					vertex->GetPointIds()->SetId(vertexIndex,vertexIndex);
					vertexIndex++;

					x=xC-dx/2;
					y=yC-dy/2;
					z=this->calcZCoordinate(x, y, lensHeightMax, &normal);
					normal=normal*-1;
					pid=points->InsertNextPoint(x,y,z);
					pointNormalsArray->SetTuple(pid, &normal.X);
					vertex->GetPointIds()->SetId(vertexIndex,vertexIndex);
					vertexIndex++;

					x=xC+dx/2;
					y=yC+dy/2;
					z=this->calcZCoordinate(x, y, lensHeightMax, &normal);
					normal=normal*-1;
					pid=points->InsertNextPoint(x,y,z);
					pointNormalsArray->SetTuple(pid, &normal.X);
					vertex->GetPointIds()->SetId(vertexIndex,vertexIndex);
					vertexIndex++;

					x=xC+dx/2;
					y=yC-dy/2;
					z=this->calcZCoordinate(x, y, lensHeightMax, &normal);
					normal=normal*-1;
					pid=points->InsertNextPoint(x,y,z);
					pointNormalsArray->SetTuple(pid, &normal.X);
					vertex->GetPointIds()->SetId(vertexIndex,vertexIndex);
					vertexIndex++;
				}
			}
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

#if  (VTK_MAJOR_VERSION <= 5)
	// request the update
	m_pPolydata->Update();
#else
    m_pMapper->Update();
#endif
}

float MicroLensArrayItem::calcZCoordinate(float x, float y, float lensHeightMax, Vec3f* normal)
{
	float z;
	float fac=floorf(x/m_microLensPitch+0.5);
	float xloc=x-fac*m_microLensPitch;
	fac=floorf(y/m_microLensPitch+0.5);
	float yloc=y-fac*m_microLensPitch;
	float r; // lateral distance to local centre
	if (m_microLensAptType==MICRORECTANGULAR)
		r=max(abs(xloc),abs(yloc));
	else
		r=sqrt(xloc*xloc+yloc*yloc);
	if ( (r>=m_microLensAptRad) || (sqrt(xloc*xloc+yloc*yloc)>=abs(m_microLensRadius)) )
	{
		z=m_thickness;
		*normal=Vec3f(0,0,-1);
	}
	else
	{
		z=sqrt(m_microLensRadius*m_microLensRadius-xloc*xloc-yloc*yloc);
		float l_vecLength=sqrt(xloc*xloc+yloc*yloc+z*z);
		if (m_microLensRadius<0)
		{
			z=z-lensHeightMax+m_thickness;
			*normal=Vec3f(-xloc/l_vecLength, -yloc/l_vecLength, -z/l_vecLength);
		}
		else
		{
			z=-z+lensHeightMax+m_thickness;
			*normal=Vec3f(xloc/l_vecLength, yloc/l_vecLength, -z/l_vecLength);
		}
	}
	return z;
}