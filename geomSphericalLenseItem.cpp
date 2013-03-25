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

#include "geomSphericalLenseItem.h"
#include "materialItemLib.h"
#include "geometryItemLib.h"
#ifdef max
#undef max
#endif
#include <limits>

#include <iostream>
using namespace std;

using namespace macrosim;

SphericalLenseItem::SphericalLenseItem(QString name, QObject *parent) :
	GeometryItem(name, SPHERICALLENSE, parent),
	m_radius1(0),
	m_radius2(0),
	m_thickness(0),
	m_apertureRadius2(2,2)
{

}

SphericalLenseItem::~SphericalLenseItem()
{
	m_childs.clear();
}

bool SphericalLenseItem::writeToXML(QDomDocument &document, QDomElement &root) const
{


	// a spherical lense consists of three surfaces: front, back, side
	QDomElement baseNode = document.createElement("geometry");
	QDomElement frontNode = document.createElement("surface");
	QDomElement backNode = document.createElement("surface");
	QDomElement sideNode = document.createElement("surface");

	// front face
	frontNode.setAttribute("faceType", "FRONTFACE");
	frontNode.setAttribute("objectType", "GEOMETRY");
	if (m_radius1 != 0)
	{
		frontNode.setAttribute("geomType", "SPHERICALSURFACE");
		frontNode.setAttribute("radius", QString::number(m_radius1));
		frontNode.setAttribute("root.x", QString::number(this->getRoot().X));
		frontNode.setAttribute("root.y", QString::number(this->getRoot().Y));
		frontNode.setAttribute("root.z", QString::number(this->getRoot().Z));
		frontNode.setAttribute("tilt.x", QString::number(this->getTilt().X));
		frontNode.setAttribute("nrSurfacesSeq", "1");
		frontNode.setAttribute("nrSurfacesNonSeq", "1");
		frontNode.setAttribute("tilt.y", QString::number(this->getTilt().Y));
		frontNode.setAttribute("tilt.z", QString::number(this->getTilt().Z));
		frontNode.setAttribute("geometryID", QString::number(this->getGeometryID()));
		frontNode.setAttribute("apertureType", apertureTypeToString(this->getApertureType()));
		frontNode.setAttribute("apertureRadius.x", QString::number(this->getApertureRadius().X));
		frontNode.setAttribute("apertureRadius.y", QString::number(this->getApertureRadius().Y));											   
	}
	else
	{
		frontNode.setAttribute("geomType", "PLANESURFACE");
		frontNode.setAttribute("root.x", QString::number(this->getRoot().X));
		frontNode.setAttribute("root.y", QString::number(this->getRoot().Y));
		frontNode.setAttribute("root.z", QString::number(this->getRoot().Z));
		frontNode.setAttribute("tilt.x", QString::number(this->getTilt().X));
		frontNode.setAttribute("nrSurfacesSeq", "1");
		frontNode.setAttribute("nrSurfacesNonSeq", "1");
		frontNode.setAttribute("tilt.y", QString::number(this->getTilt().Y));
		frontNode.setAttribute("tilt.z", QString::number(this->getTilt().Z));
		frontNode.setAttribute("geometryID", QString::number(this->getGeometryID()));
		frontNode.setAttribute("apertureType", apertureTypeToString(this->getApertureType()));
		frontNode.setAttribute("apertureRadius.x", QString::number(this->getApertureRadius().X));
		frontNode.setAttribute("apertureRadius.y", QString::number(this->getApertureRadius().Y));											   
	}
	// add material
	// we must have exactly one material
	if (m_childs.count() != 1)
		return false;
	if (!this->getChild()->writeToXML(document, frontNode))
		return false;

	// back face
	Vec3d orientation(0,0,1);
	rotateVec3d(&orientation, this->getTilt());
	Vec3d root2=this->getRoot()+orientation*this->getThickness();
	backNode.setAttribute("faceType", "BACKFACE");
	backNode.setAttribute("objectType", "GEOMETRY");

	if (m_radius2 != 0)
	{
		backNode.setAttribute("geomType", "SPHERICALSURFACE");
		backNode.setAttribute("root.x", QString::number(root2.X));
		backNode.setAttribute("root.y", QString::number(root2.Y));
		backNode.setAttribute("root.z", QString::number(root2.Z));
		backNode.setAttribute("radius", QString::number(m_radius2));
		backNode.setAttribute("nrSurfacesSeq", "1");
		backNode.setAttribute("nrSurfacesNonSeq", "1");
		backNode.setAttribute("tilt.x", QString::number(this->getTilt().X));
		backNode.setAttribute("tilt.y", QString::number(this->getTilt().Y));
		backNode.setAttribute("tilt.z", QString::number(this->getTilt().Z));
		backNode.setAttribute("geometryID", QString::number(this->getGeometryID()));
		backNode.setAttribute("apertureType", apertureTypeToString(this->getApertureType()));
		backNode.setAttribute("apertureRadius.x", QString::number(m_apertureRadius2.X));
		backNode.setAttribute("apertureRadius.y", QString::number(m_apertureRadius2.Y));
	}
	else
	{
		backNode.setAttribute("geomType", "PLANESURFACE");
		backNode.setAttribute("root.x", QString::number(root2.X));
		backNode.setAttribute("root.y", QString::number(root2.Y));
		backNode.setAttribute("root.z", QString::number(root2.Z));
		backNode.setAttribute("tilt.x", QString::number(this->getTilt().X));
		backNode.setAttribute("nrSurfacesSeq", "1");
		backNode.setAttribute("nrSurfacesNonSeq", "1");
		backNode.setAttribute("tilt.y", QString::number(this->getTilt().Y));
		backNode.setAttribute("tilt.z", QString::number(this->getTilt().Z));
		backNode.setAttribute("geometryID", QString::number(this->getGeometryID()));
		backNode.setAttribute("apertureType", apertureTypeToString(this->getApertureType()));
		backNode.setAttribute("apertureRadius.x", QString::number(m_apertureRadius2.X));
		backNode.setAttribute("apertureRadius.y", QString::number(m_apertureRadius2.Y));											   
	}

	// add material
	if (!this->getChild()->writeToXML(document, backNode))
		return false;

	// side face
	if (this->getApertureRadius() == m_apertureRadius2)
	{
		sideNode.setAttribute("faceType", "SIDEFACE");
		sideNode.setAttribute("objectType", "GEOMETRY");
		sideNode.setAttribute("geomType", "CYLPIPE");
		sideNode.setAttribute("radius", QString::number(m_radius1));
		sideNode.setAttribute("root.x", QString::number(this->getRoot().X));
		sideNode.setAttribute("root.y", QString::number(this->getRoot().Y));
		sideNode.setAttribute("root.z", QString::number(this->getRoot().Z));
		sideNode.setAttribute("tilt.x", QString::number(this->getTilt().X));
		sideNode.setAttribute("tilt.y", QString::number(this->getTilt().Y));
		sideNode.setAttribute("tilt.z", QString::number(this->getTilt().Z));
		sideNode.setAttribute("thickness", QString::number(m_thickness));
		sideNode.setAttribute("apertureRadius.x", QString::number(m_apertureRadius2.X));
		sideNode.setAttribute("apertureRadius.y", QString::number(m_apertureRadius2.Y));
		sideNode.setAttribute("nrSurfacesSeq", "1");
		sideNode.setAttribute("nrSurfacesNonSeq", "1");
		sideNode.setAttribute("geometryID", QString::number(this->getGeometryID()));
		sideNode.setAttribute("apertureType", apertureTypeToString(this->getApertureType()));

		// add material
		if (!this->getChild()->writeToXML(document, sideNode))
			return false;
	}
	else
	{
		sideNode.setAttribute("faceType", "SIDEFACE");
		sideNode.setAttribute("objectType", "GEOMETRY");
		sideNode.setAttribute("geomType", "CONEPIPE");
		sideNode.setAttribute("radius", QString::number(m_radius1));
		sideNode.setAttribute("root.x", QString::number(this->getRoot().X));
		sideNode.setAttribute("root.y", QString::number(this->getRoot().Y));
		sideNode.setAttribute("root.z", QString::number(this->getRoot().Z));
		sideNode.setAttribute("tilt.x", QString::number(this->getTilt().X));
		sideNode.setAttribute("tilt.y", QString::number(this->getTilt().Y));
		sideNode.setAttribute("tilt.z", QString::number(this->getTilt().Z));
		sideNode.setAttribute("thickness", QString::number(m_thickness));
		sideNode.setAttribute("apertureRadius.x", QString::number(this->getApertureRadius().X));
		sideNode.setAttribute("apertureRadius.y", QString::number(this->getApertureRadius().Y));
		sideNode.setAttribute("apertureRadius2.x", QString::number(m_apertureRadius2.X));
		sideNode.setAttribute("apertureRadius2.y", QString::number(m_apertureRadius2.Y));
		sideNode.setAttribute("nrSurfacesSeq", "1");
		sideNode.setAttribute("nrSurfacesNonSeq", "1");
		sideNode.setAttribute("geometryID", QString::number(this->getGeometryID()));
		sideNode.setAttribute("apertureType", apertureTypeToString(this->getApertureType()));

		// add material
		if (!this->getChild()->writeToXML(document, sideNode))
			return false;

	}

	// write parameters inherited from base class
//	if (!this->writeToXML_Base(document, baseNode))
//		return false;
	baseNode.setAttribute("objectType", "GEOMETRY");
	baseNode.setAttribute("geomType", "SPHERICALLENSE");
	baseNode.setAttribute("nrSurfacesNonSeq", "3");
	baseNode.setAttribute("nrSurfacesSeq", "2");
	baseNode.setAttribute("name", this->getName());
	if (m_render)
		baseNode.setAttribute("render", "true");
	else
		baseNode.setAttribute("render", "false");

	baseNode.appendChild(frontNode);
	baseNode.appendChild(backNode);
	baseNode.appendChild(sideNode);

	root.appendChild(baseNode);
	return true;
}

bool SphericalLenseItem::readFromXML(const QDomElement &node)
{
	QDomNodeList l_nodeList=node.elementsByTagName("surface");
	// spherical lense items consists of three surfaces
	if (l_nodeList.count() != 3)
	{
		cout << "error in sphericalLenseItem.readFromXML(): item has not three child items" << endl;
		return false;
	}

	this->setName(node.attribute("name"));

	if (!node.attribute("render").compare("true"))
		this->setRender(true);
	else
		this->setRender(false);

	QDomElement l_ele;

	for (unsigned int i=0; i<3; i++)
	{
		l_ele=l_nodeList.at(i).toElement();
		if (!l_ele.attribute("faceType").compare("FRONTFACE"))
		{
			QString l_geomType=l_ele.attribute("geomType");
			if (!l_geomType.compare("SPHERICALSURFACE"))
				m_radius1=l_ele.attribute("radius").toDouble();
			else
				m_radius1=0;
			Vec3d l_tilt;
			l_tilt.X=l_ele.attribute("tilt.x").toDouble();
			l_tilt.Y=l_ele.attribute("tilt.y").toDouble();
			l_tilt.Z=l_ele.attribute("tilt.z").toDouble();
			this->setTilt(l_tilt);
			Vec3d l_root;
			l_root.X=l_ele.attribute("root.x").toDouble();
			l_root.Y=l_ele.attribute("root.y").toDouble();
			l_root.Z=l_ele.attribute("root.z").toDouble();
			this->setRoot(l_root);
			int l_geometryID=l_ele.attribute("geometryID").toDouble();
			this->setGeometryID(l_geometryID);
//			GeometryItem::ApertureType l_at;
			ApertureType l_at;
			l_at=stringToApertureType(l_ele.attribute("apertureType"));
			Vec2d l_apertureRadius;
			l_apertureRadius.X=l_ele.attribute("apertureRadius.x").toDouble();
			l_apertureRadius.Y=l_ele.attribute("apertureRadius.y").toDouble();
			this->setApertureRadius(l_apertureRadius);

		}
		if (!l_ele.attribute("faceType").compare("BACKFACE"))
		{
			m_apertureRadius2.X=l_ele.attribute("apertureRadius.x").toDouble();
			m_apertureRadius2.Y=l_ele.attribute("apertureRadius.y").toDouble();	
			QString l_geomType=l_ele.attribute("geomType");
			if (!l_geomType.compare("SPHERICALSURFACE"))
				m_radius2=l_ele.attribute("radius").toDouble();
			else
				m_radius2=0;
			m_radius2=l_ele.attribute("radius").toDouble();
			Vec3d l_root;
			l_root.X=l_ele.attribute("root.x").toDouble();
			l_root.Y=l_ele.attribute("root.y").toDouble();
			l_root.Z=l_ele.attribute("root.z").toDouble();
			m_thickness=sqrt((l_root-this->getRoot())*(l_root-this->getRoot()));
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

void SphericalLenseItem::render(QMatrix4x4 &m, RenderOptions &options)
{
	if (this->getRender())
	{
		loadGlMatrix(m);

		glPushMatrix();

		if (this->m_focus)
			glColor4f(0.0f,1.0f,0.0f,0.6f); //green
		else
			glColor4f(0.0f,0.0f,1.0f,0.6f); //blue

		// apply current global transform
		glTranslatef(this->getRoot().X,this->getRoot().Y,this->getRoot().Z);
		glRotatef(this->getTilt().X,1.0f,0.0f,0.0f);
		glRotatef(this->getTilt().Y,0.0f,1.0f,0.0f);
		glRotatef(this->getTilt().Z,0.0f,0.0f,1.0f);

		if (this->getApertureType() == RECTANGULAR)
		{
			// ??????????
		}
		else
		{
			// front face
			double ar=this->getApertureRadius().X;
			double r=-this->getRadius1();

			// thickness gives the distance of the middle points of the spherical side faces
			// the sidewall has to be longer or shorter than that according to the sign of the radius of curvature...
			double deltaZ1=abs(r)-sqrt(r*r-ar*ar);

			// radius=0 marks radius of infinity...
			if (r==0)
			{
				r=std::numeric_limits<double>::max();
				deltaZ1=0;
			}

			double deltaU=2*PI/(options.m_slicesWidth);
			double deltaV;

			if (ar>=abs(r))
			{
				// if aperture is bigger than radius, we need to draw the full semi-sphere
				deltaV=PI/2/options.m_slicesHeight;
				deltaZ1=r;
			}
			else
				// if not, we only draw part of the hemi-sphere 
				deltaV=asin(ar/r)/options.m_slicesHeight;

			// the sign of the extra length of the side face depends on the sign of the radius of curvature
			if (r<0)
				deltaZ1=-deltaZ1;

			if (ar>=abs(r))
				// if aperture is bigger than radius, we need to draw the full semi-sphere
				deltaV=PI/2/options.m_slicesHeight;
			else
				// if not, we only draw part of the hemi-sphere
				deltaV=asin(ar/r)/options.m_slicesHeight;

			//for (float phi=0; phi <= PI/2; phi+=factor)
			for (int iv=0; iv<options.m_slicesHeight; iv++)
			{
				Vec3f neighbours[8];

				double phi=0+iv*deltaV;
				glBegin(GL_TRIANGLE_STRIP);

				float x=r*sin(phi)*cos(0.0f);
				float y=r*sin(phi)*sin(0.0f);
				float z=r*cos(phi)-r;
				// set vertex and normal
				Vec3f normal=calcNormal(Vec3f(x,y,z), &neighbours[0], 0);
				glNormal3f(normal.X, normal.Y, normal.Z);
				glVertex3f(x,y,z);
				//glVertex3f(x,y,z);

				x=r*sin(phi+deltaV)*cos(0.0f);
				y=r*sin(phi+deltaV)*sin(0.0f);
				z=r*cos(phi+deltaV)-r;
				// set vertex and normal
				normal=calcNormal(Vec3f(x,y,z), &neighbours[0], 0);
				glNormal3f(normal.X, normal.Y, normal.Z);
				glVertex3f(x,y,z);

				x=r*sin(phi)*cos(deltaU);
				y=r*sin(phi)*sin(deltaU);
				z=r*cos(phi)-r;
				// set vertex and normal
				normal=calcNormal(Vec3f(x,y,z), &neighbours[0], 0);
				glNormal3f(normal.X, normal.Y, normal.Z);
				glVertex3f(x,y,z);

				x=r*sin(phi+deltaV)*cos(deltaU);
				y=r*sin(phi+deltaV)*sin(deltaU);
				z=r*cos(phi+deltaV)-r;
				// set vertex and normal
				normal=calcNormal(Vec3f(x,y,z), &neighbours[0], 0);
				glNormal3f(normal.X, normal.Y, normal.Z);
				glVertex3f(x,y,z);

				//for (float theta=factor; theta <= 2*PI; theta+= factor)
				for (int iu=1; iu<=options.m_slicesWidth; iu++)
				{
					double theta=0+iu*deltaU;
					x=r*sin(phi)*cos(theta);
					y=r*sin(phi)*sin(theta);
					z=r*cos(phi)-r;
					// set vertex and normal
					normal=calcNormal(Vec3f(x,y,z), &neighbours[0], 0);
					glNormal3f(normal.X, normal.Y, normal.Z);
					glVertex3f(x, y, z);

					x=r*sin(phi+deltaV)*cos(theta);
					y=r*sin(phi+deltaV)*sin(theta);
					z=r*cos(phi+deltaV)-r;
					// set vertex and normal
					normal=calcNormal(Vec3f(x,y,z), &neighbours[0], 0);
					glNormal3f(normal.X, normal.Y, normal.Z);
					glVertex3f(x, y, z);
				}


				glEnd();
			}

			// back face
			r=-this->getRadius2();
			ar=this->getApertureRadius2().X;

			// thickness gives the distance of the middle points of the spherical side faces
			// the sidewall has to be longer or shorter than that according to the sign of the radius of curvature...
			double deltaZ2=abs(r)-sqrt(r*r-ar*ar);

			// radius=0 marks radius of infinity...
			if (r==0)
			{
				r=std::numeric_limits<double>::max();
				deltaZ2=0;
			}

			if (ar>=abs(r))
			{
				// if aperture is bigger than radius, we need to draw the full semi-sphere
				deltaV=PI/2/options.m_slicesHeight;
				deltaZ2=r;
			}
			else
				// if not, we only draw part of the hemi-sphere 
				deltaV=asin(ar/r)/options.m_slicesHeight;

			// the sign of the extra length of the side face depends on the sign of the radius of curvature
			if (r>0)
				deltaZ2=-deltaZ2;

			for (int iv=0; iv<options.m_slicesHeight; iv++)
			//for (int iv=0; iv<1; iv++)
			{
				Vec3f neighbours[8];

				double phi=0+iv*deltaV;
				glBegin(GL_TRIANGLE_STRIP);
				float x=r*sin(phi)*cos(0.0f);
				float y=r*sin(phi)*sin(0.0f);
				float z=r*cos(phi)-r+this->getThickness();
				// set vertex and normal
				Vec3f normal=calcNormalBack(Vec3f(x,y,z));
				glNormal3f(normal.X, normal.Y, normal.Z);
				glVertex3f(x,y,z);

				x=r*sin(phi+deltaV)*cos(0.0f);
				y=r*sin(phi+deltaV)*sin(0.0f);
				z=r*cos(phi+deltaV)-r+this->getThickness();
				// set vertex and normal
				normal=calcNormalBack(Vec3f(x,y,z));
				glNormal3f(normal.X, normal.Y, normal.Z);
				glVertex3f(x,y,z);

				x=r*sin(phi)*cos(deltaU);
				y=r*sin(phi)*sin(deltaU);
				z=r*cos(phi)-r+this->getThickness();
				// set vertex and normal
				normal=calcNormalBack(Vec3f(x,y,z));
				glNormal3f(normal.X, normal.Y, normal.Z);
				glVertex3f(x,y,z);

				x=r*sin(phi+deltaV)*cos(deltaU);
				y=r*sin(phi+deltaV)*sin(deltaU);
				z=r*cos(phi+deltaV)-r+this->getThickness();
				// set vertex and normal
				normal=calcNormalBack(Vec3f(x,y,z));
				glNormal3f(normal.X, normal.Y, normal.Z);
				glVertex3f(x,y,z);

				for (int iu=1; iu<=options.m_slicesWidth; iu++)
				//for (int iu=1; iu<=1; iu++)
				{
					double theta=0+iu*deltaU;
					x=r*sin(phi)*cos(theta);
					y=r*sin(phi)*sin(theta);
					z=r*cos(phi)-r+this->getThickness();
					// set vertex and normal
					normal=calcNormalBack(Vec3f(x,y,z));
					glNormal3f(normal.X, normal.Y, normal.Z);
					glVertex3f(x, y, z);

					x=r*sin(phi+deltaV)*cos(theta);
					y=r*sin(phi+deltaV)*sin(theta);
					z=r*cos(phi+deltaV)-r+this->getThickness();
					// set vertex and normal
					normal=calcNormalBack(Vec3f(x,y,z));
					glNormal3f(normal.X, normal.Y, normal.Z);
					glVertex3f(x, y, z);
				}

				glEnd();
			}

			// side face
			deltaU=2*PI/(options.m_slicesWidth);
			deltaV=this->getThickness()+deltaZ2;
			Vec2d r1=this->getApertureRadius();
			Vec2d r2=this->getApertureRadius2();;

			if ( (r2.X>0) && (r2.Y>0) )
			{
				Vec3f neighbours[8];

				glBegin(GL_TRIANGLE_STRIP);

				float x, y, z;
				x=r2.X*cos(0*deltaU);
				y=r2.Y*sin(0*deltaU);
				z=deltaV;
				Vec3f normal=calcNormalSide(Vec3f(x,y,z));
				glNormal3f(normal.X, normal.Y, normal.Z);
				glVertex3f(x, y, z);

				x=r1.X*cos(0*deltaU);
				y=r1.Y*sin(0*deltaU);
				z=-deltaZ1;
				normal=calcNormalSide(Vec3f(x,y,z));
				glNormal3f(normal.X, normal.Y, normal.Z);
				glVertex3f(x, y, z);

				x=r2.X*cos((0+1)*deltaU);
				y=r2.Y*sin((0+1)*deltaU);
				z=deltaV;
				normal=calcNormalSide(Vec3f(x,y,z));
				glNormal3f(normal.X, normal.Y, normal.Z);
				glVertex3f(x, y, z);

				x=r1.X*cos((0+1)*deltaU);
				y=r1.Y*sin((0+1)*deltaU);
				z=-deltaZ1;
				normal=calcNormalSide(Vec3f(x,y,z));
				glNormal3f(normal.X, normal.Y, normal.Z);
				glVertex3f(x, y, z);

				for (int iu=1; iu<options.m_slicesWidth; iu++)
				{
					x=r2.Y*cos((iu+1)*deltaU);
					y=r2.Y*sin((iu+1)*deltaU);
					z=deltaV;
					normal=calcNormalSide(Vec3f(x,y,z));
					glNormal3f(normal.X, normal.Y, normal.Z);
					glVertex3f(x, y, z);

					x=r1.X*cos((iu+1)*deltaU);
					y=r1.X*sin((iu+1)*deltaU);
					z=-deltaZ1;
					normal=calcNormalSide(Vec3f(x,y,z));
					glNormal3f(normal.X, normal.Y, normal.Z);
					glVertex3f(x, y, z);
				}
				glEnd();

			}
		}

		glPopMatrix();
	}
}

Vec3f SphericalLenseItem::calcNormal(Vec3f vertex, Vec3f* neighbours, int nr)
{
	// calc centre of sphere
	Vec3f orientation=Vec3f(0,0,1);
	Vec3f centre=Vec3f(0,0,0)+orientation*this->getRadius1();
	Vec3f normal=vertex-centre;
	// normalize
	normal=normal/(sqrt(normal*normal));
	if (this->getRadius1() <0)
		normal=normal*-1;
	return normal;
}

Vec3f SphericalLenseItem::calcNormalSide(Vec3f vertex)
{
	// tangens of opening angle of cone
	double tanTheta=(m_apertureRadius2.X-this->getApertureRadius().X)/this->getThickness();
	// distance of vertx to middle axis
	double r=sqrt(vertex.X*vertex.X+vertex.Y*vertex.Y);
	Vec3f normal=vertex-Vec3f(0, 0, vertex.Z+tanTheta*r);
	return normal/(sqrt(normal*normal))*-1;
}

Vec3f SphericalLenseItem::calcNormalBack(Vec3f vertex)
{
	// calc centre of sphere
	Vec3f orientation=Vec3f(0,0,1);
	Vec3f centre=Vec3f(0,0,0)+orientation*this->getThickness()+orientation*this->getRadius2();
	Vec3f normal=vertex-centre;
	// normalize
	normal=normal/(sqrt(normal*normal));
	if (this->getRadius2() <0)
		normal=normal*-1;
	return normal;
}