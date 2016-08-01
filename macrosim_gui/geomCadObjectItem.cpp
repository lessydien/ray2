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


#include "geomCadObjectItem.h"
//#include "glut.h"
#include <vtkCell.h>
#include <vtkCellArray.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>

using namespace macrosim;

CadObjectItem::CadObjectItem(QString name, QObject *parent) :
	GeometryItem(name, CADOBJECT, parent),
		m_objFilename("filename.obj"),
		m_objFileLoaded(false),
		m_objFileChanged(false)
{
	m_pReader = vtkSmartPointer<vtkOBJReader>::New();
	this->m_pMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
	this->m_pMapper->SetInputConnection(m_pReader->GetOutputPort());

	this->m_pActor=vtkSmartPointer<vtkActor>::New();
	m_pActor->SetMapper(m_pMapper);



  //glewInit();

  //if (!glewIsSupported( "GL_VERSION_2_0 "
  //                      "GL_EXT_framebuffer_object "))
  //  {
  //    std::cout << "error in MyGraphicsScene.initGL(): " << "Unable to load the necessary extensions" << std::endl;
  //    exit(-1);
  //  }


  //glClearColor(.2f,.2f,.2f,1.f);
  //glEnable(GL_DEPTH_TEST);

  //cgContext = cgCreateContext();
  //cgGLSetDebugMode( CG_FALSE );
  //cgSetParameterSettingMode(cgContext, CG_DEFERRED_PARAMETER_SETTING);
  //cgGLRegisterStates(cgContext);
}

CadObjectItem::~CadObjectItem()
{
	m_childs.clear();
}

bool CadObjectItem::writeToXML(QDomDocument &document, QDomElement &root) const
{
	QDomElement node = document.createElement("geometry");

	// call base class method
	if (!GeometryItem::writeToXML(document, node))
		return false;

	node.setAttribute("geomType", "CADOBJECT");
	node.setAttribute("objFilename", m_objFilename);
	node.setAttribute("nrSurfacesSeq", "1");
	node.setAttribute("nrSurfacesNonSeq", "1");

	root.appendChild(node);
	return true;
}

bool CadObjectItem::readFromXML(const QDomElement &node)
{
	// read base class from XML
	if (!GeometryItem::readFromXML(node))
		return false;
	QString l_objFilename;
	l_objFilename=node.attribute("objFilename");
	this->setObjFilename(l_objFilename);

	return true;
}

void CadObjectItem::setObjFilename(const QString in)
{
	m_objFileChanged=true;
	m_objFileLoaded=true;
	QByteArray ba = in.toLocal8Bit();
	m_pReader->SetFileName(ba.data());
	m_pReader->Update();
	m_objFilename=in; 
	
	//// load object file
	//model = new nv::Model();
	//if(!model->loadModelFromFile(ba.data())) {
	//	std::cerr << "Unable to load model '" << qPrintable(m_objFilename) << "'" << std::endl;
	//	m_objFileLoaded=false;
	//}
	//else
	//{
	//	m_objFileLoaded=true;

	//	model->removeDegeneratePrims();

	////	if(stripNormals) {
	////	model->clearNormals();
	////	}

	//	model->computeNormals();

	//	model->clearTexCoords();
	//	model->clearColors();
	//	model->clearTangents();

	//	model->compileModel();

	//}
	emit itemChanged(m_index, m_index);
};

void CadObjectItem::render(QMatrix4x4 &m, RenderOptions &options)
{
	//if (m_objFileLoaded)
	//{
	//	if (this->getRender())
	//	{
	//		// apply current global transformations
	//		loadGlMatrix(m);

	//		glPushMatrix();

	//		if (this->m_focus)
	//			glColor3f(0.0f,1.0f,0.0f); //green
	//		else
	//			glColor3f(0.0f,0.0f,1.0f); //blue

	//		int vertSize=model->getCompiledVertexSize();
	//		int vertCount=model->getCompiledVertexCount();
	//		int normSize=model->getNormalSize();
	//		int normCount=model->getNormalCount();
	//		int indCount=model->getIndexCount();
	//		int test=model->getPositionCount();

	//		glEnableClientState(GL_VERTEX_ARRAY);
	//		GLfloat *vertices=(GLfloat*)malloc(vertSize*vertCount*sizeof(GLfloat));
	//		memcpy(vertices, model->getCompiledVertices(), vertSize*vertCount*sizeof(GLfloat));
	//		glVertexPointer(3, GL_FLOAT, vertSize, vertices);

	//		GLfloat *normals=(GLfloat*)malloc(normSize*normCount*sizeof(GLfloat));
	//		if (model->hasNormals())
	//		{
	//			memcpy(normals, model->getNormals(), normSize*normCount*sizeof(GLfloat));
	//			glEnableClientState(GL_NORMAL_ARRAY);
	//			glNormalPointer(GL_FLOAT, vertSize, normals);
	//		}

	//		GLuint *indices=(GLuint*)malloc(indCount*sizeof(GLuint));
	//		memcpy(indices, model->getCompiledIndices(), indCount*sizeof(GLuint)); 

	//		glDrawElements(GL_TRIANGLES, indCount, GL_UNSIGNED_INT, indices);

	//		delete vertices;
	//		delete normals;
	//		delete indices;

	//		glPopMatrix();
	//	}
	//}
}

Vec3f CadObjectItem::calcNormal(Vec3f vertex, Vec3f* neighbours, int nr)
{
	return Vec3f(0,0,1);
}


void CadObjectItem::updateVtk()
{
	//// apply root and tilt
	////m_pActor->SetOrigin(this->getRoot().X, this->getRoot().Y, this->getRoot().Z);
	//m_pActor->SetPosition(this->getRoot().X, this->getRoot().Y, this->getRoot().Z);
	//m_pActor->SetOrientation(this->getTilt().X, this->getTilt().Y, this->getTilt().Z);
	//m_pReader->Update();

	//if (this->m_focus)
	//	m_pActor->GetProperty()->SetColor(0.0,1.0,0.0); // green
	//else
	//	m_pActor->GetProperty()->SetColor(0.0,0.0,1.0); // red


 //   if (this->getRender())
 //       m_pActor->SetVisibility(1);
 //   else
 //       m_pActor->SetVisibility(0);

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
        m_pActor->GetProperty()->SetColor(0.0, 1.0, 0.0); // green
    else
        m_pActor->GetProperty()->SetColor(0.0, 0.0, 1.0); // red

#if  (VTK_MAJOR_VERSION <= 5)
    // request the update
    m_pPolydata->Update();
#else
    m_pMapper->Update();
#endif
};

void CadObjectItem::renderVtk(vtkSmartPointer<vtkRenderer> renderer)
{
//	QByteArray ba = m_objFilename.toLocal8Bit();
//	m_pReader->SetFileName(ba.data());
//
////	m_pReader->SetFileName("E:/mauch/MacroSim_In/gearMesh.obj");
////	m_pReader->Update();
//
//	if (this->getRender())
//		m_pActor->SetVisibility(1);
//	else
//		m_pActor->SetVisibility(0);
//
//	if (this->m_focus)
//		m_pActor->GetProperty()->SetColor(0.0,1.0,0.0); // green
//	else
//		m_pActor->GetProperty()->SetColor(0.0,0.0,1.0); // red

	renderer->AddActor(m_pActor);

    updateVtk();
}