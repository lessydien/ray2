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

#include "myGraphicsScene.h"
#include "abstractItem.h"
#include "geometryItem.h"
#include "fieldItem.h"
#include "detectorItem.h"
#include <QGraphicsView>
#include <QtOpenGL\qglfunctions.h>
#include "glut.h"

using namespace macrosim;

void MyGraphicsScene::modelAboutToBeReset()
{
	// remove all items
	for (int i=0;i<m_pModel->rowCount(QModelIndex());i++)
	{
		QModelIndex l_index=m_pModel->index(i,0,QModelIndex());
		AbstractItem* l_pAbstractItem=m_pModel->getItem(l_index);
//		this->removeItem(l_pAbstractItem->getGraphViewItem());
	}
};

void MyGraphicsScene::layoutChanged()
{
		
};

void MyGraphicsScene::rowsInserted(const QModelIndex &parent, int start, int end)
{
	QModelIndex l_index=m_pModel->index(start, 0, parent);
	AbstractItem* l_pAbstractItem=m_pModel->getItem(l_index);
	connect(l_pAbstractItem, SIGNAL(itemChanged(const QModelIndex &, const QModelIndex &)), m_pModel, SLOT(changeItemData(const QModelIndex &, const QModelIndex &)));
//	if (l_pAbstractItem->getGraphViewItem() != 0)
//	{
//		this->addItem(l_pAbstractItem->getGraphViewItem());
//		// select item
//		QPointF l_point=l_pAbstractItem->getGraphViewItem()->pos();
//		QPainterPath l_path;
//		l_path.addRect(l_point.x(),l_point.y(),2,2);
//		this->setSelectionArea(l_path);
//	}
//	l_pAbstractItem->setFocus(true);
//	this->setFocusItem(l_pAbstractItem->getGraphViewItem());
};

void MyGraphicsScene::rowsAboutToBeRemoved(const QModelIndex &parent, int start, int end)
{
	for (int i=start; i<=end; i++)
	{
		QModelIndex index=m_pModel->index(i, 0, parent);
		AbstractItem* l_pAbstractItem=m_pModel->getItem(index);
//		this->removeItem(l_pAbstractItem->getGraphViewItem());	
	}
};

void MyGraphicsScene::changeItemData(const QModelIndex &topLeft, const QModelIndex &bottomRight)
{
	// find rootItem that ultimately holds the item that just changed
	QModelIndex rootItemIndex=m_pModel->getRootIndex(topLeft);
	// all items in our model are AbstractItems, so we can savely cast here
	AbstractItem* l_pAbstractItem=m_pModel->getItem(rootItemIndex);
//	l_pAbstractItem->getGraphViewItem()->update();
//	emit itemDataChanged(topLeft, bottomRight);
};


void MyGraphicsScene::changeItemFocus(const QModelIndex &index)
{
	QModelIndex rootItemIndex;
	if (index != QModelIndex())
	{
		// find rootItem that ultimately holds the item that just changed
		rootItemIndex=m_pModel->getRootIndex(index);
		// all items in our model are AbstractItems, so we can savely cast here
//		AbstractItem* l_pAbstractItem=m_pModel->getItem(rootItemIndex);
//		l_pGraphItem=l_pAbstractItem->getGraphViewItem();
//		l_pGraphItem->setFocus();
//		l_pGraphItem->update();
		// selected item
//		QPointF l_point=l_pGraphItem->pos();
//		QPainterPath l_path;
//		l_path.addRect(l_point.x(),l_point.y(),2,2);
//		this->setSelectionArea(l_path);
	}
	if (rootItemIndex != m_pModel->getFocusedItemIndex())
	{
		// unselect old item
		m_focusedItemIndex=rootItemIndex;
	}
};

//void MyGraphicsScene::wheelEvent(QGraphicsSceneWheelEvent *wheelEvent)
//{
//	int delta=wheelEvent->delta();
//	
//	//QList<QGraphicsView*> l_views=this->views();
//	//QGraphicsView *l_pView=l_views.at(0);
//
//	//if (delta>0)
//	//	l_pView->scale(0.5,0.5);//(1/(delta/10),1/(delta/10));
//	//else
//	//	l_pView->scale(2,2);//(abs(delta/10),abs(delta/10));
//	//l_pView->centerOn(wheelEvent->scenePos());
//
//	if (delta>0)
//	{
//		m_rotAngleModel.X=m_rotAngleModel.X+50;
//		if (m_rotAngleModel.X>360)
//			m_rotAngleModel.X=m_rotAngleModel.X-360;
//	}
//	else
//	{
//		m_rotAngleModel.X=m_rotAngleModel.X-50;
//		if (m_rotAngleModel.X<-360)
//			m_rotAngleModel.X=m_rotAngleModel.X+360;
//	}
//
//
//	// redraw scene
//	this->update(this->sceneRect());
//
//	QGraphicsScene::wheelEvent(wheelEvent);
//}

QPointF MyGraphicsScene::pixelPosToViewPos(const QPointF& p)
{
    return QPointF(2.0 * float(p.x()) / width() - 1.0,
                   1.0 - 2.0 * float(p.y()) / height());
}

void MyGraphicsScene::mouseMoveEvent(QGraphicsSceneMouseEvent *event)
 {
     QGraphicsScene::mouseMoveEvent(event);
     if (event->isAccepted())
         return;

     //if (event->buttons() & Qt::LeftButton) {
     //    m_trackBalls[0].move(pixelPosToViewPos(event->scenePos()), m_trackBalls[0].rotation().conjugate());
     //    event->accept();
     //} else {
     //    m_trackBalls[0].release(pixelPosToViewPos(event->scenePos()), m_trackBalls[0].rotation().conjugate());
     //}

     if (event->buttons() & Qt::RightButton) {
         m_trackBalls[1].move(pixelPosToViewPos(event->scenePos()), m_trackBalls[2].rotation().conjugate());
         event->accept();
     } else {
         m_trackBalls[1].release(pixelPosToViewPos(event->scenePos()), m_trackBalls[2].rotation().conjugate());
     }

     if (event->buttons() & Qt::LeftButton) {
         m_trackBalls[2].move(pixelPosToViewPos(event->scenePos()), QQuaternion());
         event->accept();
     } else {
         m_trackBalls[2].release(pixelPosToViewPos(event->scenePos()), QQuaternion());
     }
 }

 void MyGraphicsScene::mousePressEvent(QGraphicsSceneMouseEvent *event)
 {
     QGraphicsScene::mousePressEvent(event);
     if (event->isAccepted())
         return;

     //if (event->buttons() & Qt::LeftButton) {
     //    m_trackBalls[0].push(pixelPosToViewPos(event->scenePos()), m_trackBalls[0].rotation().conjugate());
     //    event->accept();
     //}

     if (event->buttons() & Qt::RightButton) {
         m_trackBalls[1].push(pixelPosToViewPos(event->scenePos()), QQuaternion());
         event->accept();
     }

     if (event->buttons() & Qt::LeftButton) {
         m_trackBalls[2].push(pixelPosToViewPos(event->scenePos()), QQuaternion());
         event->accept();
     }
 }

 void MyGraphicsScene::mouseReleaseEvent(QGraphicsSceneMouseEvent *event)
 {
     QGraphicsScene::mouseReleaseEvent(event);
     if (event->isAccepted())
         return;

     //if (event->button() == Qt::LeftButton) {
     //    m_trackBalls[0].release(pixelPosToViewPos(event->scenePos()), m_trackBalls[0].rotation().conjugate());
     //    event->accept();
     //}

     if (event->button() == Qt::RightButton) {
         m_trackBalls[1].release(pixelPosToViewPos(event->scenePos()), m_trackBalls[1].rotation().conjugate());
         event->accept();
     }

     if (event->button() == Qt::LeftButton) {
         m_trackBalls[2].release(pixelPosToViewPos(event->scenePos()), QQuaternion());
         event->accept();
     }
 }

 void MyGraphicsScene::wheelEvent(QGraphicsSceneWheelEvent * event)
 {
     QGraphicsScene::wheelEvent(event);
     if (!event->isAccepted()) {
		 // delta is returned in eight of degrees. The standard increment for most mice is 120 units, i.e. 15 degrees
		 // here you can finetune the resolution of the zooming...
         m_zoom +=double(event->delta())/720;
		 if (m_zoom<0)
			 m_zoom = 0;
		 event->accept();
     }
 }

void MyGraphicsScene::initGL()
{

}

void MyGraphicsScene::drawBackground(QPainter *painter, const QRectF &rect)
{
	float width = float(painter->device()->width());
	float height= float(painter->device()->height());

	this->setSceneRect(0,0,width,height);

	painter->beginNativePainting();

	//glClearColor(0.7f, 0.9f, 1.0f, 1.0f);
	glClearColor(m_renderOptions.m_backgroundColor.X, m_renderOptions.m_backgroundColor.Y, m_renderOptions.m_backgroundColor.Z, 1.0f);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glEnable(GL_COLOR_MATERIAL);
	glColorMaterial(GL_FRONT_AND_BACK, GL_SPECULAR); 
	glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE); 
	glShadeModel(GL_SMOOTH);
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_FASTEST);

	glEnable(GL_BLEND);
	glEnable(GL_NORMALIZE);

	glMatrixMode(GL_MODELVIEW);

	glPushMatrix();

	// apply transformations from trackball
	QMatrix4x4 view;
	// move origin to centre of view
	view.translate(-m_trackBalls[1].translation());
	view.translate(width/2, height/2);
//	view.rotate(m_trackBalls[2].rotation()); // we dont rotate the light source
	view.scale(m_zoom);

    static GLfloat mat[16];
    const qreal *data = view.constData();
    for (int index = 0; index < 16; ++index)
        mat[index] = data[index];

    glLoadMatrixf(mat);

	glLightModelf(GL_LIGHT_MODEL_LOCAL_VIEWER, 1.0);
	glLightModelf(GL_LIGHT_MODEL_TWO_SIDE, 1.0);

	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);

	// add positioned light
	GLfloat ambient[] = {m_renderOptions.m_ambientInt, m_renderOptions.m_ambientInt, m_renderOptions.m_ambientInt, 1.0f};
	GLfloat diffuse[] = {m_renderOptions.m_diffuseInt, m_renderOptions.m_diffuseInt, m_renderOptions.m_diffuseInt, 1.0f};
	GLfloat specular[] = {m_renderOptions.m_specularInt, m_renderOptions.m_specularInt, m_renderOptions.m_specularInt, 1.0f};
	GLfloat lightPos0[] = {m_renderOptions.m_lightPos.X, m_renderOptions.m_lightPos.Y, m_renderOptions.m_lightPos.Z, 1.0f};
	glLightfv(GL_LIGHT0, GL_AMBIENT, ambient);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuse);
	glLightfv(GL_LIGHT0, GL_SPECULAR, specular);
	glLightfv(GL_LIGHT0, GL_POSITION, lightPos0);

	glPopMatrix();

	// apply transformations from trackball
	// move origin to centre of view
	view.setToIdentity(); // start with identity
	view.translate(-m_trackBalls[1].translation());
	view.translate(width/2, height/2);
	view.rotate(m_trackBalls[2].rotation());
	view.scale(m_zoom);

    data = view.constData();
    for (int index = 0; index < 16; ++index)
        mat[index] = data[index];

	glPushMatrix();
    glLoadMatrixf(mat);

	// define material properties
//	GLfloat matAmbAndDiff[]= {1.0f, 1.0f, 1.0f, 1.0f};
	glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, ambient);
	glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, diffuse);
	glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, specular);
	glMateriali(GL_FRONT_AND_BACK, GL_SHININESS, 120);

	if (m_renderOptions.m_showCoordAxes)
		drawCoordinateAxes();

	glPopMatrix();

	m_pModel->render(view, m_renderOptions);

	// draw rays if there are any
	if (m_rayPlotData.getData()->size() != 0)
	{
		// lock
		if (m_rayPlotData.getLock()->tryLockForRead())
		{
			m_rayPlotData.render(view, m_renderOptions);
			m_rayPlotData.getLock()->unlock();
		}
	}

	glPopMatrix();

	painter->endNativePainting();
}

void MyGraphicsScene::toggleCoordinateAxes()
{
	m_renderOptions.m_showCoordAxes=!m_renderOptions.m_showCoordAxes;
}

void MyGraphicsScene::drawCoordinateAxes()
{
	 // draw coordinate axis
	 glLineWidth(5);
	 glEnable(GL_NORMALIZE);
	 //x-coordinate
	 glPushMatrix();
	 glTranslatef(50, -40, 0);
	 glColor3f(1.0f, 0.0f, 0.0f);
	 glScalef(0.3f, 0.3f, 0.3f);
	 //glutStrokeCharacter(GLUT_STROKE_ROMAN, 'X');
	 glPopMatrix();

	 glBegin(GL_LINES);
	 glVertex3f(0, 0, 0);
	 glVertex3f(100, 0, 0);
	 glEnd();
	 glBegin(GL_LINES);
	 glVertex3f(90, -10, 0);
	 glVertex3f(100, 0, 0);
	 glEnd();
	 glBegin(GL_LINES);
	 glVertex3f(90, 10, 0);
	 glVertex3f(100, 00, 0);
	 glEnd();

	 //y-coordinate
	 glPushMatrix();
	 glTranslatef(-10, 70, 0);
	 glRotatef(180.0f, 0.0f, 0.0f, 1.0f);
	 glColor3f(0.0f, 1.0f, 0.0f);
	 glScalef(0.3f, 0.3f, 0.3f);
	 //glutStrokeCharacter(GLUT_STROKE_ROMAN, 'Y');
	 glPopMatrix();

	 glBegin(GL_LINES);
	 glVertex3f(0, 0, 0);
	 glVertex3f(0, 100, 0);
	 glEnd();
	 glBegin(GL_LINES);
	 glVertex3f(0, 100, 0);
	 glVertex3f(-10, 90, 0);
	 glEnd();
	 glBegin(GL_LINES);
	 glVertex3f(0, 100, 0);
	 glVertex3f(10, 90, 0);
	 glEnd();

	 //z-coordinate
	 glPushMatrix();
	 glTranslatef(0, -40, 70);
	 glRotatef(90.0f, 0.0f, 1.0f, 0.0f);
	 glColor3f(0.0f, 0.0f, 1.0f);
	 glScalef(0.3f, 0.3f, 0.3f);
	 //glutStrokeCharacter(GLUT_STROKE_ROMAN, 'Z');
	 glPopMatrix();	 

	 glBegin(GL_LINES);
	 glVertex3f(0, 0, 0);
	 glVertex3f(0, 0, 100);
	 glEnd();
	 glBegin(GL_LINES);
	 glVertex3f(0, 0, 100);
	 glVertex3f(0, -10, 90);
	 glEnd();
	 glBegin(GL_LINES);
	 glVertex3f(0, 0, 100);
	 glVertex3f(0, 10, 90);
	 glEnd();
//	 glDisable(GL_NORMALIZE);
}

void MyGraphicsScene::setXYView()
{
	for (int i=0; i<3; i++)
		m_trackBalls[i].setRotation(QQuaternion());
}

void MyGraphicsScene::setXZView()
{
	for (int i=0; i<3; i++)
		m_trackBalls[i].setRotation(QQuaternion::fromAxisAndAngle(QVector3D(1.0f, 0.0f, 0.0f),90));
}

void MyGraphicsScene::setYZView()
{
	for (int i=0; i<3; i++)
		m_trackBalls[i].setRotation(QQuaternion::fromAxisAndAngle(QVector3D(0.0f, 1.0f, 0.0f),90));
}

//void MyGraphicsScene::showRenderOptions()
//{
//	m_pRenderOptionsDialog=new RenderOptionsDialog(m_renderOptions);
//	if (QDialog::Accepted==m_pRenderOptionsDialog->exec())
//		this->m_renderOptions=m_pRenderOptionsDialog->getOptions();
//	delete m_pRenderOptionsDialog;
//	m_pRenderOptionsDialog=NULL;
////	connect(m_pRenderOptionsDialog, SIGNAL(renderOptionsDialogAccepted(RenderOptions&)), this, SLOT(acceptRenderOptions(RenderOptions&)));
//}

void MyGraphicsScene::showRenderOptions()
{
	RenderOptionsDialog *l_pDialog=new RenderOptionsDialog(m_renderOptions);
	connect(l_pDialog,SIGNAL(renderOptionsDialogAccepted(RenderOptions&)), this, SLOT(acceptRenderOptions(RenderOptions&)));
	l_pDialog->show();
}

void MyGraphicsScene::acceptRenderOptions(RenderOptions &options)
{
	this->m_renderOptions=options;
}

void RayPlotData::render(QMatrix4x4 &view, RenderOptions &options)
{
     // static to prevent glLoadMatrixf to fail on certain drivers
     static GLfloat mat[16];
     const qreal *data = view.constData();
     for (int index = 0; index < 16; ++index)
         mat[index] = data[index];
     glLoadMatrixf(mat);

	 glLineWidth(1);

	 glColor3f(1.0f, 0.0f, 0.0f);
	 for (int iRay=0; iRay<this->m_data.size(); iRay++)
	 {
		 glBegin(GL_LINE_STRIP);
		 for (int iVertex=0; iVertex<this->m_data.at(iRay).size(); iVertex++)
		 {
			 glVertex3f(this->m_data.at(iRay).at(iVertex).X, this->m_data.at(iRay).at(iVertex).Y, this->m_data.at(iRay).at(iVertex).Z);
		 }
		 glEnd();
	 }
}