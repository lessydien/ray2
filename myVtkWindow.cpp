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

#include "myVtkWindow.h"
#include "abstractItem.h"
#include "geometryItem.h"
#include "fieldItem.h"
#include "detectorItem.h"
//#include <QtOpenGL\qglfunctions.h>
//#include "glut.h"

#include <qfiledialog.h>
#include <qmovie.h>

#include <vtkPoints.h>
#include <vtkPolyLine.h>
#include <vtkPolyDataMapper.h>
#include <vtkCellArray.h>
#include <vtkCellData.h>
#include <vtkProperty.h>

#include <vtkActor.h>
#include <vtkRenderer.h>
#include <vtkRenderWindowInteractor.h>
#include <QVTKWidget.h>
#include <vtkAxesActor.h>
#include <vtkTransform.h>
#include <vtkPoints.h>
#include <vtkPolyLine.h>
#include <vtkPolyDataMapper.h>
#include <vtkCellArray.h>
#include <vtkCellData.h>
#include <vtkTIFFWriter.h>
#include <vtkPNGWriter.h>
#include <vtkJPEGWriter.h>
#include <vtkBMPWriter.h>
#include <vtkPostScriptWriter.h>
#include <vtkWindowToImageFilter.h>
#include <vtkCamera.h>
#include <vtkLight.h>
#include <vtkLightActor.h>
#include <vtkSphereSource.h>

#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkPropPicker.h>
#include <vtkObjectFactory.h>


using namespace macrosim;


// Handle mouse events
class MouseInteractorStyle_PickActorUponDOubleClick : public vtkInteractorStyleTrackballCamera
{
  public:
    static MouseInteractorStyle_PickActorUponDOubleClick* New();
    vtkTypeMacro(MouseInteractorStyle_PickActorUponDOubleClick, vtkInteractorStyleTrackballCamera);

    MouseInteractorStyle_PickActorUponDOubleClick() : m_numberOfClicks(0), m_resetPixelDistance(5) 
    { 
      this->m_previousPosition[0] = 0;
      this->m_previousPosition[1] = 0;
    }
 
    virtual void OnLeftButtonDown()
    {
     //std::cout << "Pressed left mouse button." << std::endl;
      this->m_numberOfClicks++;
      //std::cout << "m_numberOfClicks = " << this->m_numberOfClicks << std::endl;
      int pickPosition[2];
      this->GetInteractor()->GetEventPosition(pickPosition);
 
      int xdist = pickPosition[0] - this->m_previousPosition[0];
      int ydist = pickPosition[1] - this->m_previousPosition[1];
 
      this->m_previousPosition[0] = pickPosition[0];
      this->m_previousPosition[1] = pickPosition[1];
 
      int moveDistance = (int)sqrt((double)(xdist*xdist + ydist*ydist));
 
      // Reset numClicks - If mouse moved further than m_resetPixelDistance
      if(moveDistance > this->m_resetPixelDistance)
      { 
        this->m_numberOfClicks = 1;
      }
  
      if(this->m_numberOfClicks == 2)
      {
		  int* clickPos = this->GetInteractor()->GetEventPosition();
 
		  // Pick from this location.
		  vtkSmartPointer<vtkPropPicker>  picker =
			vtkSmartPointer<vtkPropPicker>::New();
		  picker->Pick(clickPos[0], clickPos[1], 0, this->GetDefaultRenderer());

//		  double* pos = picker->GetPickPosition();
//		  std::cout << "Pick position (world coordinates) is: "
//					<< pos[0] << " " << pos[1]
//					<< " " << pos[2] << std::endl;
 
//		  std::cout << "Picked actor: " << picker->GetActor() << std::endl;
		  // if we have a callback set an we selected an actor
		  if (m_p2CallbackObject && m_callbackFuncSelectItemFromRenderWin && picker->GetActor())
		  {
			  m_callbackFuncSelectItemFromRenderWin(m_p2CallbackObject, picker->GetActor());
		  }
		  this->m_numberOfClicks = 0;
      }
      // forward events
      vtkInteractorStyleTrackballCamera::OnLeftButtonDown();

    }

	void setCallbackObject(void* in) {m_p2CallbackObject=in;};
	void setCallbackSelectItemFromRenderWin(void(*in)(void* p2Object, void* p2Actor)) {m_callbackFuncSelectItemFromRenderWin=in;};
 
  private:
    unsigned int m_numberOfClicks;
    int m_previousPosition[2];
    int m_resetPixelDistance; 
	void* m_p2CallbackObject;
	void (*m_callbackFuncSelectItemFromRenderWin)(void* p2Object, void* p2Actor);
};
 
vtkStandardNewMacro(MouseInteractorStyle_PickActorUponDOubleClick);


myVtkWindow::myVtkWindow(QWidget *parent) :
	QVTKWidget(parent),
		m_pModel(0)
{
	m_renderOptions.m_slicesHeight=31;
	m_renderOptions.m_slicesWidth=31;
	m_renderOptions.m_showCoordAxes=true;
	m_renderOptions.m_renderMode=RENDER_SOLID;

	m_pVtkScene=vtkSmartPointer<vtkRenderWindow>::New();
	this->SetRenderWindow(m_pVtkScene);
	m_pRenderer =  vtkSmartPointer<vtkRenderer>::New();
	m_pVtkScene->AddRenderer(m_pRenderer);
	m_pRenderer->GradientBackgroundOn();
	m_pRenderer->SetBackground(m_renderOptions.m_backgroundColor.X, m_renderOptions.m_backgroundColor.Y, m_renderOptions.m_backgroundColor.Z); // Background color
	m_pRenderer->SetBackground2(1,1,1);
	//// add a directional light source
	//double lightPosition[3] = {0, 0, -10};
 //	// Create a light
	//double lightFocalPoint[3] = {0,0,0};

	//vtkSmartPointer<vtkLight> light = vtkSmartPointer<vtkLight>::New();
	//light->SetLightTypeToSceneLight();
	//light->SetPosition(lightPosition[0], lightPosition[1], lightPosition[2]);
	//light->SetPositional(true); // required for vtkLightActor below
	//light->SetConeAngle(10);
	//light->SetFocalPoint(lightFocalPoint[0], lightFocalPoint[1], lightFocalPoint[2]);
	//light->SetDiffuseColor(1,1,1);
	//light->SetAmbientColor(1,1,1);
	//light->SetSpecularColor(1,1,1);

	//// Display where the light is
	//vtkSmartPointer<vtkLightActor> lightActor = vtkSmartPointer<vtkLightActor>::New();
	//lightActor->SetLight(light);
	//this->getRenderer()->AddViewProp(lightActor);
 //
	//// Display where the light is focused
	//vtkSmartPointer<vtkSphereSource> lightFocalPointSphere = vtkSmartPointer<vtkSphereSource>::New();
	//lightFocalPointSphere->SetCenter(lightFocalPoint);
	//lightFocalPointSphere->SetRadius(.1);
	//lightFocalPointSphere->Update();

	//vtkSmartPointer<vtkPolyDataMapper> lightFocalPointMapper =
	//	vtkSmartPointer<vtkPolyDataMapper>::New();
	//lightFocalPointMapper->SetInputConnection(lightFocalPointSphere->GetOutputPort());
 //
	//vtkSmartPointer<vtkActor> lightFocalPointActor = vtkSmartPointer<vtkActor>::New();
	//lightFocalPointActor->SetMapper(lightFocalPointMapper);
	//lightFocalPointActor->GetProperty()->SetColor(1.0, 1.0, 1.0); //(R,G,B)
	//this->getRenderer()->AddViewProp(lightFocalPointActor);

	// An interactor
	vtkSmartPointer<vtkRenderWindowInteractor> renderWindowInteractor = vtkSmartPointer<vtkRenderWindowInteractor>::New();
	renderWindowInteractor->SetRenderWindow(m_pVtkScene);
	// interactor style
	//vtkSmartPointer<vtkInteractorStyleTrackballCamera> style =  vtkSmartPointer<vtkInteractorStyleTrackballCamera>::New(); //like paraview
	vtkSmartPointer<MouseInteractorStyle_PickActorUponDOubleClick> style = vtkSmartPointer<MouseInteractorStyle_PickActorUponDOubleClick>::New();
	style->SetDefaultRenderer(this->m_pRenderer);
	style->setCallbackObject(this);
	style->setCallbackSelectItemFromRenderWin(myVtkWindow::callbackSelectActorFromRenderWin);
	// set style
	renderWindowInteractor->SetInteractorStyle( style);

	this->drawCoordinateAxes();

	m_pVtkScene->Render();

	//this->getRenderer()->AddLight(light);
};

vtkSmartPointer<vtkRenderWindow> myVtkWindow::getScene()
{
	return this->m_pVtkScene;
}

void myVtkWindow::resetScene()
{
	m_pRenderer->RemoveAllViewProps();
	m_pRenderer->ResetCamera();
	// draw coordinate axes
	//this->drawCoordinateAxes();
	m_pVtkScene->Render();
}

void myVtkWindow::drawCoordinateAxes()
{
	m_pVtkAxesWidget = vtkSmartPointer<vtkOrientationMarkerWidget>::New();
	m_pVtkAxesWidget->SetOutlineColor( 0.9300, 0.5700, 0.1300 );
	m_pVtkAxesWidget->SetDefaultRenderer(m_pRenderer);
	m_pVtkAxesWidget->SetInteractor( m_pRenderer->GetRenderWindow()->GetInteractor() );
	vtkAxesActor* axes = vtkAxesActor::New();
	m_pVtkAxesWidget->SetOrientationMarker ( axes );
	axes->Delete();
	m_pVtkAxesWidget->SetViewport(0.0, 0.0, 0.34, 0.34 );
	m_pVtkAxesWidget->SetEnabled( 1 );
	m_pVtkAxesWidget->InteractiveOn();
};

void myVtkWindow::renderRays()
{
	this->m_rayPlotData.renderVtk(this->m_pRenderer);
	this->m_pRenderer->Render();
};

void myVtkWindow::removeRaysFromView()
{
	this->m_rayPlotData.removeFromView(this->m_pRenderer);
};

void myVtkWindow::modelAboutToBeReset()
{
	// remove all items
	for (int i=0;i<m_pModel->rowCount(QModelIndex());i++)
	{
		QModelIndex l_index=m_pModel->index(i,0,QModelIndex());
		AbstractItem* l_pAbstractItem=m_pModel->getItem(l_index);
//		this->removeItem(l_pAbstractItem->getGraphViewItem());
	}
};

void myVtkWindow::keyPressEvent(QKeyEvent *keyEvent)
{
	// throw event up the foodchain
	emit this->vtkWinKeyEvent( keyEvent);
}

void myVtkWindow::callbackSelectActorFromRenderWin(void* p2Object, void* p2Actor)
{
	myVtkWindow* l_pVtkWin=(myVtkWindow*)p2Object;
	for (int i=0; i<l_pVtkWin->m_pModel->getNrOfItems(); i++)
	{
		QModelIndex l_modelIndex=l_pVtkWin->m_pModel->getItem(i)->hasActor(p2Actor);
		if (QModelIndex() != l_modelIndex)
		{
			emit l_pVtkWin->itemFocusChanged(l_modelIndex);
			emit l_pVtkWin->changeItemSelection(l_modelIndex);
		}
	}
}

void myVtkWindow::layoutChanged()
{
		
};

void myVtkWindow::rowsInserted(const QModelIndex &parent, int start, int end)
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

void myVtkWindow::rowsAboutToBeRemoved(const QModelIndex &parent, int start, int end)
{
	for (int i=start; i<=end; i++)
	{
		QModelIndex index=m_pModel->index(i, 0, parent);
		AbstractItem* l_pAbstractItem=m_pModel->getItem(index);
//		this->removeItem(l_pAbstractItem->getGraphViewItem());	
	}
};

void myVtkWindow::changeItemData(const QModelIndex &topLeft, const QModelIndex &bottomRight)
{
	// find rootItem that ultimately holds the item that just changed
	QModelIndex rootItemIndex=m_pModel->getBaseIndex(topLeft);
	// all items in our model are AbstractItems, so we can savely cast here
	AbstractItem* l_pAbstractItem=m_pModel->getItem(rootItemIndex);
//	l_pAbstractItem->getGraphViewItem()->update();
//	emit itemDataChanged(topLeft, bottomRight);
};


void myVtkWindow::changeItemFocus(const QModelIndex &index)
{
	QModelIndex rootItemIndex;
	if (index != QModelIndex())
	{
		// find rootItem that ultimately holds the item that just changed
		rootItemIndex=m_pModel->getBaseIndex(index);
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

//void myVtkWindow::wheelEvent(QGraphicsSceneWheelEvent *wheelEvent)
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

void myVtkWindow::toggleCoordinateAxes()
{
	m_renderOptions.m_showCoordAxes=!m_renderOptions.m_showCoordAxes;
}


void myVtkWindow::setXViewUp()
{
	this->getRenderer()->GetActiveCamera()->SetViewUp(1,0,0);
}

void myVtkWindow::setYViewUp()
{
	this->getRenderer()->GetActiveCamera()->SetViewUp(0,1,0);
}

void myVtkWindow::setZViewUp()
{
	this->getRenderer()->GetActiveCamera()->SetViewUp(0,0,1);
}

//void myVtkWindow::showRenderOptions()
//{
//	m_pRenderOptionsDialog=new RenderOptionsDialog(m_renderOptions);
//	if (QDialog::Accepted==m_pRenderOptionsDialog->exec())
//		this->m_renderOptions=m_pRenderOptionsDialog->getOptions();
//	delete m_pRenderOptionsDialog;
//	m_pRenderOptionsDialog=NULL;
////	connect(m_pRenderOptionsDialog, SIGNAL(renderOptionsDialogAccepted(RenderOptions&)), this, SLOT(acceptRenderOptions(RenderOptions&)));
//}

void myVtkWindow::showRenderOptions()
{
	RenderOptionsDialog *l_pDialog=new RenderOptionsDialog(m_renderOptions);
	connect(l_pDialog,SIGNAL(renderOptionsDialogAccepted(RenderOptions&)), this, SLOT(acceptRenderOptions(RenderOptions&)));
	this->m_pRenderer->SetBackground(this->m_renderOptions.m_backgroundColor.X, this->m_renderOptions.m_backgroundColor.Y, this->m_renderOptions.m_backgroundColor.Z);

	l_pDialog->show();
}

void myVtkWindow::acceptRenderOptions(RenderOptions &options)
{
	this->m_renderOptions=options;
	this->m_pModel->setRenderOptions(m_renderOptions);

	this->m_pRenderer->SetBackground(this->m_renderOptions.m_backgroundColor.X, this->m_renderOptions.m_backgroundColor.Y, this->m_renderOptions.m_backgroundColor.Z);
	// render model with new options
	this->m_pModel->renderVtk(this->m_pRenderer);
}

void myVtkWindow::saveImage()
{
	m_pVtkAxesWidget->SetEnabled( 0 ); // disable coordinate axis, so it wont show up in the image

	QString filename = QFileDialog::getSaveFileName(this, tr("Save file"), "", tr("Image Files (*.png *.jpg *.tiff *.bmp *.ps"));

	if (filename.isEmpty())
	{
		return;
	}

	QStringList splittedStrList=filename.split(".");
	QString extension;
	if (splittedStrList.size()<2)
		extension="bmp"; // default if no extension was entered
	else
		extension=splittedStrList.at(splittedStrList.size()-1);

	vtkSmartPointer<vtkImageWriter> writer;
	if (filename.endsWith("tiff"))
	{
		writer = vtkTIFFWriter::New(); 
	}
	if (filename.endsWith("png"))
	{
		writer = vtkPNGWriter::New(); 
	}
	if (filename.endsWith("jpg"))
	{
		writer = vtkJPEGWriter::New(); 
	}
	if (filename.endsWith("bmp"))
	{
		writer = vtkBMPWriter::New(); 
	}
	if (filename.endsWith("ps"))
	{
		writer = vtkPostScriptWriter::New(); 
	}

	vtkRenderWindow *renWin = this->GetRenderWindow(); 
	vtkWindowToImageFilter *w2img = vtkWindowToImageFilter::New(); 
	w2img->SetInput(renWin); 
	writer->SetFileName(qPrintable(filename)); 
	w2img->SetMagnification(4); 
	writer->SetInputConnection(w2img->GetOutputPort()); 
	w2img->Modified(); 

	writer->Write();

	m_pVtkAxesWidget->SetEnabled( 1 ); // enable coordinate axis again
}

void RayPlotData::renderVtk(vtkSmartPointer<vtkRenderer> renderer)
{
	// Create a cell array to store the lines in and add the lines to it
	vtkSmartPointer<vtkCellArray> cells = vtkSmartPointer<vtkCellArray>::New();
	// Create a vtkPoints object and store the points in it
	vtkSmartPointer<vtkPoints> pts = vtkSmartPointer<vtkPoints>::New();
	unsigned long pointsIndex=0;

	for (unsigned int iRay=0; iRay<this->m_data.size(); iRay++)
	{
		// Create a vtkPolyLine object and store the rays in it
		vtkSmartPointer<vtkPolyLine> ray = vtkSmartPointer<vtkPolyLine>::New();
		ray->GetPointIds()->SetNumberOfIds(m_data.at(iRay).size());
		
		for (unsigned int iVertex=0; iVertex<m_data.at(iRay).size(); iVertex++)
		{
			pts->InsertNextPoint(m_data.at(iRay).at(iVertex).X,m_data.at(iRay).at(iVertex).Y,m_data.at(iRay).at(iVertex).Z);
			ray->GetPointIds()->SetId(iVertex, pointsIndex);
			pointsIndex++;
		}
		cells->InsertNextCell(ray);
	}
	// Create a polydata to store everything in
	m_pPolydata = vtkSmartPointer<vtkPolyData>::New();
	// Add the points to the dataset
	m_pPolydata->SetPoints(pts);
	// Add the lines to the dataset
	m_pPolydata->SetLines(cells);

  // Setup actor and mapper
	m_pMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
#if VTK_MAJOR_VERSION <= 5
	m_pMapper->SetInput(m_pPolydata);
#else
	mapper->SetInputData(polyData);
#endif
	m_pActor = vtkSmartPointer<vtkActor>::New();
	m_pActor->SetMapper(m_pMapper);
	// set color to red
	m_pActor->GetProperty()->SetColor(255,0,0);

	renderer->AddActor(m_pActor);
}

void RayPlotData::updateVtk(vtkSmartPointer<vtkRenderer> renderer)
{
		
};
