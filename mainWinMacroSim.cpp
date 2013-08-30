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

#include "mainWinMacroSim.h"
#include "abstractItem.h"
#include "QPropertyEditor/CustomTypes.h"
#include "macrosim_scenemodel.h"
#include "MaterialItem.h"
#include "geometryItemLib.h"
#include "fieldItemLib.h"
#include "detectorItemLib.h"
#include "fieldLibraryContainer.h"
#include "detectorLibraryContainer.h"
#include "geometryLibraryContainer.h"
#include "miscLibraryContainer.h"
#include "geomGroupItem.h"
#include <cstring>
#include "TracerThread.h"
#include "materialItemLib.h"
#include "coatingItemLib.h"
#include "scatterItemLib.h"
#include "miscItemLib.h"
#include "oGLdrawWidget.h"
#include "qcoreapplication.h"

#include <qdir.h>

//#include "testFile.h"

//#include <qglobal.h>
#include <QGLWidget>


//#include <QXmlStreamWriter>
#include <QtXml>
#include <QFile>


#include <iostream>
using namespace std;

using namespace macrosim;

MainWinMacroSim::MainWinMacroSim(QWidget *parent) :
	QMainWindow(parent, Qt::Widget),
	m_pDockWidget_PropEditor(NULL),
	m_pItemPropertyWidget(NULL),
	m_pDockWidget_SceneTreeView(NULL),
	m_pSceneTreeView(NULL),
	m_pLibraryModel_TreeView(NULL),
	m_pDockWidget_Console(NULL),
	m_pDockWidget_LibraryView(NULL),
	m_pFileMenu(NULL),
	m_pSimMenu(NULL),
	m_pFileToolBar(NULL),
	m_pSimToolBar(NULL),
	m_pTracer(NULL),
	m_pProgBar(NULL),
	m_pTracerStatOutStream(NULL),
	m_activeItemIndex(QModelIndex()),
//	m_pScene(NULL),
	m_pTracerThreadThread(NULL),
	m_pResultField(NULL),
	m_pQVTKWidget(NULL)
{
	ui.setupUi(this);

	m_pSceneModel = new SceneModel(this);

	// create openGl widget and set it as viewport for our graphicsView
	//OGLdrawWidget *l_pWidget = new OGLdrawWidget();
	//l_pWidget->makeCurrent();

	//m_pScene = new MyGraphicsScene(this);
	//m_pScene->setModel(m_pSceneModel);

	m_pQVTKWidget=new myVtkWindow(this);
	m_pQVTKWidget->setModel(m_pSceneModel);
	this->setCentralWidget(m_pQVTKWidget);

	m_pQVTKWidget->getScene()->SetAlphaBitPlanes(1);
	m_pQVTKWidget->getScene()->SetMultiSamples(0);
	m_pQVTKWidget->getRenderer()->SetUseDepthPeeling(1);
	m_pQVTKWidget->getRenderer()->SetMaximumNumberOfPeels(100);
	m_pQVTKWidget->getRenderer()->SetOcclusionRatio(0.1);

	// create text edit in a dock widget
	m_pDockWidget_Console = new dockWidget_Console("tracer status out", this);
	this->addDockWidget(Qt::BottomDockWidgetArea, m_pDockWidget_Console);
	
	// create PropertyEditor in dock widget
	m_pDockWidget_PropEditor = new QDockWidget("MacroSim Property Editor", this);
	this->addDockWidget(Qt::BottomDockWidgetArea, m_pDockWidget_PropEditor);
	m_pItemPropertyWidget = new QPropertyEditorWidget(m_pDockWidget_PropEditor);
	// register Vec3f
	CustomTypes::registerTypes();
	m_pItemPropertyWidget->registerCustomPropertyCB(CustomTypes::createCustomProperty);
	// add items
	m_pDockWidget_PropEditor->setWidget(m_pItemPropertyWidget);


	// create libraryModel
	m_pLibraryModel = new LibraryModel(this); // library model is of the same type as the scene model, it just contains all items available...
	// now fill the model
	FieldLibContainer *l_pFieldContainer=new FieldLibContainer();
	GeometryLibContainer *l_pGeomContainer=new GeometryLibContainer();
	DetectorLibContainer *l_pDetContainer=new DetectorLibContainer();
	MiscLibContainer	*l_pMiscContainer=new MiscLibContainer();
	m_pLibraryModel->appendItem(l_pFieldContainer, NULL);
	m_pLibraryModel->appendItem(l_pGeomContainer, NULL);
	m_pLibraryModel->appendItem(l_pDetContainer, NULL);
	m_pLibraryModel->appendItem(l_pMiscContainer, NULL);

	// create LibraryModel tree view in dock widget
	m_pDockWidget_LibraryView = new QDockWidget("MacroSim Library", this);
	this->addDockWidget(Qt::LeftDockWidgetArea, m_pDockWidget_LibraryView);
	m_pLibraryModel_TreeView = new QTreeView(m_pDockWidget_LibraryView);
	m_pLibraryModel_TreeView->setItemsExpandable(false);
	// connect model to view
	m_pLibraryModel_TreeView->setModel(m_pLibraryModel);
	// expand field, geometry and detector containers
	QModelIndex l_modelIndex=m_pLibraryModel->index(0,0,QModelIndex());
	m_pLibraryModel_TreeView->setExpanded(l_modelIndex,true);
	l_modelIndex=m_pLibraryModel->index(1,0,QModelIndex());
	m_pLibraryModel_TreeView->setExpanded(l_modelIndex,true);
	l_modelIndex=m_pLibraryModel->index(2,0,QModelIndex());
	m_pLibraryModel_TreeView->setExpanded(l_modelIndex,true);
	l_modelIndex=m_pLibraryModel->index(3,0,QModelIndex());
	m_pLibraryModel_TreeView->setExpanded(l_modelIndex,true);

	//m_pLibraryModel_TreeView->setColumnHidden(1, true);

	m_pDockWidget_LibraryView->setWidget(m_pLibraryModel_TreeView);

	// attach m_pSceneModel to treeView
	m_pDockWidget_SceneTreeView = new QDockWidget("Scene Treeview", this);
	this->addDockWidget(Qt::RightDockWidgetArea, m_pDockWidget_SceneTreeView);
	m_pSceneTreeView = new QTreeView(m_pDockWidget_SceneTreeView);
	m_pSceneTreeView->setModel(m_pSceneModel);
	m_pDockWidget_SceneTreeView->setWidget(m_pSceneTreeView);


	// connect signals
	bool test;
	test=connect(m_pLibraryModel_TreeView, SIGNAL(doubleClicked(QModelIndex)), this, SLOT(addItemToScene(QModelIndex)));

	// wire sceneModel with our graphicsView, i.e. our scene
	test=connect(m_pSceneModel, SIGNAL(modelReset()), this->m_pQVTKWidget, SLOT(clear()));
	test=connect(m_pSceneModel, SIGNAL(layoutChanged()), m_pQVTKWidget, SLOT(layoutChanged()));
	test=connect(m_pSceneModel, SIGNAL(rowsInserted(const QModelIndex &, int, int)), m_pQVTKWidget, SLOT(rowsInserted(const QModelIndex &, int, int)));
	test=connect(m_pSceneModel, SIGNAL(rowsAboutToBeRemoved(const QModelIndex &, int, int)), m_pQVTKWidget, SLOT(rowsAboutToBeRemoved(const QModelIndex &, int, int)));
	test=connect(m_pSceneModel, SIGNAL(dataChanged(const QModelIndex &, const QModelIndex &)), m_pQVTKWidget, SLOT(changeItemData(const QModelIndex &, const QModelIndex &)));
	// as the sceneModel does not know the index in the PropEditor-model of the property that changed, we need to take a detour through a mainWinFunction and simply push the item to the propertyEditor again
	test=connect(m_pSceneModel, SIGNAL(dataChanged(const QModelIndex &, const QModelIndex &)), this, SLOT(sceneDataChangedFromPropEdit(const QModelIndex &, const QModelIndex &)));
	//	test=connect(m_pSceneModel, SIGNAL(dataChanged(const QModelIndex &, const QModelIndex &)), m_pSceneTreeView, SLOT(dataChanged(QModelIndex &, const QModelIndex &)));

	// change of item focus
	test=connect(m_pSceneTreeView,SIGNAL(doubleClicked(QModelIndex)), m_pSceneModel, SLOT(changeItemFocus(QModelIndex)));
	test=connect(m_pSceneTreeView,SIGNAL(doubleClicked(QModelIndex)), this, SLOT(pushItemToPropertyEditor(QModelIndex)));
	test=connect(m_pQVTKWidget,SIGNAL(itemFocusChanged(QModelIndex)), m_pSceneModel, SLOT(changeItemFocus(QModelIndex)));
	test=connect(m_pQVTKWidget,SIGNAL(itemFocusChanged(QModelIndex)), this, SLOT(pushItemToPropertyEditor(QModelIndex)));
	test=connect(m_pQVTKWidget,SIGNAL(changeItemSelection(const QModelIndex &)), m_pSceneTreeView, SLOT(setCurrentIndex(const QModelIndex &)));
	test=connect(m_pQVTKWidget,SIGNAL(vtkWinKeyEvent(QKeyEvent *)), this, SLOT(processKeyEvent(QKeyEvent *)));
	// as the graphicsView is not able to pass the model index of the newly selected item, we need to take a detour through a mainWin Function...
//	test=connect(m_pScene, SIGNAL(selectionChanged()), this, SLOT(sceneSelectionChanged()));
//	test=connect(this, SIGNAL(signalSceneSelectionChanged(const QModelIndex)), m_pSceneModel, SLOT(changeItemFocus(const QModelIndex)));
	// tell propertyEditor and GraphicsView about the change
	//test=connect(m_pSceneModel, SIGNAL(itemFocusChanged(QModelIndex)), this, SLOT(pushItemToPropertyEditor(QModelIndex)));
	test=connect(m_pSceneModel, SIGNAL(itemFocusChanged(QModelIndex)), m_pQVTKWidget, SLOT(changeItemFocus(QModelIndex)));

	// item data changed
	// as the propertyEditor does not know the index of its item in the sceneModel, we need to take a detour through a mainWinFunction and the focusedItemIndex
	test=connect(this, SIGNAL(signalSceneDataChangedFromPropEdit(const QModelIndex &, const QModelIndex &)), m_pSceneModel, SLOT(changeItemData(const QModelIndex &, const QModelIndex &)));

	test=connect(this, SIGNAL(signalSaveImage()), m_pQVTKWidget, SLOT(saveImage()));
	test=connect(m_pQVTKWidget, SIGNAL(saveImageDone()), this, SLOT(saveImageDone()));

	createMenus();
	createToolBars();
	(void*)statusBar();

	statusBar()->showMessage("started up succesfully");

	// create thread for actual tracer
	m_pTracerThreadThread = new QThread();

	this->showMaximized(); // maximize window
}

MainWinMacroSim::~MainWinMacroSim()
{
	if (m_pTracerThreadThread != NULL)
	{
		m_pTracerThreadThread->quit();
		m_pTracerThreadThread->wait();
		delete m_pTracerThreadThread;
		m_pTracerThreadThread=NULL;
	}
	if (m_pResultField != NULL)
	{
		delete m_pResultField;
		m_pResultField=NULL;
	}
}

void MainWinMacroSim::processKeyEvent(QKeyEvent *keyEvent)
{
	// if we have an item focused and hit a delete button
	QModelIndex l_index=m_pSceneModel->getFocusedItemIndex();
	if ( l_index!=QModelIndex() && ( (keyEvent->key()==Qt::Key_Escape) || (keyEvent->key()==Qt::Key_Delete) || (keyEvent->key()==Qt::Key_Clear) || (keyEvent->key()==Qt::Key_Backspace) ) )
	{
		QMessageBox::StandardButton ret;
		ret=QMessageBox::warning(this, tr("MacroSim"), tr("Do you really want to delete the selected item?"),QMessageBox::Yes | QMessageBox::No);

		QModelIndex l_rootIndex=this->m_pSceneModel->getRootIndex(l_index);
		if (ret == QMessageBox::Yes)
		{
			// if we have to remove a geometry, we need to expand the geometryGroup again, to hide errors of our model in the tree view...
			AbstractItem* l_pAbstractItem=reinterpret_cast<AbstractItem*>(l_index.internalPointer());
			l_pAbstractItem->removeFromView(this->m_pQVTKWidget->getRenderer());
			//this->m_pQVTKWidget->getRenderer()->RemoveActor(l_pAbstractItem->getActor());
			//l_pAbstractItem->setRender(false);
			m_pSceneModel->removeFocusedItem();
			if (l_pAbstractItem->getObjectType() == AbstractItem::GEOMETRY)
			{
				m_pSceneTreeView->setExpanded( l_rootIndex ,false);
				m_pSceneTreeView->setExpanded( l_rootIndex ,true);
			}

		}
	}

}

void MainWinMacroSim::keyPressEvent(QKeyEvent *keyEvent)
{
	this->processKeyEvent(keyEvent);
	// throw event up the foodchain
	QMainWindow::keyPressEvent(keyEvent);
}

//----------------------------------------------------------------------------------------------------------------------------------

//! slot method connected to pushItemToPropertyEditor
/*!

    \return void
*/
void MainWinMacroSim::pushItemToPropertyEditor(const QModelIndex index)
{
	m_pItemPropertyWidget->setObject(m_pSceneModel->getItem(index));
	this->m_pQVTKWidget->setItemFocus(index);
	m_activeItemIndex=index;
	this->m_pQVTKWidget->getScene()->Render();
}

//! slot method 
/*!
	adds another item to our SceneModel

    \return void
*/
void MainWinMacroSim::addItemToScene(const QModelIndex index)
{
	AbstractItem *l_pAbstractItem=NULL;
	l_pAbstractItem=reinterpret_cast<AbstractItem*>(index.internalPointer());
	AbstractItem::ObjectType l_objType=l_pAbstractItem->getObjectType();
	
	AbstractItem *l_pNewAbstractItem=NULL;
	FieldItem *l_pField;
	GeometryItem *l_pGeom;
	MiscItem *l_pMiscItem;
	DetectorItem *l_pDet;
	FieldItemLib l_fieldLib;
	GeometryItemLib l_geomLib;
	DetectorItemLib l_detLib;
	MiscItemLib l_miscItemLib;
	bool test;
	switch (l_objType)
	{
	case AbstractItem::FIELD:
		l_pField=reinterpret_cast<FieldItem*>(l_pAbstractItem);
		l_pNewAbstractItem=l_fieldLib.createField(l_pField->getFieldType());
		break;

	case AbstractItem::GEOMETRY:
		l_pGeom=reinterpret_cast<GeometryItem*>(l_pAbstractItem);
		l_pNewAbstractItem=l_geomLib.createGeometry(l_pGeom->getGeomType());
		this->m_pActorList.append(l_pGeom->getActor());
		break;

	case AbstractItem::DETECTOR:
		l_pDet=reinterpret_cast<DetectorItem*>(l_pAbstractItem);
		l_pNewAbstractItem=l_detLib.createDetector(l_pDet->getDetType());
		l_pDet=reinterpret_cast<DetectorItem*>(l_pNewAbstractItem);
		test=connect(this, SIGNAL(simulationFinished(ito::DataObject)), l_pDet, SLOT(simulationFinished(ito::DataObject)));
		break;

	case AbstractItem::MISCITEM:
		l_pMiscItem=reinterpret_cast<MiscItem*>(l_pAbstractItem);
		l_pNewAbstractItem=l_miscItemLib.createMiscItem(l_pMiscItem->getMiscType());
		break;

	default:
		break;
	}

	if (l_pNewAbstractItem==NULL)
	{
		//string str=item.toAscii();
		//cout << "error in MainWinMacroSim.addItemToScene(): unknown geometry: " << str << endl;//error msg
	}
	else
		// append geometry to scene model
		m_pSceneModel->appendItem(l_pNewAbstractItem, this->m_pQVTKWidget->getRenderer());

	// render vtk window
	this->m_pQVTKWidget->getScene()->Render();
	this->m_pQVTKWidget->getRenderer()->ResetCamera();
}

bool MainWinMacroSim::writeSceneToXML()
{
	QDomDocument document;

	// create the root element
	QDomElement root = document.createElement("scene");
	switch (m_guiSimParams.traceMode)
	{
	case GuiSimParams::SEQUENTIAL:
		root.setAttribute("mode", "SEQUENTIAL");
		break;
	case GuiSimParams::NONSEQUENTIAL:
		root.setAttribute("mode", "NONSEQUENTIAL");
		break;
	default:
		root.setAttribute("mode", "NONSEQUENTIAL");
		break;
	}
	if (m_guiSimParams.GPU_acceleration)
		root.setAttribute("GPUacceleration", "TRUE");
	else
		root.setAttribute("GPUacceleration", "FALSE");

	root.setAttribute("numCPU", QString::number(m_guiSimParams.numCPU));

	root.setAttribute("glassCatalog", m_guiSimParams.glassCatalog);
	root.setAttribute("outputFilePath", m_guiSimParams.outputFilePath);
	root.setAttribute("inputFilePath", m_guiSimParams.inputFilePath);
	root.setAttribute("rayTilingWidth", QString::number(m_guiSimParams.subsetWidth));
	root.setAttribute("rayTilingHeight", QString::number(m_guiSimParams.subsetWidth));

	// write the sources
	for (int i=0; i<m_pSceneModel->rowCount(QModelIndex()); i++)
	{
		AbstractItem* t_pItem=m_pSceneModel->getItem(m_pSceneModel->index(i,0,QModelIndex()));
		//// we only write the field items here
		//if (t_pItem->getObjectType() == AbstractItem::FIELD)
		//{
			if (!t_pItem->writeToXML(document, root))
			{
				cout << "error writing element " << i << " to XML" << endl;
				return false;
			}
//		}
	}

	//// create the geometryGroup
	//QDomElement geometryGroup = document.createElement("geometryGroup");

	//// add the geometries to our document
	//for(int i = 0; i < m_pSceneModel->rowCount(QModelIndex()); i++)
	//{
	//	AbstractItem* t_pItem=m_pSceneModel->getItem(m_pSceneModel->index(i,0,QModelIndex()));

	//	// we only write the geometry items items here
	//	if (t_pItem->getObjectType() == AbstractItem::GEOMETRY)
	//	{
	//		if (!t_pItem->writeToXML(document, geometryGroup))
	//		{
	//			cout << "error writing element " << i << " to XML" << endl;
	//			return false;
	//		}
	//	}
	//}
	//// add geometryGroup to the scene
	//root.appendChild(geometryGroup);


	//// write the detectors
	//for (int i=0; i<m_pSceneModel->rowCount(QModelIndex()); i++)
	//{
	//	AbstractItem* t_pItem=m_pSceneModel->getItem(m_pSceneModel->index(i,0,QModelIndex()));
	//	// we only write the field items here
	//	if (t_pItem->getObjectType() == AbstractItem::DETECTOR)
	//	{
	//		if (!t_pItem->writeToXML(document, root))
	//		{
	//			cout << "error writing element " << i << " to XML" << endl;
	//			return false;
	//		}
	//	}
	//}


	// add scene to the document
	document.appendChild(root);
	// clear any old versions
	m_html.clear();
	// write new string
	m_html=document.toString();
	return true;

}

void MainWinMacroSim::saveScene()
{
	// write to file
	if (m_fileName.isNull())
		saveSceneAs();
	else
	{
		writeSceneToXML();
		QFile file(m_fileName);
		if(!file.open(QIODevice::WriteOnly | QIODevice::Text))
		{
			cout << "failed to open file for writing" << endl;
		}
		else
		{
			QTextStream stream(&file);
			stream << m_html;
			file.close();
			cout << "done saving to file";
		}
	}
}

bool MainWinMacroSim::resetScene()
{
	if (maybeSave())
	{
		// clear current model
		if (m_pSceneModel!=NULL)
		{
			m_pSceneModel->clearModel();
			//delete m_pSceneModel;
		}
		this->m_pQVTKWidget->resetScene();
	}
	return true;
}

bool MainWinMacroSim::loadScene()
{
	static QString defaultDir;

	if (defaultDir.isEmpty())
	{
		defaultDir = QDir::currentPath();
	}

	QString filename=QFileDialog::getOpenFileName(this, tr("Open Scene"), defaultDir, tr("Scene Files (*.xml)"));

	if (filename.isEmpty())
	{
		return false;
	}

	QFile inFile(filename);
	if (!inFile.open(QIODevice::ReadOnly))
	{
		return false;
	}

	//store new directory (of indicated file) in defaultDir (default for next-time opening)
	QFileInfo info(filename);
	defaultDir = info.absolutePath();

	// clear current model
	if (m_pSceneModel!=NULL)
	{
		m_pSceneModel->clearModel();
		//delete m_pSceneModel;
	}
	this->m_pQVTKWidget->resetScene();

	// set up the new model according to file
	QDomDocument doc;
	if (!doc.setContent(&inFile))
	{
		cout << "error in MainWinMacroSim.loadScene(): could not load xml file" << endl;
		return false;
	}

	QDomNodeList l_sceneNodeList = doc.elementsByTagName("scene");
	if (l_sceneNodeList.count() != 1)
	{
		cout << "error in MainWinMacroSim.loadScene(): xml file must have one scene object." << endl;
		return false;
	}

	QDomElement sceneElement = l_sceneNodeList.at(0).toElement();

	QString str=sceneElement.attribute("mode");
	if (!str.compare("SEQUENTIAL"))
		m_guiSimParams.traceMode=GuiSimParams::SEQUENTIAL;
	else
		m_guiSimParams.traceMode=GuiSimParams::NONSEQUENTIAL;
	str=sceneElement.attribute("GPUacceleration");
	if (!str.compare("TRUE"))
		m_guiSimParams.GPU_acceleration=true;
	else
		m_guiSimParams.GPU_acceleration=false;

	m_guiSimParams.numCPU=sceneElement.attribute("numCPU").toInt();

	m_guiSimParams.glassCatalog=sceneElement.attribute("glassCatalog");
	m_guiSimParams.outputFilePath=sceneElement.attribute("outputFilePath");
	m_guiSimParams.inputFilePath=sceneElement.attribute("inputFilePath");
	m_guiSimParams.subsetHeight=sceneElement.attribute("rayTilingWidth").toULong();
	m_guiSimParams.subsetWidth=sceneElement.attribute("rayTilingHeight").toULong();

	// load all fields in the scene and add them to our sceneModel
	FieldItemLib l_fieldItemLib;

	QDomNodeList l_fieldNodeList = sceneElement.elementsByTagName("field");
	for (int iField=0; iField<l_fieldNodeList.count(); iField++)
	{
		QDomElement l_fieldElement=l_fieldNodeList.at(iField).toElement();

		QString l_fieldTypeStr=l_fieldElement.attribute("fieldType");
		FieldItem *l_pField=l_fieldItemLib.createField(l_fieldItemLib.stringToFieldType(l_fieldTypeStr));
		if (l_pField==NULL)
		{
			string str=l_fieldTypeStr.toAscii();
			cout << "error in MainWinMacoSim.loadScene(): could not create field of type: " << str << endl;
			return false;
		}
		l_pField->readFromXML(l_fieldElement);
		m_pSceneModel->appendItem(l_pField, this->m_pQVTKWidget->getRenderer());
	}

	// load all geometries in the scene and add them to our sceneModel
	GeometryItemLib l_geomItemLib;

	QDomNodeList l_geomGroupNodeList = sceneElement.elementsByTagName("geometryGroup");
	for (int iGeomGroup=0; iGeomGroup<l_geomGroupNodeList.count(); iGeomGroup++)
	{
		QDomElement l_geomGroupElement=l_geomGroupNodeList.at(iGeomGroup).toElement();
		GeomGroupItem *l_pGeomGroup=new GeomGroupItem();
		QDomNodeList l_geomNodeList = l_geomGroupElement.elementsByTagName("geometry");

		for (int iGeom=0; iGeom<l_geomNodeList.count(); iGeom++)
		{
			QDomElement l_geomElement=l_geomNodeList.at(iGeom).toElement();
			QString l_geomTypeStr=l_geomElement.attribute("geomType");
			GeometryItem *l_pGeom=l_geomItemLib.createGeometry(l_geomItemLib.stringToGeomType(l_geomTypeStr));
			if (l_pGeom==NULL)
			{
				string str=l_geomTypeStr.toAscii();
				cout << "error in MainWinMacoSim.loadScene(): could not create geometry of type: " << str << endl;
				return false;
			}
			l_pGeom->readFromXML(l_geomElement);
			l_pGeomGroup->setChild(l_pGeom);		
		}
		m_pSceneModel->appendItem(l_pGeomGroup, this->m_pQVTKWidget->getRenderer());
	}

	// load all fields in the scene and add them to our sceneModel
	DetectorItemLib l_detItemLib;

	QDomNodeList l_detNodeList = sceneElement.elementsByTagName("detector");
	for (int iDet=0; iDet<l_detNodeList.count(); iDet++)
	{
		QDomElement l_detElement=l_detNodeList.at(iDet).toElement();

		QString l_detTypeStr=l_detElement.attribute("detType");
		DetectorItem *l_pDet=l_detItemLib.createDetector(l_detItemLib.stringToDetType(l_detTypeStr));
		if (l_pDet==NULL)
		{
			string str=l_detTypeStr.toAscii();
			cout << "error in MainWinMacoSim.loadScene(): could not create detector of type: " << str << endl;
			return false;
		}
		bool test=connect(this, SIGNAL(simulationFinished(ito::DataObject)), l_pDet, SLOT(simulationFinished(ito::DataObject)));
		l_pDet->readFromXML(l_detElement);
		m_pSceneModel->appendItem(l_pDet, this->m_pQVTKWidget->getRenderer());
	}

	// save filename for future calls to save the scene
	m_fileName=filename;

	// render vtk window
	this->m_pQVTKWidget->getScene()->Render();
	m_pQVTKWidget->getRenderer()->ResetCamera();

	return true;
}

void MainWinMacroSim::sceneDataChangedFromPropEdit(const QModelIndex &topLeft, const QModelIndex &bottomRight)
{
	// if materialType, scatterType or coatingType changed, we might need to create new instances of those and append it to its parents
	// find rootItem that ultimately holds the item that just changed
	QModelIndex l_rootItemIndex=m_pSceneModel->getBaseIndex(m_activeItemIndex);

//	m_pSceneTreeView->setExpanded( l_rootItemIndex ,false);
//	m_pSceneTreeView->setExpanded( l_rootItemIndex ,true);

	// render vtk window
	this->m_pQVTKWidget->getScene()->Render();

//	m_pSceneTreeView->repaint();
}

//void MainWinMacroSim::sceneSelectionChanged()
//{
//	QList<QGraphicsItem*> l_pTest=m_pScene->selectedItems();
//	if (l_pTest.size() > 0)
//	{
//		Abstract_GraphViewItem* l_pGraphViewItem=reinterpret_cast<Abstract_GraphViewItem*>(l_pTest.at(0));
//		if (l_pGraphViewItem != 0)
//		{
//			AbstractItem* l_pItem=l_pGraphViewItem->getItem();
//			// invoke
//			emit signalSceneSelectionChanged(l_pItem->getModelIndex());
//			m_pSceneTreeView->setCurrentIndex(l_pItem->getModelIndex());
//			//QMetaObject::invokeMethod(m_pSceneModel, "changeItemFocus", Qt::QueuedConnection, Q_ARG(QModelIndex, l_pItem->getModelIndex()));
//		}
//		else
//		{
//			emit signalSceneSelectionChanged(QModelIndex());
//			m_pSceneTreeView->setCurrentIndex(QModelIndex());
//			//QMetaObject::invokeMethod(m_pSceneModel, "changeItemFocus", Qt::QueuedConnection, Q_ARG(QModelIndex, QModelIndex()));
//		}
//	}
//	else
//	{
//		emit signalSceneSelectionChanged(QModelIndex());
//		m_pSceneTreeView->setCurrentIndex(QModelIndex());
//		//QMetaObject::invokeMethod(m_pSceneModel, "changeItemFocus", Qt::QueuedConnection, Q_ARG(QModelIndex, QModelIndex()));
//	}
//}

void MainWinMacroSim::stopSimulation()
{
	emit terminateSimulation();
	statusBar()->showMessage("simulation terminated by user");
}

void MainWinMacroSim::startSimulation()
{
	m_pProgBar = new QProgressBar;
	statusBar()->clearMessage();
	// set min and max value
	m_pProgBar->setRange(0, 100);
	m_pProgBar->setFormat("executing simulation... %p%");
	statusBar()->addWidget(m_pProgBar);
	m_pProgBar->show();

	m_pTracer = new TracerThread();

	m_pTracerThreadThread->start(QThread::TimeCriticalPriority);

	// write xmlto member variable m_html
	writeSceneToXML();

	// connect output stream of tracer to our output widget
	ostream* l_pTracerStandardOutStream=m_pTracer->getCout();

	m_pTracerStatOutStream = new ConsoleStream((*l_pTracerStandardOutStream));

	bool test=connect(m_pTracerStatOutStream, SIGNAL(flushStream(QString)), this, SLOT(tracerStreamFlushed(QString)),Qt::QueuedConnection);

	//create output data object
	if (m_pResultField)
	{
		delete m_pResultField;
		m_pResultField=NULL;
	}
	m_pResultField = new ito::DataObject();

	// get file to ptx
	QString path = QCoreApplication::applicationDirPath();
	QDir l_path_to_ptx(path);
	if (l_path_to_ptx.cd("plugins\\MacroSim\\ptx"))
	{
		m_guiSimParams.path_to_ptx=l_path_to_ptx.canonicalPath();

		m_pTracer->init(m_html, m_pResultField, m_pTracerStatOutStream, m_guiSimParams);

		m_pTracer->moveToThread(m_pTracerThreadThread);

		test=connect(m_pTracer, SIGNAL(percentageCompleted(int)), m_pProgBar, SLOT(setValue(int)),Qt::QueuedConnection);
	//	test=connect(m_pTracer, SIGNAL(percentageCompleted(int)), this, SLOT(tracerThreadUpdate(int)),Qt::QueuedConnection);
		test=connect(m_pTracer, SIGNAL(finished(bool)), this, SLOT(tracerThreadFinished(bool)),Qt::QueuedConnection);
		test=connect(m_pTracer, SIGNAL(terminated()), this, SLOT(tracerThreadTerminated()),Qt::QueuedConnection);
		test=connect(this, SIGNAL(runSimulation()), m_pTracer, SLOT(runSimulation()), Qt::QueuedConnection);
		test=connect(this, SIGNAL(terminateSimulation()), m_pTracer, SLOT(terminate()));

		//qDebug() << QThread::currentThreadId();

		//m_pTracer->runSimulation();
		emit runSimulation();
	}
	else
		cout << "error in startSimulation(): path to ptx files does not exist at: path/plugins/MacroSim" << endl;


}

void MainWinMacroSim::startLayoutMode()
{
	m_pTracer = new TracerThread();
	m_pTracerThreadThread->start(QThread::HighestPriority);

	// delete old rayPlotData
	m_pQVTKWidget->getRayPlotData()->getData()->clear();

	// write xmlto member variable m_html
	writeSceneToXML();

	if (m_pResultField)
	{
		delete m_pResultField;
		m_pResultField=NULL;
	}

	// connect output stream of tracer to our output widget
	ostream* l_pTracerStandardOutStream=m_pTracer->getCout();

	m_pTracerStatOutStream = new ConsoleStream((*l_pTracerStandardOutStream));

	bool test=connect(m_pTracerStatOutStream, SIGNAL(flushStream(QString)), this, SLOT(tracerStreamFlushed(QString)),Qt::QueuedConnection);
	
	m_pTracer->init(m_html, m_pResultField, m_pTracerStatOutStream, m_guiSimParams);

	m_pTracer->moveToThread(m_pTracerThreadThread);

	test=connect(m_pTracer, SIGNAL(finished(bool)), this, SLOT(tracerThreadFinished(bool)));
	test=connect(m_pTracer, SIGNAL(terminated()), this, SLOT(tracerThreadTerminated()));
	test=connect(this, SIGNAL(terminateSimulation()), m_pTracer, SLOT(terminate()));
	test=connect(this, SIGNAL(runLayoutMode(RayPlotData *)), m_pTracer, SLOT(runLayoutMode(RayPlotData *)), Qt::QueuedConnection);

	test=connect(m_pTracerStatOutStream, SIGNAL(flushStream(QString)), this, SLOT(tracerStreamFlushed(QString)));

	emit runLayoutMode(this->m_pQVTKWidget->getRayPlotData());
}

void MainWinMacroSim::tracerThreadTerminated()
{
	statusBar()->showMessage("simulation terminated by user");
	// delete progress bar
	if (m_pProgBar != NULL)
	{
		statusBar()->removeWidget(m_pProgBar);
		delete m_pProgBar;
		m_pProgBar=0;
	}
	if (m_pTracerStatOutStream != NULL)
	{
		delete m_pTracerStatOutStream;
		m_pTracerStatOutStream=0;
	}
	if (m_pTracer != NULL)
	{
		delete m_pTracer;
		m_pTracer=NULL;
	}
}

void MainWinMacroSim::tracerThreadFinished(bool success)
{
	statusBar()->showMessage("finished simulation");
	if (m_pTracerStatOutStream!=0)
	{
		delete m_pTracerStatOutStream;
		m_pTracerStatOutStream = NULL;
	}
	if (m_pProgBar != NULL)
	{
		// delete progress bar
		statusBar()->removeWidget(m_pProgBar);
		delete m_pProgBar;
		m_pProgBar = NULL;
	}
	if (m_pTracer != NULL)
	{
		delete m_pTracer;
		m_pTracer=NULL;
	}
	// render ray plot data
	if (this->m_pQVTKWidget->getRayPlotData() != NULL)
	{
		this->m_pQVTKWidget->removeRaysFromView();
		this->m_pQVTKWidget->renderRays();
	}

	if(m_pResultField && success)
	{
		emit this->simulationFinished(*m_pResultField);
	}
}

void MainWinMacroSim::saveSceneAs()
{
	static QString defaultDir;

	if (defaultDir.isEmpty())
	{
		defaultDir = QDir::currentPath();
	}

	QString fileName=QFileDialog::getSaveFileName(this, tr("Save Scene"), defaultDir, tr("Scene Files (*.xml)"));

	if (fileName.isEmpty())
	{
		return;
	}

	QFile saveFile(fileName);
	if (saveFile.open(QIODevice::WriteOnly))
	{
		m_fileName = fileName;
		QFile file(fileName);
		if(!file.open(QIODevice::WriteOnly | QIODevice::Text))
		{
			cout << "failed to open file for writing" << endl;
		}
		else
		{
			QFileInfo info(fileName);
			defaultDir = info.absolutePath();

			writeSceneToXML();
			QTextStream stream(&file);
			stream << m_html;
			file.close();
			cout << "done saving to file";
		}
	}
	else
	{
		cout << "failed to open file for writing" << endl;
	}

}

void MainWinMacroSim::createMenus()
{
	m_pFileMenu = new QMenu(tr("&File"), this);
	menuBar()->addMenu(m_pFileMenu);

	// add file menu entries
	m_pFileMenu->addAction(QIcon("::gui/icons/new.png"), tr("&new"), this, SLOT(resetScene()), QKeySequence(tr("Ctrl+N", "File|new")));
	m_pFileMenu->addAction(QIcon("::gui/icons/open.png"), tr("&open"), this, SLOT(loadScene()), QKeySequence(tr("Ctrl+O", "File|open")));
	m_pFileMenu->addAction(QIcon("::gui/icons/close.png"), tr("&exit"), this, SLOT(exit()), QKeySequence(tr("Ctrl+E", "File|exit")));
	m_pFileMenu->addAction(QIcon("::gui/icons/fileSave.png"), tr("&save"), this, SLOT(saveScene()), QKeySequence(tr("Ctrl+S", "File|save")));
	m_pFileMenu->addAction(QIcon("::gui/icons/fileSaveAs.png"), tr("&save as"), this, SLOT(saveSceneAs()));

	m_pSimMenu = new QMenu(tr("&Simulation"), this);
	menuBar()->addMenu(m_pSimMenu);
	m_pSimMenu->addAction(QIcon("::gui/icons/Run.png"), tr("run"), this, SLOT(startSimulation()), QKeySequence(tr("Ctrl+R", "Simulation|run")));
	m_pSimMenu->addAction(QIcon("::gui/icons/Abort.png"), tr("abort"), this, SLOT(stopSimulation()), QKeySequence(tr("Ctrl+A", "Simulation|abort")));
	m_pSimMenu->addAction(QIcon("::gui/icons/pencil.ico"), tr("run layout"), this, SLOT(startLayoutMode()), QKeySequence(tr("Ctrl+L", "Simulation|layout")));
	m_pSimMenu->addAction(QIcon("::gui/icons/configure.png"), tr("configure"), this, SLOT(showSimConfigDialog()), QKeySequence(tr("Ctrl+P", "Simulation|configure")));

	m_pViewMenu = new QMenu(tr("&View"), this);
	menuBar()->addMenu(m_pViewMenu);
	m_pViewMenu->addAction(QIcon(), tr("toggle coordinate axes"), m_pQVTKWidget, SLOT(toggleCoordinateAxes()));
	m_pViewMenu->addAction(QIcon(), tr("set x-view up"), m_pQVTKWidget, SLOT(setXViewUp()));
	m_pViewMenu->addAction(QIcon(), tr("set y-view up"), m_pQVTKWidget, SLOT(setYViewUp()));
	m_pViewMenu->addAction(QIcon(), tr("set z-view up"), m_pQVTKWidget, SLOT(setZViewUp()));
	m_pViewMenu->addAction(QIcon(), tr("render options"), m_pQVTKWidget, SLOT(showRenderOptions()));
//	m_pViewMenu->addAction(QIcon("::gui/icons/Abort.png"), tr("abort"), this, SLOT(stopSimulation()), QKeySequence(tr("Ctrl+A", "Simulation|abort")));
//	m_pViewMenu->addAction(QIcon("::gui/icons/configure.png"), tr("configure"), this, SLOT(showSimConfigDialog()), QKeySequence(tr("Ctrl+C", "Simulation|configure")));
}

void MainWinMacroSim::showSimConfigDialog()
{
	DialogSimConfig *l_pDialog=new DialogSimConfig(this, m_guiSimParams);
	l_pDialog->show();
}

void MainWinMacroSim::createToolBars()
{
	m_pFileToolBar=addToolBar(tr("File"));
	m_pFileToolBar->addAction(QIcon(":gui/icons/new.png"), tr("&new"), this, SLOT(resetScene()));
	m_pFileToolBar->addAction(QIcon(":gui/icons/open.png"), tr("&open"), this, SLOT(loadScene()));
	m_pFileToolBar->addAction(QIcon(":gui/icons/close.png"), tr("&exit"), this, SLOT(close()));
	m_pFileToolBar->addAction(QIcon(":gui/icons/fileSave.png"), tr("&save"), this, SLOT(saveScene()));
	m_pFileToolBar->addAction(QIcon(":gui/icons/fileSaveAs.png"), tr("&save as"), this, SLOT(saveSceneAs()));

	m_pSimToolBar=addToolBar(tr("Simulation"));
	m_pSimToolBar->addAction(QIcon(":/gui/icons/Run.png"), tr("run"), this, SLOT(startSimulation()));
	m_pSimToolBar->addAction(QIcon(":/gui/icons/Abort.png"), tr("abort"), this, SLOT(stopSimulation()));
	m_pSimToolBar->addAction(QIcon(":/gui/icons/pencil.ico"), tr("run layout"), this, SLOT(startLayoutMode()));
	m_pSimToolBar->addAction(QIcon(":/gui/icons/configure.png"), tr("configure"), this, SLOT(showSimConfigDialog()));
	m_pSimToolBar->addAction(QIcon(":/gui/icons/snapshot.png"), tr("saveImage"), m_pQVTKWidget, SLOT(saveImage()));
}

void MainWinMacroSim::closeEvent(QCloseEvent *event)
{
	if (maybeSave())
	{
		event->accept();
	}
	else
		event->ignore();
}

bool MainWinMacroSim::maybeSave()
{
	QMessageBox::StandardButton ret;
	ret=QMessageBox::warning(this, tr("MacroSim"), tr("Do you want to save your changes to the Scene?"),QMessageBox::Save | QMessageBox::Discard | QMessageBox::Cancel);

	if (ret == QMessageBox::Save)
	{
		saveScene();
		return true;
	}
	else if (ret == QMessageBox::Discard)
		return true;
	return false;
}

void MainWinMacroSim::tracerThreadUpdate(int i)
{
	int test=i;
	//char lchar[512];
	//m_pTracerStatOutStream->getline(lchar, 512);
	//string str;
	//str = string(lchar);
	//if (!str.empty())
	//	m_pDockWidget_Console->appendText(QString::fromStdString(str));
}

void MainWinMacroSim::setGuiSimParams(GuiSimParams params)
{
	this->m_guiSimParams=params;
}

void MainWinMacroSim::tracerStreamFlushed(QString msg)
{
	m_pDockWidget_Console->appendText(msg);
}


DialogSimConfig::DialogSimConfig(MainWinMacroSim *parent, GuiSimParams params) 
{
	ui.setupUi(this);
	m_pParent=parent;
	ui.checkBox_GPU->setChecked(params.GPU_acceleration);
	switch (params.traceMode)
	{
	case GuiSimParams::NONSEQUENTIAL:
		ui.comboBox_SimMode->setCurrentIndex(1);
		break;
	case GuiSimParams::SEQUENTIAL:
		ui.comboBox_SimMode->setCurrentIndex(0);
		break;
	default:
		ui.comboBox_SimMode->setCurrentIndex(1);
		break;
	}
	ui.lineEditGlassCatalog->setText(params.glassCatalog);
	ui.lineEditOutputFilePath->setText(params.outputFilePath);
	ui.lineEditInputFilePath->setText(params.inputFilePath);
	ui.sBoxSubsetWidth->setValue(params.subsetWidth);
	ui.sBoxSubsetHeight->setValue(params.subsetHeight);
	ui.sBoxCPUNumber->setValue(params.numCPU);
	bool test=connect(this->ui.buttonGlassCatalog, SIGNAL(clicked(bool)), this, SLOT(browseGlassCatalog(bool)));
	test=connect(this->ui.buttonOutputFilePath, SIGNAL(clicked(bool)), this, SLOT(browseOutputFilePath(bool)));
	test=connect(this->ui.buttonInputFilePath, SIGNAL(clicked(bool)), this, SLOT(browseInputFilePath(bool)));
}

DialogSimConfig::~DialogSimConfig()
{

}

void DialogSimConfig::accept()
{
	GuiSimParams params;
	params.GPU_acceleration=ui.checkBox_GPU->isChecked();
	switch (ui.comboBox_SimMode->currentIndex())
	{
	case 0:
		params.traceMode=GuiSimParams::SEQUENTIAL;
		break;
	case 1:
		params.traceMode=GuiSimParams::NONSEQUENTIAL;
		break;
	default:
		params.traceMode=GuiSimParams::NONSEQUENTIAL;
		break;
	}
	int currentIndex=ui.comboBox_SimMode->currentIndex();
	params.glassCatalog=ui.lineEditGlassCatalog->text();
	params.outputFilePath=ui.lineEditOutputFilePath->text();
	params.inputFilePath=ui.lineEditInputFilePath->text();
	params.subsetHeight=ui.sBoxSubsetHeight->value();
	params.subsetWidth=ui.sBoxSubsetWidth->value();
	params.numCPU=ui.sBoxCPUNumber->value();
	m_pParent->setGuiSimParams(params);

	QDialog::accept();
}