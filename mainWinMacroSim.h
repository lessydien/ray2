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

#ifndef MAINWINMACROSIM_H
#define	MAINWINMACROSIM_H

#include "common/sharedStructures.h"

#include "DataObject/dataobj.h"


//#include "MyDelegate.h"
#include "geometryItem.h"
#include "geomSphericalLenseItem.h"
#include "materialItem.h"
#include "scatterItem.h"
#include "coatingItem.h"
#include "macrosim_scenemodel.h"
#include "macrosim_librarymodel.h"


//#include "QPropertyEditor/QPropertyModel.h"
#include "QPropertyEditor/QPropertyEditorWidget.h"

//#include "myGraphicsScene.h"
#include <qplaintextedit.h>
#include "TracerThread.h"
#include "dockWidget_Console.h"
#include "consoleStream.h"

#include "myVtkWindow.h"

#include <QtGui>
#include <qdialog.h>
#include <QtCore>


#include "ui_mainWinMacroSim.h"
#include "ui_simConfigDialog.h"

using namespace macrosim;

class MainWinMacroSim : public QMainWindow
{
	Q_OBJECT
public:
	MainWinMacroSim(QWidget *parent = NULL);
	~MainWinMacroSim();



	void setGuiSimParams(GuiSimParams params);

protected:

	void createMenus();
	void createToolBars();

	void closeEvent(QCloseEvent *event);
	void keyPressEvent(QKeyEvent *event);
	bool maybeSave();
	bool writeSceneToXML();

private:
	Ui::MainWinMacroSim ui;

//	MyGraphicsScene *m_pScene;

	macrosim::SceneModel *m_pSceneModel;
	macrosim::SceneModel *m_pLibraryModel;
	QStringList m_Library;

	myVtkWindow *m_pQVTKWidget;

	QDockWidget *m_pDockWidget_PropEditor;
	QPropertyEditorWidget *m_pItemPropertyWidget;
	QDockWidget *m_pDockWidget_SceneTreeView;
	QTreeView *m_pSceneTreeView;
	QDockWidget *m_pDockWidget_LibraryView;
	QTreeView *m_pLibraryModel_TreeView;
	dockWidget_Console *m_pDockWidget_Console;
	
	QMenu *m_pFileMenu;
	QMenu *m_pSimMenu;
	QMenu *m_pViewMenu;
	QString m_fileName;
	QString m_html;
	QToolBar *m_pFileToolBar;
	QToolBar *m_pSimToolBar;
	

	TracerThread *m_pTracer;
	QProgressBar *m_pProgBar;
	ConsoleStream *m_pTracerStatOutStream;

	QModelIndex m_activeItemIndex;

	GuiSimParams m_guiSimParams;

	QThread *m_pTracerThreadThread;

	ito::DataObject *m_pResultField;

	QList<vtkSmartPointer<vtkActor>> m_pActorList;

signals:
//	void signalSceneSelectionChanged(const QModelIndex topLeft);
	void signalSceneDataChangedFromPropEdit(const QModelIndex &topLeft, const QModelIndex &bottomRight);
	void terminateSimulation();
	void runMacroSimLayoutTrace(RayPlotData *rayPlotData);
	void runMacroSimRayTrace();
	void simulationFinished(ito::DataObject resultObject);

public slots:

	void pushItemToPropertyEditor(const QModelIndex topLeft);
	void addItemToScene(const QModelIndex index);
	void saveScene();
	void saveSceneAs();
	bool loadScene();
	bool resetScene();
	void sceneDataChangedFromPropEdit(const QModelIndex &topLeft, const QModelIndex &bottomRight);
//	void sceneSelectionChanged();
	void startSimulation();
	void exit() {close();};
	void tracerThreadUpdate(int in);
	void tracerStreamFlushed(QString msg);
	void tracerThreadFinished(bool success);
	void tracerThreadTerminated();
	void stopSimulation();
	void showSimConfigDialog();
	void startLayoutMode();
	void processKeyEvent(QKeyEvent *keyEvent);

private slots:


};

class DialogSimConfig : public QDialog
{
	Q_OBJECT
public:
	DialogSimConfig(MainWinMacroSim *parent = NULL, GuiSimParams params = *(new GuiSimParams()));
	~DialogSimConfig();

protected:

private:
	Ui::Dialog_SimConfig ui;
	MainWinMacroSim *m_pParent;
	GuiSimParams m_params;

signals:

public slots:
	void accept();
	void browseGlassCatalog(bool clicked)
	{
		ui.lineEditGlassCatalog->setText(QFileDialog::getOpenFileName(this, tr("Glass Catalog"), "E:/m12/trunk/iTOM/Qitom", tr("glass catalog files (*.AGF)")));
	};
	void browseOutputFilePath(bool clicked)
	{
		ui.lineEditOutputFilePath->setText(QFileDialog::getExistingDirectory(this, tr("Output Filepath"), "E:/m12/trunk/iTOM/Qitom", QFileDialog::ShowDirsOnly|  QFileDialog::DontResolveSymlinks));
	};
	void browseInputFilePath(bool clicked)
	{
		ui.lineEditInputFilePath->setText(QFileDialog::getExistingDirectory(this, tr("Input Filepath"), "E:/m12/trunk/iTOM/Qitom", QFileDialog::ShowDirsOnly|  QFileDialog::DontResolveSymlinks));
	};
};

#endif
