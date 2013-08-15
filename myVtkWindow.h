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

#ifndef MYVTKWINDOW
#define MYVTKWINDOW

#include <qModelIndex>
#include "macrosim_scenemodel.h"
#include "abstractItem.h"
#include <QReadWriteLock>
#include "MacroSimLib.h"

#include "ui_renderOptionsDialog.h"
#include "ui_imageSavingIdleDialog.h"
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkActor.h>
#include <QVTKWidget.h>

#include <vtkRenderWindow.h>
#include <vtkSmartPointer.h>
#include <vtkOrientationMarkerWidget.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkActor.h>

#include <qmovie.h>


//using namespace macrosim;
#ifndef PI
	#define PI 3.14159265358979
#endif

namespace macrosim
{

class IdleDialog : public QDialog
{
	Q_OBJECT
public:
	IdleDialog(QWidget *parent = NULL)
	{
	};

	~IdleDialog()
	{

	};

protected:

private:
	Ui::idleDialog ui;
	QMovie *m_pMovie;
	

signals:

public slots:
};

class RenderOptionsDialog :
	public QDialog
{
	Q_OBJECT

public:
	RenderOptionsDialog(RenderOptions &optionsIn, QWidget *parent=0) :
		QDialog(parent)
	{
		ui.setupUi(this);
		ui.m_cBoxCoordAxes->setChecked(optionsIn.m_showCoordAxes);
		ui.m_sBoxRenderSlicesHeight->setValue(optionsIn.m_slicesHeight);
		ui.m_sBoxRenderSlicesWidth->setValue(optionsIn.m_slicesWidth);
		ui.m_dsBoxAmbientInt->setValue(optionsIn.m_ambientInt);
		ui.m_dsBoxDiffuseInt->setValue(optionsIn.m_diffuseInt);
		ui.m_dsBoxSpecularInt->setValue(optionsIn.m_specularInt);
		ui.m_dsBoxBackgroundR->setValue(optionsIn.m_backgroundColor.X);
		ui.m_dsBoxBackgroundG->setValue(optionsIn.m_backgroundColor.Y);
		ui.m_dsBoxBackgroundB->setValue(optionsIn.m_backgroundColor.Z);
		ui.m_dsBoxLightPosX->setValue(optionsIn.m_lightPos.X);
		ui.m_dsBoxLightPosY->setValue(optionsIn.m_lightPos.Y);
		ui.m_dsBoxLightPosZ->setValue(optionsIn.m_lightPos.Z);
		ui.comboBox->setCurrentIndex(renderModeToComboBoxIndex(optionsIn.m_renderMode));
	}
	~RenderOptionsDialog()
	{

	}

	void setOptions(RenderOptions &in) {this->m_renderOptions=in;};
	RenderOptions getOptions(void) {return this->m_renderOptions;};

private:
	Ui::RenderOptionsDialog ui;

	RenderOptions m_renderOptions;

signals:
	void renderOptionsDialogAccepted(RenderOptions &options);

public slots:
	void accept() {
		m_renderOptions.m_showCoordAxes=ui.m_cBoxCoordAxes->isChecked();
		m_renderOptions.m_slicesHeight=ui.m_sBoxRenderSlicesHeight->value();
		m_renderOptions.m_slicesWidth=ui.m_sBoxRenderSlicesWidth->value();
		m_renderOptions.m_ambientInt=ui.m_dsBoxAmbientInt->value();
		m_renderOptions.m_diffuseInt=ui.m_dsBoxDiffuseInt->value();
		m_renderOptions.m_specularInt=ui.m_dsBoxSpecularInt->value();
		m_renderOptions.m_backgroundColor.X=ui.m_dsBoxBackgroundR->value();
		m_renderOptions.m_backgroundColor.Y=ui.m_dsBoxBackgroundG->value();
		m_renderOptions.m_backgroundColor.Z=ui.m_dsBoxBackgroundB->value();
		m_renderOptions.m_lightPos.X=ui.m_dsBoxLightPosX->value();
		m_renderOptions.m_lightPos.Y=ui.m_dsBoxLightPosY->value();
		m_renderOptions.m_lightPos.Z=ui.m_dsBoxLightPosZ->value();
		m_renderOptions.m_renderMode=stringToRenderMode(ui.comboBox->currentText());
		emit renderOptionsDialogAccepted(m_renderOptions);
		this->done(QDialog::Accepted);
	};


};

class RayPlotData
{
public:
	RayPlotData() :
	  m_pLock(NULL)
	{
		m_pLock=QSharedPointer<QReadWriteLock>(new QReadWriteLock());
	}
	~RayPlotData()
	{
		m_pLock.clear();
	}

protected:
	QSharedPointer<QReadWriteLock> m_pLock;
	QVector<QVector<Vec3d>> m_data;
	vtkSmartPointer<vtkActor> m_pActor;
	vtkSmartPointer<vtkPolyDataMapper> m_pMapper;
	vtkSmartPointer<vtkPolyData> m_pPolydata;

public:
	QVector<QVector<Vec3d>> *getData() {return &m_data;};
	bool initData(unsigned long long size) 
	{ 
		if (m_data.size()!=0)
			return false;
		else
			m_data.reserve(size);
		return true;
	};
	QSharedPointer<QReadWriteLock> getLock() {return m_pLock;};
	void render(QMatrix4x4 &view, RenderOptions &options);
	void renderVtk(vtkSmartPointer<vtkRenderer> renderer);
	void updateVtk(vtkSmartPointer<vtkRenderer> renderer);
	void removeFromView(vtkSmartPointer<vtkRenderer> renderer) {renderer->RemoveActor(m_pActor);};
};

class myVtkWindow :
	public QVTKWidget
{
	Q_OBJECT

public:
	myVtkWindow(QWidget *paren);
	~myVtkWindow()
	{

	};

	SceneModel* getModel() const {return m_pModel;};
	void setModel(SceneModel* in) {m_pModel = in; m_pModel->setRenderOptions(this->m_renderOptions);};
	void setItemFocus(const QModelIndex &in) {m_focusedItemIndex=in;};
	QModelIndex getItemFocus() const {return m_focusedItemIndex;};
	RayPlotData* getRayPlotData() {return &m_rayPlotData;};
	vtkSmartPointer<vtkRenderWindow> getScene();
	void resetScene();
	void drawCoordinateAxes();
	void renderRays();
	void removeRaysFromView();
	vtkSmartPointer<vtkRenderer> getRenderer() {return m_pRenderer;};

private:
	SceneModel* m_pModel;
	QModelIndex m_focusedItemIndex;
	RenderOptions m_renderOptions;
	RayPlotData m_rayPlotData;

	vtkSmartPointer<vtkRenderWindow> m_pVtkScene;
	vtkSmartPointer<vtkRenderer> m_pRenderer;
	vtkSmartPointer<vtkOrientationMarkerWidget> m_pVtkAxesWidget;

protected:
	static void callbackSelectActorFromRenderWin(void* p2Object, void* p2Actor);
	void keyPressEvent(QKeyEvent *event);

signals:
	void itemFocusChanged(const QModelIndex topLeft);
	void itemDataChanged(const QModelIndex &topLeft, const QModelIndex &bottomRight);
	void changeItemSelection(const QModelIndex &index);
	void vtkWinKeyEvent(QKeyEvent *keyEvent);

public slots:
	// slot functions
	void modelAboutToBeReset();
	void layoutChanged();
	void rowsInserted(const QModelIndex &index, int start, int end);
	void rowsAboutToBeRemoved(const QModelIndex &index, int start, int end);
	void changeItemData(const QModelIndex &topLeft, const QModelIndex &bottomRight);
	void changeItemFocus(const QModelIndex &index);
	void toggleCoordinateAxes();
	void setXViewUp();
	void setYViewUp();
	void setZViewUp();
	void showRenderOptions();
	void acceptRenderOptions(RenderOptions &options);
	void saveImage();
};


} // end namespace macrosim

#endif