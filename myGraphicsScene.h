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

#ifndef MYGRAPHICSSCENE
#define MYGRAPHICSSCENE

#include <qgraphicsscene.h>
#include <qModelIndex>
#include "macrosim_scenemodel.h"
#include <QGraphicsSceneWheelEvent>
#include "trackball.h"
#include "abstractItem.h"
#include <QReadWriteLock>
#include "MacroSimLib.h"

#include "ui_renderOptionsDialog.h"

#include <QtOpenGL>

//using namespace macrosim;
#ifndef PI
	#define PI 3.14159265358979
#endif

namespace macrosim
{

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
};

class MyGraphicsScene :
	public QGraphicsScene
{
	Q_OBJECT

public:
	MyGraphicsScene(QObject *parent) :
		QGraphicsScene(parent),
			m_pModel(0)
	{
		m_trackBalls[0] = TrackBall(0.0f, QVector3D(0, 1, 0), TrackBall::Sphere);
		m_trackBalls[1] = TrackBall(0.0f, QVector3D(0, 0, 1), TrackBall::Translation);
		m_trackBalls[2] = TrackBall(0.0f, QVector3D(0, 1, 0), TrackBall::Plane);
		m_zoom=1;
		initGL();

		m_timer = new QTimer(this);
		m_timer->setInterval(20);
		connect(m_timer, SIGNAL(timeout()), this, SLOT(update()));
		m_timer->start();

		m_time.start();

		m_renderOptions.m_slicesHeight=10;
		m_renderOptions.m_slicesWidth=11;
		m_renderOptions.m_showCoordAxes=true;
	};
	~MyGraphicsScene()
	{

	};

	SceneModel* getModel() const {return m_pModel;};
	void setModel(SceneModel* in) {m_pModel = in;};
	void setItemFocus(const QModelIndex &in) {m_focusedItemIndex=in;};
	QModelIndex getItemFocus() const {return m_focusedItemIndex;};
	RayPlotData* getRayPlotData() {return &m_rayPlotData;};

	virtual void drawBackground(QPainter *painter, const QRectF &rect);

protected:
    virtual void mousePressEvent(QGraphicsSceneMouseEvent *event);
    virtual void mouseReleaseEvent(QGraphicsSceneMouseEvent *event);
    virtual void mouseMoveEvent(QGraphicsSceneMouseEvent *event);
    virtual void wheelEvent(QGraphicsSceneWheelEvent * event);

private:
	SceneModel* m_pModel;
	QModelIndex m_focusedItemIndex;
	TrackBall m_trackBalls[3]; // virtual trackballs for rotating and moving the rendered scene
	QTimer *m_timer;
	QTime m_time;
    int m_lastTime;
    int m_mouseEventTime;
	float m_zoom;
	RenderOptions m_renderOptions;
	RayPlotData m_rayPlotData;

	void initGL();
	void drawCoordinateAxes();
	QPointF pixelPosToViewPos(const QPointF& p);

signals:
	void itemFocusChanged(const QModelIndex topLeft);
	void itemDataChanged(const QModelIndex &topLeft, const QModelIndex &bottomRight);

public slots:
	// slot functions
	void modelAboutToBeReset();
	void layoutChanged();
	void rowsInserted(const QModelIndex &index, int start, int end);
	void rowsAboutToBeRemoved(const QModelIndex &index, int start, int end);
	void changeItemData(const QModelIndex &topLeft, const QModelIndex &bottomRight);
	void changeItemFocus(const QModelIndex &index);
	void toggleCoordinateAxes();
	void setXYView();
	void setXZView();
	void setYZView();
	void showRenderOptions();
	void acceptRenderOptions(RenderOptions &options);
};


} // end namespace macrosim

#endif