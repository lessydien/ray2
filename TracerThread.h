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

#ifndef TRACERTHREAD_H
#define TRACERTHREAD_H
#include <QThread>
#include <sstream>

#include "DataObject/dataobj.h"

#include "MacroSimLib.h"
#include "consoleStream.h"
#include <qvector.h>
#include <qlist.h>
#include "QPropertyEditor/CustomTypes.h"
#include "qdir.h"
//#include "myGraphicsScene.h"
#include "myVtkWindow.h"

using namespace macrosim;

class GuiSimParams
{
public :
	GuiSimParams() :
	   GPU_acceleration(false),
		   traceMode(NONSEQUENTIAL),
		   glassCatalog(QString()),
		   outputFilePath(QString()),
		   inputFilePath(QString()),
		   subsetWidth(2000),
		   subsetHeight(2000),
		   numCPU(1),
		   path_to_ptx(QString())
		   
	{
	};
	enum TraceMode {SEQUENTIAL, NONSEQUENTIAL};

	bool GPU_acceleration;
	TraceMode traceMode;
	QString glassCatalog;
	QString outputFilePath;
	QString inputFilePath;
	unsigned long subsetWidth;
	unsigned long subsetHeight;
	int numCPU;
	QString path_to_ptx;
};

class TracerThread :
	public QObject //QThread
{
	Q_OBJECT
public:
	TracerThread(void);
	~TracerThread(void);

	void init(QString& scene, ito::DataObject* field, ConsoleStream* outBuffer, GuiSimParams &guiParams);
	ostream* getCout();
	QVector<QVector<Vec3d>>* getRayPlotData();
	void setRayPlotData(RayPlotData* in) {m_pRayPlotData = in;};

private:
	MacroSimTracerParams m_params;
	
protected:
	QString m_scene;
	ito::DataObject* m_pFieldObject;
	ConsoleStream* m_pOutBuffer;
	MacroSimTracer* m_pTracer;
	RayPlotData *m_pRayPlotData;
	QDir m_path_to_ptx;

	static void callbackProgressWrapper(void* p2Object, int progressValue);
	static void callbackRayPlotData(void* p2Object, double* rayPlotData, RayPlotDataParams* params);

signals:
	void percentageCompleted(int);
	void rayPlotDataUpdated(QVector<unsigned int>);
	void finished(bool);

public slots:
	void runSimulation();
	void runLayoutMode(RayPlotData *rayPlotData);

};

#endif