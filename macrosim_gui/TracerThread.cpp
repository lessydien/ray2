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

#include "TracerThread.h"
#include "macroSim.h"
#include <cstring>
#include <iostream>
#include <sstream>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>


using namespace std;


TracerThread::TracerThread(void) :
	m_pTracer(NULL),
		m_pFieldObject(NULL),
		m_pOutBuffer(NULL),
		m_pRayPlotData(NULL)
{
}


TracerThread::~TracerThread(void)
{
}

ostream* TracerThread::getCout()
{
	return m_pTracer->getCout();
}

void TracerThread::init(QString& scene, ito::DataObject* field, ConsoleStream *outBuffer, GuiSimParams &guiParams)
{
	m_scene=scene;
	m_pFieldObject=field;
	m_pOutBuffer=outBuffer;
	m_pTracer=new MacroSimTracer();

	// transfer gui sim params to macrosim params
	this->m_params.subsetHeight=guiParams.subsetHeight;
	this->m_params.subsetWidth=guiParams.subsetWidth;
	this->m_params.numCPU=guiParams.numCPU;
	switch (guiParams.traceMode)
	{
	case GuiSimParams::SEQUENTIAL:
        this->m_params.simParams.traceMode=TRACE_SEQ;
		break;
	case GuiSimParams::NONSEQUENTIAL:
        this->m_params.simParams.traceMode=TRACE_NONSEQ;
		break;
	}

    QString glassFile = QDir::toNativeSeparators( guiParams.glassCatalog );
    glassFile.replace("//","/");
    strcpy( &(m_params.glassFilePath[0]), glassFile.toLatin1().data() );

    QString outFilePath = QDir::toNativeSeparators( guiParams.outputFilePath );
    outFilePath.replace("//","/");
    outFilePath.replace( QDir::separator(), "\\" );
    if (outFilePath.endsWith( "\\" ))
    {
        outFilePath = outFilePath.mid(0, outFilePath.length() - 1);
    }
    strcpy(&(m_params.outputFilesPath[0]), outFilePath.toLatin1().data() );

    QString inFilePath = QDir::toNativeSeparators( guiParams.inputFilePath );
    inFilePath.replace("//","/");
    inFilePath.replace( QDir::separator(), "\\" );
    if (inFilePath.endsWith( "\\" ))
    {
        inFilePath = inFilePath.mid(0, inFilePath.length() - 1);
    }
    strcpy(&(m_params.inputFilesPath[0]), inFilePath.toLatin1().data() );

	/*string l_glassString=guiParams.glassCatalog.toLatin1();
	strcpy(&(this->m_params.glassFilePath[0]), l_glassString.c_str());
	string l_outFilePathString=guiParams.outputFilePath.toLatin1();
	strcpy(&(this->m_params.outputFilesPath[0]), l_outFilePathString.c_str());
	string l_inFilePathString=guiParams.inputFilePath.toLatin1();
	strcpy(&(this->m_params.inputFilesPath[0]), l_inFilePathString.c_str());*/
    QString ptxPath = QDir::toNativeSeparators( guiParams.path_to_ptx );
    ptxPath.replace("//","/");
    ptxPath.replace( QDir::separator(), "\\" );
    if (ptxPath.endsWith( "\\" ))
    {
        ptxPath = ptxPath.mid(0, ptxPath.length() - 1);
    }
	strcpy(&(m_params.path_to_ptx[0]), ptxPath.toLatin1().data() );
}

//void TracerThread::run()
//{
//	int i= exec();
//	int j=2;
////	runMacroSimRayTrace();
////emit finished();
//}

void TracerThread::runMacroSimRayTrace()
{
	//qDebug() << QThread::currentThreadId();
	string l_sceneStr=m_scene.toLatin1();
	char* sceneChar=new char [l_sceneStr.size()+1];
	sceneChar=strcpy(sceneChar,l_sceneStr.c_str());

	bool breakCond;
	void* l_pField=NULL;
	ItomFieldParams l_fieldParams;
	bool ret=m_pTracer->runMacroSimRayTrace(sceneChar, &l_pField, &l_fieldParams, (void*)this, TracerThread::callbackProgressWrapper);

	if (ret)
	{
		createDataObjectFromMacroSimResult(l_fieldParams, m_pFieldObject, l_pField);
	}

	emit finished(ret);
	// cleanup when we're done
	delete m_pTracer;
	m_pTracer=NULL;
}

void TracerThread::runMacroSimLayoutTrace(RayPlotData *rayPlotData)
{
	string l_sceneStr=m_scene.toLatin1();
	char* sceneChar=new char [l_sceneStr.size()+1];
	sceneChar=strcpy(sceneChar,l_sceneStr.c_str());

	bool breakCond;
	
	// save pointer to rayPlotData
	this->m_pRayPlotData=rayPlotData;

	bool ret=m_pTracer->runMacroSimLayoutTrace(sceneChar, (void*)this, TracerThread::callbackRayPlotData);

	// cleanup when we're done
	delete m_pTracer;
	m_pTracer=NULL;
	emit finished(ret);
}

void TracerThread::callbackProgressWrapper(void* p2Object, int progressValue)
{
	TracerThread* l_pTracerThread=(TracerThread*)p2Object;
	emit l_pTracerThread->percentageCompleted(progressValue);
}

void TracerThread::callbackRayPlotData(void* p2Object, double* rayPlotDataTracer, RayPlotDataParams *params)
{
	TracerThread* l_pTracerThread=(TracerThread*)p2Object;
	// get lock to rayPlotData
	l_pTracerThread->m_pRayPlotData->getLock()->lockForWrite();
	// if this is the first call to this callback, init the rayPlotData-Vector to the size of the launch
	if (l_pTracerThread->m_pRayPlotData->getData()->size() == 0)
	{
		l_pTracerThread->m_pRayPlotData->getData()->resize(params->launchDim[1]*params->launchDim[0]);
	}
	// loop through rayPlotData
	for (unsigned long long jy=0;jy<params->subsetDim[1];jy++)
	{
		for (unsigned long long jx=0;jx<params->subsetDim[0];jx++)
		{
			// retrieve data from double pointer
			Vec3d newPos=Vec3d(rayPlotDataTracer[0+jx*3+jy*params->subsetDim[0]*3], rayPlotDataTracer[1+jx*3+jy*params->subsetDim[0]*3], rayPlotDataTracer[2+jx*3+jy*params->subsetDim[0]*3]);
			// pick current ray from rayPlotData
			QVector<Vec3d> *l_pRay=&(l_pTracerThread->m_pRayPlotData->getData()->data()[jx+params->launchOffset[0]+(jy+params->launchOffset[1])*params->launchDim[0]]);
			if (l_pRay->size() == 0)
				l_pRay->append(newPos); // save first pos
			else if ( newPos != l_pRay->at(l_pRay->size()-1) )
			{
				// save new pos if pos changed
				l_pRay->append(newPos);
			}
		}
	}
	l_pTracerThread->m_pRayPlotData->getLock()->unlock();
}