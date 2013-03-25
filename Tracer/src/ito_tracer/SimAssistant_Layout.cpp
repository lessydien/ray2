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

/**\file SimAssistantLayout.cpp
* \brief 
* 
*           
* \author Mauch
*/

#include "SimAssistant_Layout.h"
#include "myUtil.h"
#include "sampleConfig.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "Geometry.h"
#include "GeometryLib.h"
#include "math.h"
#include "randomGenerator.h"
#include "Detector.h"
#include "DetectorLib.h"
#include <ctime>

/**
 * \detail setPathToPtx 
 *
 * sets the path to the ptx file that the .cu file defining the behaviour of the SimAssistantLayout on the GPU of the surface will be compiled to
 *
 * \param[in] char* path
 * 
 * \return void
 * \sa 
 * \remarks 
 * \author Mauch
 */
void SimAssistantLayout::setPathToPtx(char* path)
{
	memcpy(this->path_to_ptx, path, sizeof(this->path_to_ptx));
};

/**
 * \detail getPathToPtx 
 *
 * returns the path to the ptx file that the .cu file defining the behaviour of the SimAssistantLayout on the GPU of the surface will be compiled to
 *
 * \param[in] void
 * 
 * \return const char*
 * \sa 
 * \remarks 
 * \author Mauch
 */
const char* SimAssistantLayout::getPathToPtx(void)
{
	return this->path_to_ptx;
};

///**
// * \detail setParamsPtr 
// *
// * \param[in] detParams *paramsPtr
// * 
// * \return void
// * \sa 
// * \remarks 
// * \author Mauch
// */
//void SimAssistantLayout::setParamsPtr(simAssParams *paramsPtrIn)
//{
//	this->paramsPtr=static_cast<simAssLayoutParams *>(paramsPtrIn);
//};

///**
// * \detail getParamsPtr 
// *
// * \param[in] void
// * 
// * \return detParams*
// * \sa 
// * \remarks 
// * \author Mauch
// */
//simAssLayoutParams* SimAssistantLayout::getParamsPtr(void)
//{
//	return this->paramsPtr;
//};

///**
// * \detail initSimulation 
// *
// * \param[in] Group *oGroupPtr, RayField *SourceListPtrPtr
// * 
// * \return simAssError
// * \sa 
// * \remarks 
// * \author Mauch
// */
//simAssError SimAssistantLayout::initSimulation( Group *oGroupPtr, Field *SourceListPtrPtr)
//{
//	std::cout << "*****************************************************" << std::endl;
//	std::cout << "starting to initialize Simulation..." << std::endl;
//
//	clock_t start, end;
//	start=clock();
//
//	SourceListPtrPtr->createLayoutInstance();
//	if (GROUP_NO_ERR != oGroupPtr->createCPUSimInstance(SourceListPtrPtr->getParamsPtr()->lambda, paramsPtr->mode) )
//	{
//		std::cout << "error in SimAssistant.initSimulation(): group.createCPUSimInstance() returned an error" << std::endl;
//		return SIMASS_ERROR;
//	}
//
//	end=clock();
//	double msecs;
//	msecs=((end-start)/(double)CLOCKS_PER_SEC*1000.0);
//	std::cout << msecs <<"ms to initialize Simulation." << std::endl;
//
//	return SIMASS_NO_ERROR;
//};

/**
 * \detail run
 *
 * runs the simulation
 *
 * \param[in] 
 * 
 * \return simAssError
 * \sa 
 * \remarks 
 * \author Mauch
 */
simAssError SimAssistantLayout::run(Group *oGroupPtr, RayField *SourceListPtrPtr, Detector **DetectorListPtrPtr)
{
//	//RTsize				buffer_width, buffer_height; // get size of output buffer
//	//void				*data; // pointer to cast output buffer into
// //	rayStruct			*bufferData;

	std::cout << "*****************************************************" << std::endl;
	std::cout << "starting simulation..." << std::endl;

	clock_t start, end;
	double msecs_Saving=0;
	double msecs=0;

	start=clock();

	unsigned long long width=SourceListPtrPtr->getParamsPtr()->widthLayout;//SourceListPtrPtr->getParamsPtr()->width*SourceListPtrPtr->getParamsPtr()->nrRayDirections.x*SourceListPtrPtr->getParamsPtr()->nrRayDirections.y;
	unsigned long long height=SourceListPtrPtr->getParamsPtr()->heightLayout;//SourceListPtrPtr->getParamsPtr()->height;

	int subSetNr_Width=0;
	unsigned long long overheadX=0; // overhead from one row to the next in case width is not an integer multiple of GPUSubset_width
	unsigned long long count=0;
	unsigned long long tracedRayNr=0;

	double3 *oldPositionsPtr;
	// create matlab variables
//	mxArray *mat_oldPos=NULL, *mat_newPos=NULL;

	// loop over GPU subsets
	while (SourceListPtrPtr->getParamsPtr()->launchOffsetY<=height-1)
	{
		while (SourceListPtrPtr->getParamsPtr()->launchOffsetX<=width-1)
		{
			// flag to indicate wether there is any ray still running
			bool anythingRunning=true;

			long2 l_GPUSubsetDim=make_long2(width, height);
			SourceListPtrPtr->getParamsPtr()->GPUSubset_width=width;
			SourceListPtrPtr->getParamsPtr()->GPUSubset_height=height;

			if (FIELD_NO_ERR!=SourceListPtrPtr->initCPUSubset())
			{
				std::cout << "error in SimAssistantLayout.traceScene: initCPUSubset returned an error" << std::endl;
				return SIMASS_ERROR;
			}

			// transfer starting positions to gui for plotting
			// first we allocate memory for the vertices of each ray ( each ray consists of three double variables... )
			double *l_pRayPlotData = (double*) malloc(l_GPUSubsetDim.x*l_GPUSubsetDim.y*3*sizeof(double));
			// now copy ray data
			for (unsigned long long jy=0;jy<l_GPUSubsetDim.y;jy++)
			{
				for (unsigned long long jx=0;jx<l_GPUSubsetDim.x;jx++)
				{
					l_pRayPlotData[0+jx*3+jy*l_GPUSubsetDim.x*3]=SourceListPtrPtr->getRay(jx+jy*SourceListPtrPtr->getSubsetWidthMax())->position.x;
					l_pRayPlotData[1+jx*3+jy*l_GPUSubsetDim.x*3]=SourceListPtrPtr->getRay(jx+jy*SourceListPtrPtr->getSubsetWidthMax())->position.y;
					l_pRayPlotData[2+jx*3+jy*l_GPUSubsetDim.x*3]=SourceListPtrPtr->getRay(jx+jy*SourceListPtrPtr->getSubsetWidthMax())->position.z;
				}
			}
			RayPlotDataParams l_rayPlotParams;
			l_rayPlotParams.launchDim[0]=width;
			l_rayPlotParams.launchDim[1]=height;
			l_rayPlotParams.launchOffset[0]=SourceListPtrPtr->getParamsPtr()->launchOffsetX;
			l_rayPlotParams.launchOffset[1]=SourceListPtrPtr->getParamsPtr()->launchOffsetY;
			l_rayPlotParams.subsetDim[0]=l_GPUSubsetDim.x;
			l_rayPlotParams.subsetDim[1]=l_GPUSubsetDim.y;
			this->callbackRayPlotData(this->p2CallbackObject, l_pRayPlotData, &l_rayPlotParams);

			//// allocate space for old positions of rays
			//oldPositionsPtr=(double3*)calloc(l_GPUSubsetDim.y*l_GPUSubsetDim.x,sizeof(double3));
			//// save old positions
			//for (unsigned long long jy=0;jy<l_GPUSubsetDim.y;jy++)
			//{
			//	for (unsigned long long jx=0;jx<l_GPUSubsetDim.x;jx++)
			//	{
			//		oldPositionsPtr[jx+jy*l_GPUSubsetDim.x]=SourceListPtrPtr->getRay(jx+jy*SourceListPtrPtr->getSubsetWidthMax())->position;
			//	}
			//}

			/***********************************************
			/	trace rays
			/***********************************************/

			int result;
			while (anythingRunning)
			{
				if (FIELD_NO_ERR != SourceListPtrPtr->traceStep(*oGroupPtr, this->paramsPtr->RunOnCPU) )
				{
					std::cout << "error in SimAssistant.run(): Source.traceStep() returned an error" << std::endl;
					return SIMASS_ERROR;
				}
				anythingRunning=false;
				// transfer new poseitions
				for (unsigned long long jy=0;jy<l_GPUSubsetDim.y;jy++)
				{
					for (unsigned long long jx=0;jx<l_GPUSubsetDim.x;jx++)
					{
						l_pRayPlotData[0+jx*3+jy*l_GPUSubsetDim.x*3]=SourceListPtrPtr->getRay(jx+jy*SourceListPtrPtr->getSubsetWidthMax())->position.x;
						l_pRayPlotData[1+jx*3+jy*l_GPUSubsetDim.x*3]=SourceListPtrPtr->getRay(jx+jy*SourceListPtrPtr->getSubsetWidthMax())->position.y;
						l_pRayPlotData[2+jx*3+jy*l_GPUSubsetDim.x*3]=SourceListPtrPtr->getRay(jx+jy*SourceListPtrPtr->getSubsetWidthMax())->position.z;
						// check wether there is any ray still running
						anythingRunning = anythingRunning || SourceListPtrPtr->getRay(jx+jy*SourceListPtrPtr->getSubsetWidthMax())->running;

					}
				}
				this->callbackRayPlotData(this->p2CallbackObject, l_pRayPlotData, &l_rayPlotParams);
			}

			// clean up
			delete l_pRayPlotData;


			//	// go through the rayList and plot a line from each old position to the recent position
			//	for (unsigned long long jy=0;jy<l_GPUSubsetDim.y;jy++)
			//	{
			//		for (unsigned long long jx=0;jx<l_GPUSubsetDim.x;jx++)
			//		{
			//			// transfer params to matlab variables
			//			mat_oldPos = mxCreateDoubleMatrix(3, 1, mxREAL);
			//			memcpy((char *) mxGetPr(mat_oldPos), (char *) &(oldPositionsPtr[jx+jy*l_GPUSubsetDim.x]), sizeof(double3));
			//			engPutVariable(this->oMatlabInterface.getEnginePtr(), "oldPos", mat_oldPos);
			//			mat_newPos = mxCreateDoubleMatrix(3, 1, mxREAL);
			//			memcpy((char *) mxGetPr(mat_newPos), (char *) &(SourceListPtrPtr->getRay(jx+jy*SourceListPtrPtr->getSubsetHeightMax())->position), sizeof(double3));
			//			engPutVariable(this->oMatlabInterface.getEnginePtr(), "newPos", mat_newPos);
			//			// plot ray
			//			result=engEvalString(this->oMatlabInterface.getEnginePtr(), "line([oldPos(1) newPos(1)], [oldPos(2) newPos(2)], [oldPos(3) newPos(3)]);");
			//			// save new old positions
			//			oldPositionsPtr[jx+jy*l_GPUSubsetDim.x]=SourceListPtrPtr->getRay(jx+jy*SourceListPtrPtr->getSubsetWidthMax())->position;
			//			// check wether there is any ray still running
			//			anythingRunning = anythingRunning || SourceListPtrPtr->getRay(jx+jy*SourceListPtrPtr->getSubsetWidthMax())->running;
			//		}
			//	}

			//}
			//delete oldPositionsPtr;
			//oldPositionsPtr=NULL;

			// increment x-offset
			SourceListPtrPtr->getParamsPtr()->launchOffsetX=SourceListPtrPtr->getParamsPtr()->launchOffsetX+SourceListPtrPtr->getParamsPtr()->GPUSubset_width;				
				
			tracedRayNr=tracedRayNr+l_GPUSubsetDim.x*l_GPUSubsetDim.y;
			std::cout << " " << tracedRayNr <<" out of " << width*height << " rays traced in total" << std::endl;
			//std::cout << std::endl;

			count++;
		}
		// increment y-offset
		SourceListPtrPtr->getParamsPtr()->launchOffsetY=SourceListPtrPtr->getParamsPtr()->launchOffsetY+SourceListPtrPtr->getParamsPtr()->GPUSubset_height;
		// reset x-offset
		SourceListPtrPtr->getParamsPtr()->launchOffsetX=0;
	}
	end = clock();
	msecs=((end-start)/(double)CLOCKS_PER_SEC*1000.0);
	std::cout << std::endl;
	std::cout << "****************************************************** " << std::endl;
	std::cout << "summary:  " << std::endl;
	std::cout << msecs <<"ms to trace and process " << tracedRayNr << " rays." << std::endl;

	return SIMASS_NO_ERROR;
};

simAssError SimAssistantLayout::initSimulation( Group *oGroupPtr, Field *SourceListPtrPtr)
{
	std::cout << "*****************************************************" << std::endl;
	std::cout << "starting to initialize Simulation..." << std::endl;

	clock_t start, end;
	start=clock();

	if (SIMASS_NO_ERROR != SourceListPtrPtr->initLayout(*oGroupPtr, *(this->getParamsPtr())))
	{
		std::cout << "error in SimAssistant.initSimulationBaseClass(): field.initSimulation() returned an error." << std::endl;
		return SIMASS_ERROR;
	}

	end=clock();
	double msecs;
	msecs=((end-start)/(double)CLOCKS_PER_SEC*1000.0);
	std::cout << msecs <<"ms to initialize Simulation." << std::endl;

	return SIMASS_NO_ERROR;
}