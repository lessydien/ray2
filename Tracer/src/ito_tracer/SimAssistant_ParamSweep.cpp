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

/**\file SimAssistantParamSweep.cpp
* \brief base class of detectors
* 
*           
* \author Mauch
*/

#include "SimAssistant_ParamSweep.h"
#include "myUtil.h"
#include "sampleConfig.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "Geometry.h"
#include "math.h"
#include "randomGenerator.h"
#include <ctime>
#include "IntensityField_Stack.h"
#include "Geometry.h"
#include "PlaneSurface.h"
#include "DetectorLib.h"

/**
 * \detail setPathToPtx 
 *
 * sets the path to the ptx file that the .cu file defining the behaviour of the SimAssistantParamSweep on the GPU of the surface will be compiled to
 *
 * \param[in] char* path
 * 
 * \return void
 * \sa 
 * \remarks 
 * \author Mauch
 */
void SimAssistantParamSweep::setPathToPtx(char* path)
{
	memcpy(this->path_to_ptx, path, sizeof(this->path_to_ptx));
};

/**
 * \detail getPathToPtx 
 *
 * returns the path to the ptx file that the .cu file defining the behaviour of the SimAssistantParamSweep on the GPU of the surface will be compiled to
 *
 * \param[in] void
 * 
 * \return const char*
 * \sa 
 * \remarks 
 * \author Mauch
 */
const char* SimAssistantParamSweep::getPathToPtx(void)
{
	return this->path_to_ptx;
};

/**
 * \detail setDetParamsPtr 
 *
 * \param[in] detParams *paramsPtr
 * 
 * \return void
 * \sa 
 * \remarks 
 * \author Mauch
 */
void SimAssistantParamSweep::setParamsPtr(simAssParams *paramsPtrIn)
{
	this->paramsPtr=static_cast<simAssParamSweepParams *>(paramsPtrIn);
};

/**
 * \detail getParamsPtr 
 *
 * \param[in] void
 * 
 * \return simAssParams*
 * \sa 
 * \remarks 
 * \author Mauch
 */
simAssParamSweepParams* SimAssistantParamSweep::getParamsPtr(void)
{
	return this->paramsPtr;
};

simAssError SimAssistantParamSweep::updateSimulation( Group *oGroupPtr, RayField *SourceListPtrPtr, simAssParams *paramsPtr)
{
//	if (paramsPtr->RunOnCPU)
//	{
//		SourceListPtrPtr->createCPUSimInstance();
//		if (GROUP_NO_ERR != oGroupPtr->createCPUSimInstance(SourceListPtrPtr->getParamsPtr()->lambda, paramsPtr->mode) )
//		{
//			std::cout << "error in SimAssistantParamSweep.updateSimulation(): group.createCPUSimInstance() returned an error" << std::endl;
//			return SIMASS_ERROR;
//		}
//	}
//	else
//	{
//			// convert geometry to GPU code
//			if ( GROUP_NO_ERR != oGroupPtr->updateOptixInstance(context, paramsPtr->mode, SourceListPtrPtr->getParamsPtr()->lambda) )
//			{
//				std::cout << "error in SimAssistantParamSweep.updateSimulation(): group.updateOptixInstance returned an error" << std::endl;
//				return ( SIMASS_ERROR );
//			}
//				
//			// convert rayfield to GPU code
////			initRayField_AsphereTestGPU( context, hfileWaveFront_Qxy, hfileWaveFront_Pxy, SourceList, RadiusSourceReference, zSourceReference, MNmn, width, height, lambda);
//			if ( FIELD_NO_ERR != (SourceListPtrPtr)->createOptixInstance(context, output_buffer_obj, seed_buffer_obj) )
//			{
//				std::cout << "error in SimAssistantParamSweep.updateSimulation(): SourceList[i]-updateOptixInstance returned an error at index:" << 0 << std::endl;
//				return ( SIMASS_ERROR );
//			}
////			RT_CHECK_ERROR_NOEXIT( rtContextValidate( context ) );
////			RT_CHECK_ERROR_NOEXIT( rtContextCompile( context ) );
//	}

	return SIMASS_NO_ERROR;
}


/**
 * \detail initSimulation 
 *
 * \param[in] Group *oGroupPtr, RayField *SourceListPtrPtr
 * 
 * \return simAssError
 * \sa 
 * \remarks 
 * \author Mauch
 */
simAssError SimAssistantParamSweep::initSimulation( Group *oGroupPtr, Field *SourceListPtrPtr)
{
//	initSimulationBaseClass(oGroupPtr, SourceListPtrPtr, this->paramsPtr);
	return SIMASS_NO_ERROR;
};

/**
 * \detail run
 *
 * runs the simulation
 *
 * \param[in] Group *oGroupPtr, RayField *SourceListPtrPtr, Detector **DetectorListPtrPtr
 * 
 * \return simAssError
 * \sa 
 * \remarks 
 * \author Mauch
 */
simAssError SimAssistantParamSweep::run(Group *oGroupPtr, RayField *SourceListPtrPtr, Detector **DetectorListPtrPtr)
{
//	RTsize				buffer_width, buffer_height; // get size of output buffer
//	void				*data; // pointer to cast output buffer into
// 	rayStruct			*bufferData;
//
//	clock_t start, end;
//	double msecs;
//
//	// we do not allow for a raydata-detector in paramsweep-mode as this would consume memory too fast...
//	if (dynamic_cast<DetectorRaydata*>(DetectorListPtrPtr[0]) != NULL)
//	{
//		std::cout << "error in SimAssistantParamSweep.run(): DetectorRaydata is not allowed param sweep mode" << std::endl;
//		return SIMASS_ERROR;
//	}
//	std::cout << "*****************************************************" << std::endl;
//	std::cout << "starting parameter sweep..." << std::endl;
//	unsigned long long jGeoms;
//	unsigned long long jSrc;
//	unsigned long long jDet;
//	// loop over geoemtryParams
//	for (jGeoms=0;jGeoms<=this->paramsPtr->geomParamsSweepLength;jGeoms++)
//	{
//		if (jGeoms>0)
//		{
//			// set new set of params for Geometry
//			(oGroupPtr->getGeometryGroup(0)->getGeometry(paramsPtr->geometrySweepParamsList[jGeoms-1]->geomObjectIndex))->setParams(paramsPtr->geometrySweepParamsList[jGeoms-1]->geometryParams);
//		}
//		// loop over sourceParams
//		for (jSrc=0;jSrc<=this->paramsPtr->srcParamsSweepLength;jSrc++)
//		{
//			if (jSrc>0)
//			{
//				// update source
//				SourceListPtrPtr->setParamsPtr(this->paramsPtr->srcParamsList[jSrc]);
//			}
//			// loop over detectorParams
//			for (jDet=0;jDet<=this->paramsPtr->detParamsSweepLength;jDet++)
//			{
//				unsigned long long paramSweepNr=jDet+jSrc*(this->paramsPtr->detParamsSweepLength+1)+jGeoms*(this->paramsPtr->detParamsSweepLength+this->paramsPtr->srcParamsSweepLength+1);
//				std::cout << "param sweep number " << paramSweepNr << " out of " << this->paramsPtr->detParamsSweepLength*this->paramsPtr->srcParamsSweepLength*this->paramsPtr->geomParamsSweepLength << std::endl;
//
//				if (jDet>0)
//				{
//					// update detector
//					DetectorListPtrPtr[this->paramsPtr->detIndex]->setDetParamsPtr(this->paramsPtr->detParamsList[jDet]);
//				}
//				unsigned long long width=SourceListPtrPtr->getParamsPtr()->width;
//				unsigned long long height=SourceListPtrPtr->getParamsPtr()->height;
//
//				// update the simulation. The first simulation runs with the original parameters, so we don't have to update..
//				if (paramSweepNr>0)
//					this->updateSimulation(oGroupPtr, SourceListPtrPtr, this->paramsPtr);
//
//				this->doSim(*oGroupPtr, SourceListPtrPtr, DetectorListPtrPtr, &(this->FieldPtrList[jGeoms+jSrc*this->paramsPtr->geomParamsSweepLength+jDet*this->paramsPtr->geomParamsSweepLength*this->paramsPtr->srcParamsSweepLength]), this->paramsPtr->RunOnCPU, context, output_buffer_obj, seed_buffer_obj);				
//
//			} // end loop detectors
//		} // end loop sources
//	} // end loop geometries
//
//	if (!this->paramsPtr->RunOnCPU)
//	{
//		// clean up
//		if (!RT_CHECK_ERROR_NOEXIT( rtContextDestroy( context ), context ))
//			return SIMASS_ERROR;
//	}
//
//	/********************************************
//	*		   output results					*
//	********************************************/
////	FILE* hFileOut;
//	char filepath[512];
//	//sprintf(filepath, "%s" PATH_SEPARATOR "%s", OUTPUT_FILEPATH, "ParamSweepResults.TXT");
//	sprintf(filepath, "%s", OUTPUT_FILEPATH);
////	hFileOut = fopen( filepath, "w" ) ;
////	if (!hFileOut)
////	{
////		std::cout << "error in SimAssistantParamSweep.run(): could not open output file: " << filepath << std::endl;
////		return SIMASS_ERROR;
////	}
//
//	std::cout << "**************************************************************" << std::endl;
//	std::cout << "starting to save simulation results..." << std::endl;
//
//	start=clock();
//	// loop over geoemtryParams
//	for (jGeoms=0;jGeoms<=this->paramsPtr->geomParamsSweepLength;jGeoms++)
//	{
//		// loop over sourceParams
//		for (jSrc=0;jSrc<=this->paramsPtr->srcParamsSweepLength;jSrc++)
//		{
//			// loop over detectorParams
//			for (jDet=0;jDet<=this->paramsPtr->detParamsSweepLength;jDet++)
//			{
//				if (FIELD_NO_ERR != (this->FieldPtrList[jDet+jSrc*(this->paramsPtr->detParamsSweepLength+1)+jGeoms*(this->paramsPtr->detParamsSweepLength+this->paramsPtr->srcParamsSweepLength+1)])->write2File(filepath,*(DetectorListPtrPtr[0]->getDetParamsPtr())) )//if (FIELD_NO_ERR != (this->FieldPtrList[jDet+jSrc*(this->paramsPtr->detParamsSweepLength+1)+jGeoms*(this->paramsPtr->detParamsSweepLength+this->paramsPtr->srcParamsSweepLength+1)])->write2TextFile(hFileOut,*(DetectorListPtrPtr[0]->getDetParamsPtr())) )
//				{
//					std::cout << "error in SimAssistantParamSweep.run(): Field.write2File() returned an error" << std::endl;
//					return SIMASS_ERROR;
//				}
//			} // end loop detectorParams
//		} // end loop sourceParams
//	} // end geometryParams
//	end =clock();
//	msecs=((end-start)/(double)CLOCKS_PER_SEC*1000.0);
//	std::cout << msecs <<"ms to save data." << std::endl;

	return SIMASS_NO_ERROR;
};