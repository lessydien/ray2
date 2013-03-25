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

/**\file SimAssistantSingleSim.cpp
* \brief base class of detectors
* 
*           
* \author Mauch
*/

#include "SimAssistant_SingleSim.h"
#include "myUtil.h"
#include "sampleConfig.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "Geometry.h"
#include "math.h"
#include "randomGenerator.h"
#include "Detector.h"
#include "DetectorLib.h"
#include <ctime>

/**
 * \detail setPathToPtx 
 *
 * sets the path to the ptx file that the .cu file defining the behaviour of the SimAssistantSingleSim on the GPU of the surface will be compiled to
 *
 * \param[in] char* path
 * 
 * \return void
 * \sa 
 * \remarks 
 * \author Mauch
 */
void SimAssistantSingleSim::setPathToPtx(char* path)
{
	memcpy(this->path_to_ptx, path, sizeof(this->path_to_ptx));
};

/**
 * \detail getPathToPtx 
 *
 * returns the path to the ptx file that the .cu file defining the behaviour of the SimAssistantSingleSim on the GPU of the surface will be compiled to
 *
 * \param[in] void
 * 
 * \return const char*
 * \sa 
 * \remarks 
 * \author Mauch
 */
const char* SimAssistantSingleSim::getPathToPtx(void)
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
//void SimAssistantSingleSim::setParamsPtr(simAssParams *paramsPtrIn)
//{
//	this->paramsPtr=static_cast<simAssSingleSimParams *>(paramsPtrIn);
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
//simAssSingleSimParams* SimAssistantSingleSim::getParamsPtr(void)
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
//simAssError SimAssistantSingleSim::initSimulation( Group *oGroupPtr, Field *SourceListPtrPtr)
//{
//	std::cout << "*****************************************************" << std::endl;
//	std::cout << "starting to initialize Simulation..." << std::endl;
//
//	clock_t start, end;
//	start=clock();
//
//	if (SIMASS_NO_ERROR != initSimulationBaseClass(oGroupPtr, SourceListPtrPtr, this->paramsPtr))
//	{
//		std::cout << "error in SimAssistantSingleSim.initSimulation(): initSimulationBaseClass() returned an error" << std::endl;
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
simAssError SimAssistantSingleSim::run(Group *oGroupPtr, Field *SourceListPtrPtr, Detector **DetectorListPtrPtr)
{
	//RTsize				buffer_width, buffer_height; // get size of output buffer
	//void				*data; // pointer to cast output buffer into
 //	rayStruct			*bufferData;

	std::cout << "*****************************************************" << std::endl;
	std::cout << "starting simulation..." << std::endl;

	clock_t start, end;
	double msecs_Saving=0;
	double msecs=0;

//	if (SIMASS_NO_ERROR != this->doSim(*oGroupPtr, SourceListPtrPtr, DetectorListPtrPtr, &(this->oFieldPtr), this->paramsPtr->RunOnCPU, context, output_buffer_obj, seed_buffer_obj))
	if (SIMASS_NO_ERROR != this->doSim(*oGroupPtr, SourceListPtrPtr, DetectorListPtrPtr, &(this->oFieldPtr), this->paramsPtr->RunOnCPU))
	{
		std::cout << "error in SimAssistantSingleSim.run(): doSim() returned an error" << std::endl;
		return SIMASS_ERROR;
	}

	/********************************************
	*		   output results					*
	********************************************/
	std::cout << "**************************************************************" << std::endl;
	std::cout << "starting to save simulation results..." << std::endl;

	start=clock();
	char filepath[512];
	sprintf(filepath, "%s", OUTPUT_FILEPATH);
	//sprintf(filepath, "%s", "");

	// increase subset number one more time so that if we were writing raydata into files we do not write the same ray data twice into the last file. instead we write the last subset into two different files...
	DetectorListPtrPtr[0]->getDetParamsPtr()->subSetNr++;
	// comment out for benchmark
	if (FIELD_NO_ERR != this->oFieldPtr->write2File(filepath, *(DetectorListPtrPtr[0]->getDetParamsPtr())) )
	{
		std::cout << "error in SimAssistantSingleSim.run(): Field.write2File() returned an error" << std::endl;
		return SIMASS_ERROR;
	}
	// end comment out
	end =clock();
	msecs_Saving=((end-start)/(double)CLOCKS_PER_SEC*1000.0);
	std::cout << msecs_Saving <<"ms to save data." << std::endl;

	//if (!this->paramsPtr->RunOnCPU)
	//{
	//	// clean up
	//	if (!RT_CHECK_ERROR_NOEXIT( rtContextDestroy( context ), context ))
	//		return SIMASS_ERROR;
	//}

	return SIMASS_NO_ERROR;
};
