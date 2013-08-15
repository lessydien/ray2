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

/**\file SimAssistant.cpp
* \brief base class of detectors
* 
*           
* \author Mauch
*/

/**
 *\defgroup SimAssistant
 */

#include "myUtil.h"
#include "sampleConfig.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "Geometry.h"
#include "math.h"
#include "randomGenerator.h"
#include "SimAssistant.h"
#include <ctime>
#include <iostream>

/**
 * \detail setPathToPtx 
 *
 * sets the path to the ptx file that the .cu file defining the behaviour of the SimAssistant on the GPU of the surface will be compiled to
 *
 * \param[in] char* path
 * 
 * \return void
 * \sa 
 * \remarks 
 * \author Mauch
 */
void SimAssistant::setPathToPtx(char* path)
{
	memcpy(this->path_to_ptx, path, sizeof(this->path_to_ptx));
};

/**
 * \detail getPathToPtx 
 *
 * returns the path to the ptx file that the .cu file defining the behaviour of the SimAssistant on the GPU of the surface will be compiled to
 *
 * \param[in] void
 * 
 * \return const char*
 * \sa 
 * \remarks 
 * \author Mauch
 */
const char* SimAssistant::getPathToPtx(void)
{
	return this->path_to_ptx;
};

/**
 * \detail setDetParamsPtr 
 *
 * \param[in] simAssParams *paramsPtr
 * 
 * \return void
 * \sa 
 * \remarks 
 * \author Mauch
 */
void SimAssistant::setParamsPtr(simAssParams *paramsPtr)
{
	this->paramsPtr=paramsPtr;
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
simAssParams* SimAssistant::getParamsPtr(void)
{
	return this->paramsPtr;
};

/**
 * \detail run
 *
 * runs the simulation
 *
 * \param[in] 
 * 
 * \return detError
 * \sa 
 * \remarks 
 * \author Mauch
 */
simAssError SimAssistant::run(Group *oGroupPtr, Field *SourceListPtrPtr, Detector **DetectorListPtrPtr)
{
	std::cout << "error in SimAssistant.run(): not implemented yet for given SimAssistant" << std::endl;
	return SIMASS_ERROR;
};

//simAssError SimAssistant::initSimulationBaseClass( Group *oGroupPtr, Field *SourceListPtrPtr, simAssParams *paramsPtr)
//{
//	if (SIMASS_NO_ERROR != SourceListPtrPtr->initSimulation(*oGroupPtr, *paramsPtr))
//	{
//		std::cout << "error in SimAssistant.initSimulationBaseClass(): field.initSimulation() returned an error." << std::endl;
//		return SIMASS_ERROR;
//	}
////	RTprogram			exception_program;
////	if (paramsPtr->RunOnCPU)
////	{
////		SourceListPtrPtr->createCPUSimInstance();
////		if (GROUP_NO_ERR != oGroupPtr->createCPUSimInstance(SourceListPtrPtr->getParamsPtr()->lambda, paramsPtr->mode) )
////		{
////			std::cout << "error in SimAssistant.initSimulation(): group.createCPUSimInstance() returned an error" << std::endl;
////			return SIMASS_ERROR;
////		}
////	}
////	else
////	{
////		// create Context with maximum size
////		if (SIMASS_NO_ERROR != createOptiXContext() )
////		{
////			std::cout << "error in SimAssistant.initSimulation(): createOptiXContext() returned an error" << std::endl;
////			return SIMASS_ERROR;
////		}
////
////			int devices[3];
////			devices[0]=rtContextGetDevices(context, &(devices[0]));
////			rtContextSetDevices(context, 1, &(devices[0]));
////
////			/* Setup state */
////			rtContextSetExceptionEnabled(context, RT_EXCEPTION_ALL, 1);
////			char path_to_ptx_exception[512];
////			sprintf( path_to_ptx_exception, "%s" PATH_SEPARATOR "%s", PATH_TO_PTX, "ITO-MacroSim_generated_exception.cu.ptx" );
////			RT_CHECK_ERROR_NOEXIT( rtProgramCreateFromPTXFile( context, path_to_ptx_exception, "exception_program", &exception_program ), context );
////			rtContextSetExceptionProgram( context,0, exception_program );
////
////			rtContextSetPrintEnabled(context, 1);
////			rtContextSetPrintBufferSize(context, 14096 );
////			rtContextSetPrintLaunchIndex(context, -1, 0, 0);
////			//rtPrintf("test");
////
////			rtContextSetStackSize(context, 1536);
////			//rtContextGetStackSize(context, &stack_size_bytes);
////
////			// convert geometry to GPU code
////			if ( GROUP_NO_ERR != oGroupPtr->createOptixInstance(context, paramsPtr->mode, SourceListPtrPtr->getParamsPtr()->lambda) )
////			{
////				std::cout << "error in SimAssistant.initSimulation(): group.createOptixInstance returned an error" << std::endl;
////				return ( SIMASS_ERROR );
////			}
////				
////			// convert rayfield to GPU code
//////			initRayField_AsphereTestGPU( context, hfileWaveFront_Qxy, hfileWaveFront_Pxy, SourceList, RadiusSourceReference, zSourceReference, MNmn, width, height, lambda);
////			if ( FIELD_NO_ERR != (SourceListPtrPtr)->createOptixInstance(context, output_buffer_obj, seed_buffer_obj) )
////			{
////				std::cout << "error in SimAssistant.initSimulation(): SourceList[i]->createOptixInstance returned an error at index:" << 0 << std::endl;
////				return ( SIMASS_ERROR );
////			}
////			if (!RT_CHECK_ERROR_NOEXIT( rtContextValidate( context ), context ))
////				return SIMASS_ERROR;
////			if (!RT_CHECK_ERROR_NOEXIT( rtContextCompile( context ), context ))
////				return SIMASS_ERROR;
////	}
//
//	return SIMASS_NO_ERROR;
//}

simAssError SimAssistant::initSimulation( Group *oGroupPtr, Field *SourceListPtrPtr)
{
	std::cout << "*****************************************************" << std::endl;
	std::cout << "starting to initialize Simulation..." << std::endl;

	clock_t start, end;
	start=clock();

	if (SIMASS_NO_ERROR != SourceListPtrPtr->initSimulation(*oGroupPtr, *(this->getParamsPtr())))
	{
		std::cout << "error in SimAssistant.initSimulation(): field.initSimulation() returned an error." << std::endl;
		return SIMASS_ERROR;
	}

	end=clock();
	double msecs;
	msecs=((end-start)/(double)CLOCKS_PER_SEC*1000.0);
	std::cout << msecs <<"ms to initialize Simulation." << std::endl;

	return SIMASS_NO_ERROR;
}

//simAssError SimAssistant::createOptiXContext()
//{
//	RTprogram  miss_program;
//    //RTvariable output_buffer;
//
//    /* variables for the miss program */
//
//    /* Setup context */
//    if (!RT_CHECK_ERROR_NOEXIT( rtContextCreate( &context ), context ))
//		return SIMASS_ERROR;
//    if (!RT_CHECK_ERROR_NOEXIT( rtContextSetRayTypeCount( context, 1 ), context )) /* shadow and radiance */
//		return SIMASS_ERROR;
//    if (!RT_CHECK_ERROR_NOEXIT( rtContextSetEntryPointCount( context, 1 ), context ))
//		return SIMASS_ERROR;
//
//	//rtContextSetExceptionEnabled(context, RT_EXCEPTION_ALL, 1);
//	//rtContextSetPrintEnabled(context, 1);
//	//rtContextSetPrintBufferSize(context, 14096 );
//	//rtContextSetPrintLaunchIndex(context, -1, 0, 0);
//
//	char* path_to_ptx;
//	path_to_ptx=(char*)malloc(512*sizeof(char));
//    /* Miss program */
//	sprintf( path_to_ptx, "%s" PATH_SEPARATOR "%s", PATH_TO_PTX, "ITO-MacroSim_generated_missFunction.cu.ptx" );
//    if (!RT_CHECK_ERROR_NOEXIT( rtProgramCreateFromPTXFile( context, path_to_ptx, "miss", &miss_program ), context ))
//		return SIMASS_ERROR;
//    if (!RT_CHECK_ERROR_NOEXIT( rtContextSetMissProgram( context, 0, miss_program ), context ))
//		return SIMASS_ERROR;
//
//	rtContextSetStackSize(context, 1536);
//	//rtContextGetStackSize(context, &stack_size_bytes);
//
//	delete path_to_ptx;
//	return SIMASS_NO_ERROR;
//}

void SimAssistant::setCallback(void* p2CallbackObjectIn, void (*callbackProgressIn)(void* p2Object, int progressValue))
{
	this->p2CallbackObject=p2CallbackObjectIn;
	this->callbackProgress=callbackProgressIn;
}

//simAssError SimAssistant::doSim(Group &oGroup, Field *SourceListPtrPtr, Detector **DetectorListPtrPtr, Field **ResultFieldPtrPtr, bool RunOnCPU, RTcontext &context, RTbuffer &output_buffer_obj, RTbuffer &seed_buffer_obj)
simAssError SimAssistant::doSim(Group &oGroup, Field *SourceListPtrPtr, Detector **DetectorListPtrPtr, Field **ResultFieldPtrPtr, bool RunOnCPU)
{

	//clock_t start, end;
	//double msecs=0;

	bool simDone=false;
	unsigned long long count=0;
	while (!simDone)
	{
//		start=clock();
		// do the tracing
		if (FIELD_NO_ERR != SourceListPtrPtr->doSim(oGroup, *this->getParamsPtr(), simDone))
		{
			simDone=true;
			std::cout << "error in SimAssistant.doSim(): Field.doSim() returned an error" << std::endl;
			return SIMASS_ERROR;
		}

		//end = clock();
		//msecs=((end-start)/(double)CLOCKS_PER_SEC*1000.0);
		//std::cout << msecs << "ms for tracing of current subset..." << std::endl;

		/***********************************************
		/	do the detection
		/***********************************************/
		// comment out for benchmark
//		start=clock();
		DetectorListPtrPtr[0]->getDetParamsPtr()->subSetNr=count;

		if (DET_NO_ERROR != DetectorListPtrPtr[0]->detect(SourceListPtrPtr, (ResultFieldPtrPtr)) )
		{
			std::cout << "error in SimAssistant.doSim(): Detector.detect() returned an error" << std::endl;
			return SIMASS_ERROR;
		}
		//end = clock();
		//msecs=((end-start)/(double)CLOCKS_PER_SEC*1000.0);
		//std::cout << msecs << "for detection of current subset..." << std::endl;

		count++;
	}

	return SIMASS_NO_ERROR;
}