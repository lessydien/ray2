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

/**\file Pupil.cpp
* \brief base class of all Pupils
* 
*           
* \author Mauch
*/

#include "Pupil.h"
#include "myUtil.h"
#include "sampleConfig.h"
#include <iostream>
#include <string.h>

/**
 * \detail processParseResults 
 *
 * sets the parameters of the detector according to the given parse results
 *
 * \param[in] PupilParseParamStruct &parseResults_Pupil
 * 
 * \return detError
 * \sa 
 * \remarks 
 * \author Mauch
 */
PupilError Pupil::processParseResults(PupilParseParamStruct &parseResults_Pupil)
{
	std::cout << "error in Pupil.processParseResults(): not defined for the given Field representation" << "...\n";
	return PUP_ERR;
};

/**
 * \detail setPathToPtx 
 *
 * sets the path to the ptx file that the .cu file defining the behaviour of the Pupil on the GPU of the surface will be compiled to
 *
 * \param[in] char* path
 * 
 * \return void
 * \sa 
 * \remarks 
 * \author Mauch
 */
void Pupil::setPathToPtx(char* path)
{
	memcpy(this->path_to_ptx, path, sizeof(this->path_to_ptx));	
};

/**
 * \detail getPathToPtx 
 *
 * returns the path to the ptx file that the .cu file defining the behaviour of the Pupil on the GPU of the surface will be compiled to
 *
 * \param[in] void
 * 
 * \return char* path
 * \sa 
 * \remarks 
 * \author Mauch
 */
char* Pupil::getPathToPtx(void)
{
	return this->path_to_ptx;
};

/**
 * \detail createPupilHitProgramPtx 
 *
 * creates a ptx file from the given path
 *
 * \param[in] RTcontext context
 * 
 * \return PupilError
 * \sa 
 * \remarks 
 * \author Mauch
 */
PupilError Pupil::createPupilAimProgramPtx(RTcontext context, SimMode mode)
{
	//if ( (mode.traceMode==SIM_DIFFRAYS_NONSEQ) || (mode.traceMode==TRACE_SEQ) )
	//	strcat(this->path_to_ptx, "_DiffRays");
	//strcat(this->path_to_ptx, ".cu.ptx");
	//RT_CHECK_ERROR_NOEXIT( rtProgramCreateFromPTXFile( context, this->path_to_ptx, "closestHit", &closest_hit_program ) );
 //   RT_CHECK_ERROR_NOEXIT( rtProgramCreateFromPTXFile( context, this->path_to_ptx, "anyHit", &any_hit_program ) );

	return PUP_NO_ERR;
}

/**
 * \detail createOptiXInstance 
 *
 * \param[in] RTcontext context, RTgeometryinstance &instance, int index, TraceMode mode, double lambda
 * 
 * \return PupilError
 * \sa 
 * \remarks 
 * \author Mauch
 */
//PupilError Pupil::createOptiXInstance(RTcontext context, RTgeometryinstance &instance, int index, SimParams simParams, double lambda)
//{
//	std::cout << "error in Pupil.createOptiXInstance(): not defined for the given Pupil" << "...\n";
//	return PupilError;
//};

/**
 * \detail updateOptiXInstance 
 *
 * \param[in] RTcontext context, RTgeometryinstance &instance, int index, TraceMode mode, double lambda
 * 
 * \return PupilError
 * \sa 
 * \remarks 
 * \author Mauch
 */
//PupilError Pupil::updateOptiXInstance(RTcontext context, RTgeometryinstance &instance, int index, SimParams simParams, double lambda)
//{
//	std::cout << "error in Pupil.updateOptiXInstance(): not defined for the given Pupil" << "...\n";
//	return PupilError;
//};

/**
 * \detail createCPUSimInstance 
 *
 * \param[in] RTcontext context, RTgeometryinstance &instance, int index, TraceMode mode, double lambda
 * 
 * \return PupilError
 * \sa 
 * \remarks 
 * \author Mauch
 */
PupilError Pupil::createCPUSimInstance(double lambda)
{
	std::cout << "error in Pupil.createCPUSimInstance(): not defined for the given Pupil" << "...\n";
	return PUP_ERR;
};

/**
 * \detail updateCPUSimInstance 
 *
 * \param[in] RTcontext context, RTgeometryinstance &instance, int index, TraceMode mode, double lambda
 * 
 * \return PupilError
 * \sa 
 * \remarks 
 * \author Mauch
 */
PupilError Pupil::updateCPUSimInstance(double lambda)
{
	std::cout << "error in Pupil.updateCPUSimInstance(): not defined for the given Pupil" << "...\n";
	return PUP_ERR;
};

/**
 * \detail calcSourceImmersion 
 *
 * \param[in] double lambda
 * 
 * \return double nRefr
 * \sa 
 * \remarks 
 * \author Mauch
 */
PupilError Pupil::reduceParams(double lambda)
{
	std::cout << "error in Pupil.reduceParams(): not defined for the given Pupil" << "...\n";
	return PUP_ERR;	// if the function is not overwritten by the child class, we return a standard value of one for the refractive index of the immersion Pupil
};

/**
 * \detail hit function of the Pupil for geometric rays
 *
 * \param[in] rayStruct &ray, double3 normal, double t, int geometryID
 * 
 * \return void
 * \sa 
 * \remarks 
 * \author Mauch
 */
void Pupil::aim(rayStruct &ray, unsigned long long iX, unsigned long long iY)
{
	// dummy function to be overwritten by child class
	std::cout << "error in Pupil.aim(): hit is not yet implemented for geometric rays for the given Pupil. Pupil_DiffRays is ignored..." << "...\n";
};


/**
 * \detail setFullParams
 *
 * \param[in] pupilParams *params
 * 
 * \return void
 * \sa 
 * \remarks 
 * \author Mauch
 */
void Pupil::setFullParamsPtr(pupilParams *params)
{
	std::cout << "error in Pupil.setFullParams(): not defined for the given Pupil" << "...\n";
};

/**
 * \detail getFullParams
 *
 * \param[in] void
 * 
 * \return pupilParams
 * \sa 
 * \remarks 
 * \author Mauch
 */
pupilParams* Pupil::getFullParamsPtr(void)
{
	std::cout << "error in Pupil.getFullParams(): not defined for the given Pupil" << "...\n";
	return NULL;
};
