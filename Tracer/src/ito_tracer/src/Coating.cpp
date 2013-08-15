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

/**\file Coating.cpp
* \brief base class of coatings for surfaces
* 
*           
* \author Mauch
*/

#include "Coating.h"
#include "myUtil.h"
#include "sampleConfig.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <iostream>

/**
 * \detail processParseResults 
 *
 * sets the parameters of the detector according to the given parse results
 *
 * \param[in] MaterialParseParamStruct &parseResults_Mat
 * 
 * \return detError
 * \sa 
 * \remarks 
 * \author Mauch
 */
CoatingError Coating::processParseResults(MaterialParseParamStruct &parseResults_Mat)
{
	std::cout << "error in Coating.processParseResults(): not defined for the given Coating" << std::endl;
	return COAT_ERROR;
};

/**
 * \detail parseXml 
 *
 * sets the parameters of the scatter according to the given xml node
 *
 * \param[in] xml_node &geometry
 * 
 * \return geometryError
 * \sa 
 * \remarks 
 * \author Mauch
 */
CoatingError Coating::parseXml(pugi::xml_node &geometry)
{
	return COAT_NO_ERROR;
}

/**
 * \detail setPathToPtx 
 *
 * sets the path to the ptx file that the .cu file defining the behaviour of the Coating on the GPU of the surface will be compiled to
 *
 * \param[in] char* path
 * 
 * \return void
 * \sa 
 * \remarks 
 * \author Mauch
 */
void Coating::setPathToPtx(char* path)
{
	memcpy(this->path_to_ptx, path, sizeof(this->path_to_ptx));	
};

/**
 * \detail getPathToPtx 
 *
 * returns the path to the ptx file that the .cu file defining the behaviour of the Coating on the GPU of the surface will be compiled to
 *
 * \param[in] void
 * 
 * \return char* path
 * \sa 
 * \remarks 
 * \author Mauch
 */
char* Coating::getPathToPtx(void)
{
	return this->path_to_ptx;
};

/**
 * \detail createCPUInstance
 *
 * calculates the reduced params of the Coating from its full params for the given wavelength
 *
 * \param[in] double lambda
 * 
 * \return CoatingError
 * \sa 
 * \remarks 
 * \author Mauch
 */
CoatingError Coating::createCPUSimInstance(double lambda)
{
	return COAT_NO_ERROR;	
};

/**
 * \detail createOptiXInstance
 *
 * calculates the reduced params of the Coating from its full params for the given wavelength and creates an OptiX instance of the Coating
 *
 * \param[in] double lambda
 * 
 * \return CoatingError
 * \sa 
 * \remarks 
 * \author Mauch
 */
CoatingError Coating::createOptiXInstance(double lambda, char** path_to_ptx_in)
{
	this->createCPUSimInstance(lambda);
	strcat(*path_to_ptx_in, this->getPathToPtx());
	this->update=false;
	return COAT_NO_ERROR;	
};

CoatingError Coating::setParams2Program( RTcontext context, RTprogram *closest_hit_programPtr, RTvariable *l_scatterParamsPtr)
{
	this->update=false;
	return COAT_NO_ERROR;
}

/**
 * \detail setFullParams
 *
 * \param[in] Coating_FullParams* ptrIn
 * 
 * \return CoatingError
 * \sa 
 * \remarks 
 * \author Mauch
 */
CoatingError Coating::setFullParams(Coating_FullParams* ptrIn)
{
	this->fullParamsPtr=ptrIn;
	return COAT_NO_ERROR;
};

/**
 * \detail getFullParams
 *
 * \param[in] void
 * 
 * \return Coating_FullParams* ptrIn
 * \sa 
 * \remarks 
 * \author Mauch
 */
Coating_FullParams* Coating::getFullParams(void)
{
	return this->fullParamsPtr;
};

/**
 * \detail getReducedParams
 *
 * \param[in] void
 * 
 * \return Coating_ReducedParams* ptrIn
 * \sa 
 * \remarks 
 * \author Mauch
 */
Coating_ReducedParams* Coating::getReducedParams(void)
{
	return this->reducedParamsPtr;
}

/**
 * \detail hit function of the Coating for geometric rays
 *
 * \param[in] rayStruct &ray, double3 normal
 * 
 * \return bool
 * \sa 
 * \remarks 
 * \author Mauch
 */
bool Coating::hit(rayStruct &ray, Mat_hitParams hitParams)
{
	// dummy function to be overwritten by child class
	std::cout << "error in Coating.hit(): hit is not yet implemented for geometric rays for the given coating" << std::endl;
	return false;
}

/**
 * \detail hit function of the Coating for differential rays
 *
 * \param[in] diffRayStruct &ray, double3 normal
 * 
 * \return bool
 * \sa 
 * \remarks 
 * \author Mauch
 */
bool Coating::hit(diffRayStruct &ray, Mat_hitParams hitParams)
{
	// dummy function to be overwritten by child class
	std::cout << "error in Coating.hit(): hit is not yet implemented for differential rays for the given coating. no reflection assumed." << std::endl;
	return false;
}

/**
 * \detail hit function of the Coating for gaussian beam rays
 *
 * \param[in] gaussBeamRayStruct &ray, gaussBeam_geometricNormal normal
 * 
 * \return bool
 * \sa 
 * \remarks 
 * \author Mauch
 */
bool Coating::hit(gaussBeamRayStruct &ray, gaussBeam_geometricNormal normal)
{
	// dummy function to be overwritten by child class
	std::cout << "error in Coating.hit(): hit is not yet implemented for gaussian beamlets for the given coating. no reflection assumed." << std::endl;
	return false;
}

/**
 * \detail calcCoatingCoeffs
 *
 * calc the transmission and reflection coefficients of the Coating for the given wavelentgh and the incident ray
 *
 * \param[in] gaussBeamRayStruct &ray, gaussBeam_geometricNormal normal
 * 
 * \return bool
 * \sa 
 * \remarks 
 * \author Mauch
 */
CoatingError Coating::calcCoatingCoeffs(double lambda, double3 normal, double3 direction)
{
	return COAT_NO_ERROR;
};