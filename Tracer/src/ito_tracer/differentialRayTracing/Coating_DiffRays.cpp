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

/**\file Coating_DiffRays.cpp
* \brief base class of coatings for surfaces
* 
*           
* \author Mauch
*/

#include "Coating_DiffRays.h"
#include "../myUtil.h"
#include <sampleConfig.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <iostream>

/**
 * \detail setFullParams
 *
 * \param[in] Coating_DiffRays_FullParams* ptrIn
 * 
 * \return CoatingError
 * \sa 
 * \remarks 
 * \author Mauch
 */
CoatingError Coating_DiffRays::setFullParams(Coating_DiffRays_FullParams* ptrIn)
{
	this->fullParamsPtr=ptrIn;
	return COAT_NO_ERROR;
};

/**
 * \detail getFullParams
 *
 * \param[in] void
 * 
 * \return Coating_DiffRays_FullParams* ptrIn
 * \sa 
 * \remarks 
 * \author Mauch
 */
Coating_DiffRays_FullParams* Coating_DiffRays::getFullParams(void)
{
	return this->fullParamsPtr;
};

/**
 * \detail getReducedParams
 *
 * \param[in] void
 * 
 * \return Coating_DiffRays_ReducedParams* ptrIn
 * \sa 
 * \remarks 
 * \author Mauch
 */
Coating_DiffRays_ReducedParams* Coating_DiffRays::getReducedParams(void)
{
	return this->reducedParamsPtr;
}

/**
 * \detail hit function of the Coating_DiffRays for geometric rays
 *
 * \param[in] rayStruct &ray, double3 normal
 * 
 * \return bool
 * \sa 
 * \remarks 
 * \author Mauch
 */
bool Coating_DiffRays::hit(rayStruct &ray, Mat_hitParams hitParams)
{
	// dummy function to be overwritten by child class
	std::cout << "error in Coating_DiffRays.hit(): hit is not yet implemented for geometric rays for the given coating" << std::endl;
	return false;
}

/**
 * \detail hit function of the Coating_DiffRays for differential rays
 *
 * \param[in] diffRayStruct &ray, double3 normal
 * 
 * \return bool
 * \sa 
 * \remarks 
 * \author Mauch
 */
bool Coating_DiffRays::hit(diffRayStruct &ray, Mat_hitParams hitParams)
{
	// dummy function to be overwritten by child class
	std::cout << "error in Coating_DiffRays.hit(): hit is not yet implemented for differential rays for the given coating. no reflection assumed." << std::endl;
	return false;
}

/**
 * \detail hit function of the Coating_DiffRays for gaussian beam rays
 *
 * \param[in] gaussBeamRayStruct &ray, gaussBeam_geometricNormal normal
 * 
 * \return bool
 * \sa 
 * \remarks 
 * \author Mauch
 */
bool Coating_DiffRays::hit(gaussBeamRayStruct &ray, gaussBeam_geometricNormal normal)
{
	// dummy function to be overwritten by child class
	std::cout << "error in Coating_DiffRays.hit(): hit is not yet implemented for gaussian beamlets for the given coating. no reflection assumed." << std::endl;
	return false;
}

/**
 * \detail calcCoating_DiffRaysCoeffs
 *
 * calc the transmission and reflection coefficients of the Coating_DiffRays for the given wavelentgh and the incident ray
 *
 * \param[in] gaussBeamRayStruct &ray, gaussBeam_geometricNormal normal
 * 
 * \return bool
 * \sa 
 * \remarks 
 * \author Mauch
 */
CoatingError Coating_DiffRays::calcCoating_DiffRaysCoeffs(double lambda, double3 normal, double3 direction)
{
	return COAT_NO_ERROR;
};