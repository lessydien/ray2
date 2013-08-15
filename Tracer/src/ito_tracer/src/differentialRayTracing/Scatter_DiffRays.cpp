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

/**\file Scatter_DiffRays.cpp
* \brief base class to all scattering properties that can be applied to surfaces
* 
*           
* \author Mauch
*/

#include "Scatter_DiffRays.h"
#include "../myUtil.h"
#include "sampleConfig.h"
#include <iostream>

ScatterError Scatter_DiffRays::setFullParams(Scatter_DiffRays_Params* ptrIn)
{
	this->fullParamsPtr=ptrIn;
	return SCAT_NO_ERROR;
};

Scatter_DiffRays_Params* Scatter_DiffRays::getFullParams(void)
{
	return this->fullParamsPtr;
};

ScatterError Scatter_DiffRays::setReducedParams(Scatter_DiffRays_ReducedParams* ptrIn)
{
	this->reducedParams=*ptrIn;
	return SCAT_NO_ERROR;
};

Scatter_DiffRays_ReducedParams* Scatter_DiffRays::getReducedParams(void)
{
	return &(this->reducedParams);
};

void Scatter_DiffRays::hit(rayStruct &ray, Mat_hitParams hitParams)
{
	std::cout << "error in Scatter_DiffRays.hit(): hit is not yet implemented for geometric rays for the given scatter. no scatter assumed." << std::endl;
	// dummy function to be overwritten by child class
}

void Scatter_DiffRays::hit(diffRayStruct &ray, Mat_hitParams hitParams)
{
	std::cout << "error in Scatter_DiffRays.hit(): hit is not yet implemented for differential rays for the given scatter. no scatter assumed." << std::endl;
	// dummy function to be overwritten by child class
}

void Scatter_DiffRays::hit(gaussBeamRayStruct &ray, gaussBeam_geometricNormal normal)
{
	std::cout << "error in Scatter_DiffRays.hit(): hit is not yet implemented for gaussian beamlets for the given scatter. no scatter assumed." << std::endl;
	// dummy function to be overwritten by child class
}