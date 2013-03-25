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

/**\file Coating_NumCoeffs_DiffRays.cpp
* \brief Coating with predefined coefficients for transmissione and reflection
* 
*           
* \author Mauch
*/

#include "Coating_NumCoeffs_DiffRays.h"
#include "../myUtil.h"
#include <sampleConfig.h>
#include <string.h>
#include <iostream>

/**
 * \detail createCPUInstance
 *
 * \param[in] double lambda
 * 
 * \return CoatingError
 * \sa 
 * \remarks 
 * \author Mauch
 */
CoatingError Coating_NumCoeffs_DiffRays::createCPUSimInstance(double lambda)
{
	this->reducedParamsPtr->t=this->fullParamsPtr->t;
	this->reducedParamsPtr->r=this->fullParamsPtr->r;
	this->update=false;
	return COAT_NO_ERROR;	
};

/**
 * \detail hit function of the Coating for differential rays
 *
 * \param[in] diffRayStruct &ray, Mat_DiffRays_hitParams hitParams
 * 
 * \return bool
 * \sa 
 * \remarks 
 * \author Mauch
 */
bool  Coating_NumCoeffs_DiffRays::hit(diffRayStruct &ray, Mat_DiffRays_hitParams hitParams)
{
	return hitCoatingNumCoeff_DiffRays(ray, hitParams, *this->reducedParamsPtr);
}
