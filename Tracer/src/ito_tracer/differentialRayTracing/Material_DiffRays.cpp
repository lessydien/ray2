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

/**\file Material_DiffRays.cpp
* \brief base class of all materials that can be applied to the geometries
* 
*           
* \author Mauch
*/

#include "Material_DiffRays.h"
#include "../myUtil.h"
#include "sampleConfig.h"
#include <iostream>
#include <string.h>

/**
 * \detail hit function of the material for geometric rays
 *
 * \param[in] diffRayStruct &ray, double3 normal, double t, int geometryID
 * 
 * \return void
 * \sa 
 * \remarks 
 * \author Mauch
 */
void Material_DiffRays::hit(diffRayStruct &ray, Mat_DiffRays_hitParams hitParams, double t_hit, int geometryID)
{
	// dummy function to be overwritten by child class
	std::cout << "error in Material_DiffRays.hit(): hit is not yet implemented for differential rays for the given material. Material_DiffRays is ignored..." << std::endl;
};