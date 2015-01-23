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

/**\file MaterialAbsorbing_DiffRays.cpp
* \brief absorbing material
* 
*           
* \author Mauch
*/

#include "MaterialAbsorbing_DiffRays.h"
#include "../myUtil.h"
#include "sampleConfig.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>


/**
 * \detail hit function of material for differential rays
 *
 * we call hitAbsorbing that describes the interaction of the ray with the material and can be called from GPU as well. Then we call the hit function of the coatin attached to the Material. Finally we call the hit function of the Scatter attached to the material
 *
 * \param[in] rayStruct &ray, double3 normal, double t_hit, int geometryID
 * 
 * \return MaterialError
 * \sa 
 * \remarks 
 * \author Mauch
 */
void MaterialAbsorbing_DiffRays::hit(diffRayStruct &ray, Mat_DiffRays_hitParams hitParams, double t_hit, int geometryID)
{
	//ray.direction=make_double3(0.0);
	if ( hitAbsorbing_DiffRays(ray, hitParams, t_hit, geometryID) )
	{
		ray.running=false;//stop ray
		bool reflected;
		if ( this->coatingPtr->getFullParams()->type != CT_NOCOATING )
		{
			reflected=this->coatingPtr->hit(ray, hitParams);
			if (reflected)
			{
				ray.direction=reflect(ray.direction, hitParams.normal);
				ray.running=true; // keep ray alive
			}
		}

		if ( this->scatterPtr->getFullParams()->type != ST_NOSCATTER )
		{
			this->scatterPtr->hit(ray, hitParams);
			ray.running=true; // keep ray alive
		}
	}
}