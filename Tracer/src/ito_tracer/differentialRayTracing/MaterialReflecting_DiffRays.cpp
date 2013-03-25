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

/**\file MaterialReflecting_DiffRays.cpp
* \brief reflecting material
* 
*           
* \author Mauch
*/

#include "MaterialReflecting_DiffRays.h"
#include "../GlobalConstants.h"
#include "../myUtil.h"
#include <sampleConfig.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

/**
 * \detail hit function of material for geometric rays
 *
 * Here we need to call the hit function of the coating first. If the coating transmits the ray through the material it passes without any further deflectiosn. Finally we call the hit function of the Scatter attached to the material
 *
 * \param[in] rayStruct &ray, double3 normal, double t_hit, int geometryID
 * 
 * \return MaterialError
 * \sa 
 * \remarks 
 * \author Mauch
 */
void MaterialReflecting_DiffRays::hit(diffRayStruct &ray, Mat_DiffRays_hitParams hitParams, double t_hit, int geometryID)
{
//	extern Group oGroup;
	double3 n=hitParams.normal;
	bool coat_reflected = true;
	if ( this->coatingPtr->getFullParams()->type != CT_NOCOATING )
		coat_reflected=this->coatingPtr->hit(ray, hitParams);
	// if the coating wants transmission we do not change the ray direction at all !!!
	if (coat_reflected)
		hitReflecting_DiffRays(ray, hitParams, t_hit, geometryID);
	if ( this->scatterPtr->getFullParams()->type != ST_NOSCATTER )
		this->scatterPtr->hit(ray, hitParams);

	//if (ray.depth<MAX_DEPTH_CPU && ray.flux>MIN_FLUX_CPU)
	//{			
	//	oGroup.trace(ray);
	//}
	if (ray.depth>MAX_DEPTH_CPU || ray.flux<MIN_FLUX_CPU)
		ray.running=false;//stop ray
	ray.running=false;
}
