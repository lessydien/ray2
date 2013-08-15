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

/**\file MaterialDiffracting_DiffRays.cpp
* \brief refracting material
* 
*           
* \author Mauch
*/

#include "MaterialDiffracting_DiffRays.h"
#include "../GlobalConstants.h"
#include "../myUtil.h"
#include <sampleConfig.h>
#include <iostream>
#include <string.h>
#include "../rayTracingMath.h"



/**
 * \detail hit function of material for geometric rays
 *
 * we call the hit function of the coating first. Then we call the hit function of the material. Finally we call the hit function of the Scatter attached to the material
 *
 * \param[in] diffRayStruct &ray, double3 normal, double t_hit, int geometryID
 * 
 * \return MaterialError
 * \sa 
 * \remarks 
 * \author Mauch
 */
void MaterialDiffracting_DiffRays::hit(diffRayStruct &ray, Mat_DiffRays_hitParams hitParams, double t_hit, int geometryID)
{
	
//	extern Group oGroup;
	bool coat_reflected=false;
	if ( this->coatingPtr->getFullParams()->type != CT_NOCOATING )
		coat_reflected=this->coatingPtr->hit(ray, hitParams);

	if ( hitDiffracting(ray, hitParams, this->params, t_hit, geometryID, coat_reflected) )
	{
		if ( this->scatterPtr->getFullParams()->type != ST_NOSCATTER )
			this->scatterPtr->hit(ray, hitParams);
		//if (ray.depth<MAX_DEPTH_CPU && ray.flux>MIN_FLUX_CPU)
		//{			
		//	oGroup.trace(ray);
		//}
		if (ray.depth>MAX_DEPTH_CPU || ray.flux<MIN_FLUX_CPU)
			ray.running=false;//stop ray

	}

}