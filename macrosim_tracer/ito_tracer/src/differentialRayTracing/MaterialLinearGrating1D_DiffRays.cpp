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

/**\file MaterialLinearGrating1D_DiffRays.cpp
* \brief material of a linear line grating
* 
*           
* \author Mauch
*/

#include "MaterialLinearGrating1D_DiffRays.h"
#include "../GlobalConstants.h"
#include "../myUtil.h"
#include <sampleConfig.h>
#include <iostream>
#include <string.h>

/**
 * \detail hit function of material for differential rays
 *
 * Here we need to call the hit function of the coating first as the grating diffracts different for reflection and for transmission. Then we call hitLinearGrating1D that describes the interaction of the ray with the material and can be called from GPU as well. Finally we call the hit function of the Scatter attached to the material
 *
 * \param[in] rayStruct &ray, double3 normal, double t_hit, int geometryID
 * 
 * \return MaterialError
 * \sa 
 * \remarks 
 * \author Mauch
 */
void MaterialLinearGrating1D_DiffRays::hit(diffRayStruct &ray, Mat_DiffRays_hitParams hitParams, double t_hit, int geometryID)
{
	bool reflected=false; // init flag indicating reflection
	if (this->params.nRefr1==0)
		reflected =true; // check wether we had a reflective grating or not
	if ( this->coatingPtr->getFullParams()->type != CT_NOCOATING )
		reflected=this->coatingPtr->hit(ray, hitParams); // now see wether the coating wants reflection or refraction

	if (hitLinearGrating1D_DiffRays(ray, hitParams, this->params, t_hit, geometryID, reflected) )
	{
		if ( this->scatterPtr->getFullParams()->type != ST_NOSCATTER )
			this->scatterPtr->hit(ray, hitParams);

//		if (ray.depth<MAX_DEPTH_CPU && ray.flux>MIN_FLUX_CPU)
//		{			
//			oGroup.trace(ray);
//		}
		if (ray.depth>MAX_DEPTH_CPU || ray.flux<MIN_FLUX_CPU)
			ray.running=false;//stop ray
	}
};
