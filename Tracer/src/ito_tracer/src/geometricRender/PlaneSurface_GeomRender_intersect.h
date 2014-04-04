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

/**\file PlaneSurface_DiffRays_intersect.h
* \brief header file containing definitions of functions and datastructures that can be used on the GPU as well
* 
*           
* \author Mauch
*/

#ifndef PLANESURFACE_GEOMRENDER_INTERSECT_H
  #define PLANESURFACE_GEOMRENDER_INTERSECT_H

/* include header of basis class */
#include "../PlaneSurface_intersect.h"
#include "../rayTracingMath.h"


/**
 * \detail intersectRayPlaneSurface_DiffRays 
 *
 * calculates the intersection of a differential ray with an cylindrical pipe
 *
 * \param[in] double3 rayPosition, double3 rayDirection, PlaneSurface_ReducedParams params
 * 
 * \return double t
 * \sa 
 * \remarks this function is defined inline so it can be used on GPU and CPU
 * \author Mauch
 */
inline RT_HOSTDEVICE double intersectRayPlaneSurface_GeomRender(double3 rayPosition, double3 rayDirection, PlaneSurface_ReducedParams params)
{
	return intersectRayPlaneSurface(rayPosition, rayDirection, params);
};


#endif
