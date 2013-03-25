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

/**\file IdealLense_intersect.h
* \brief header file containing definitions of functions and datastructures that can be used on the GPU as well
* 
*           
* \author Mauch
*/

#ifndef IDEALLENSE_DIFFRAYS_INTERSECT_H
  #define IDEALLENSE_DIFFRAYS_INTERSECT_H

/* include header of basis class */
#include "Material_DiffRays_hit.h"
#include "../IdealLense_intersect.h"
#include "../rayTracingMath.h"

/* declare class */
/**
  *\class   IdealLense_DiffRays_ReducedParams
  *\ingroup Geometry
  *\brief   reduced set of params that is calculated before the actual tracing from the full set of params. This parameter set will be loaded onto the GPU if the tracing is done there
  *
  *         
  *
  *         \todo
  *         \remarks           
  *         \sa       NA
  *         \date     04.01.2011
  *         \author  Mauch
  *
  */
class IdealLense_DiffRays_ReducedParams : public IdealLense_ReducedParams
{
  public:
	double3 tilt;
};

/**
 * \detail intersectRayIdealLense_DiffRays 
 *
 * calculates the intersection of a differential ray with an cylindrical pipe
 *
 * \param[in] double3 rayPosition, double3 rayDirection, IdealLense_DiffRays_ReducedParams params
 * 
 * \return double t
 * \sa 
 * \remarks this function is defined inline so it can be used on GPU and CPU
 * \author Mauch
 */

inline RT_HOSTDEVICE double intersectRayIdealLense_DiffRays(double3 rayPosition, double3 rayDirection, IdealLense_DiffRays_ReducedParams params)
{
	return intersectRayIdealLense(rayPosition, rayDirection, params);
};

/**
 * \detail calcHitParamsIdealLense_DiffRays 
 *
 * calculates the parameters of the surface at the intersection point that are needed within the hit-function
 *
 * \param[in] double3 position, IdealLense_DiffRays_ReducedParams params
 * 
 * \return Mat_DiffRays_hitParams
 * \sa 
 * \remarks this function is defined inline so it can be used on GPU and CPU
 * \author Mauch
 */
inline RT_HOSTDEVICE Mat_DiffRays_hitParams calcHitParamsIdealLense_DiffRays(double3 position, IdealLense_DiffRays_ReducedParams params)
{
	Mat_DiffRays_hitParams t_hitParams;
	return t_hitParams;
};


#endif
