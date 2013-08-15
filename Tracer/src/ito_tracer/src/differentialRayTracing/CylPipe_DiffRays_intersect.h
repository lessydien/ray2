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

/**\file CylPipe_intersect.h
* \brief header file containing definitions of functions and datastructures that can be used on the GPU as well
* 
*           
* \author Mauch
*/

#ifndef CYLPIPE_DIFFRAYS_INTERSECT_H
  #define CYLPIPE_DIFFRAYS_INTERSECT_H

#include "Material_DiffRays_hit.h"
#include "../CylPipe_intersect.h"
#include "../rayTracingMath.h"

/* declare class */
/**
  *\class   CylPipe_DiffRays_ReducedParams
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
class CylPipe_DiffRays_ReducedParams : public CylPipe_ReducedParams
{
  public:
	double3 tilt;
};

/**
 * \detail intersectRayCylPipe_DiffRays 
 *
 * calculates the intersection of a differential ray with an cylindrical pipe
 *
 * \param[in] double3 rayPosition, double3 rayDirection, CylPipe_DiffRays_ReducedParams params
 * 
 * \return double t
 * \sa 
 * \remarks this function is defined inline so it can be used on GPU and CPU
 * \author Mauch
 */
inline RT_HOSTDEVICE double intersectRayCylPipe_DiffRays(double3 rayPosition, double3 rayDirection, CylPipe_DiffRays_ReducedParams params)
{
	return intersectRayCylPipe(rayPosition, rayDirection, params);
}

/**
 * \detail calcHitParamsCylPipe_DiffRays 
 *
 * calculates the parameters of the surface at the intersection point that are needed within the hit-function
 *
 * \param[in] double3 position,CylPipe_DiffRays_ReducedParams params
 * 
 * \return Mat_DiffRays_hitParams
 * \sa 
 * \remarks this function is defined inline so it can be used on GPU and CPU
 * \author Mauch
 */
inline RT_HOSTDEVICE Mat_DiffRays_hitParams calcHitParamsCylPipe_DiffRays(double3 position,CylPipe_DiffRays_ReducedParams params)
{
	Mat_DiffRays_hitParams t_hitParams_DiffRays;
	Mat_hitParams t_hitParams;
	t_hitParams=calcHitParamsCylPipe(position, params);
	t_hitParams_DiffRays.normal=t_hitParams.normal;
	return t_hitParams_DiffRays;
}

#endif
