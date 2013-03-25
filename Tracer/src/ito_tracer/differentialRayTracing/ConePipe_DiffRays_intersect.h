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

/**\file ConePipe_intersect.h
* \brief header file containing definitions of functions and datastructures that can be used on the GPU as well
* 
*           
* \author Mauch
*/

#ifndef CONEPIPE_DIFFRAYS_INTERSECT_H
  #define CONEPIPE_DIFFRAYS_INTERSECT_H
  
/* include header of basis class */
#include "Material_DiffRays_hit.h"
#include "../ConePipe_intersect.h"
#include "../rayTracingMath.h"

/* declare class */
/**
  *\class   ConePipe_DiffRays_ReducedParams 
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
class ConePipe_DiffRays_ReducedParams : public ConePipe_ReducedParams
{
  public:
	double3 tilt;
};

/**
 * \detail intersectRayConePipe_DiffRays 
 *
 * calculates the intersection of a differential ray with an cone pipe
 *
 * \param[in] double3 rayPosition, double3 rayDirection, ConePipe_DiffRays_ReducedParams params
 * 
 * \return double t
 * \sa 
 * \remarks this function is defined inline so it can be used on GPU and CPU
 * \author Mauch
 */
inline RT_HOSTDEVICE double intersectRayConePipe_DiffRays(double3 rayPosition, double3 rayDirection, ConePipe_DiffRays_ReducedParams params)
{
	return intersectRayConePipe(rayPosition, rayDirection, params);
}

/**
 * \detail calcHitParamsConePipe_DiffRays 
 *
 * calculates the parameters of the surface at the intersection point that are needed within the hit-function
 *
 * \param[in] double3 position,ConePipe_DiffRays_ReducedParams params
 * 
 * \return Mat_DiffRays_hitParams
 * \sa 
 * \remarks this function is defined inline so it can be used on GPU and CPU
 * \author Mauch
 */
inline RT_HOSTDEVICE Mat_DiffRays_hitParams calcHitParamsConePipe_DiffRays(double3 position,ConePipe_DiffRays_ReducedParams params)
{
	Mat_DiffRays_hitParams t_hitParams_DiffRays;
	Mat_hitParams t_hitParams;
	t_hitParams=calcHitParamsConePipe(position, params);
	t_hitParams_DiffRays.normal=t_hitParams.normal;
	return t_hitParams_DiffRays;
}

#endif
