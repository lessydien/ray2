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

/**\file SphericalSurface_intersect.h
* \brief header file containing definitions of functions and datastructures that can be used on the GPU as well
* 
*           
* \author Mauch
*/

#ifndef SPHERICALSURFACE_DIFFRAYS_INTERSECT_H
  #define SPHERICALSURFACE_DIFFRAYS_INTERSECT_H
  
/* include geometry lib */
#include "Material_DiffRays_hit.h"
#include "../SphericalSurface_intersect.h"
#include "../rayTracingMath.h"

/* declare class */
/**
  *\class   SphericalSurface_DiffRays_ReducedParams
  *\ingroup Geometry
  *\brief   reduced set of params that is loaded onto the GPU if the tracing is done there
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
class SphericalSurface_DiffRays_ReducedParams : public SphericalSurface_ReducedParams
{
	public:
		double3 tilt;
};

/**
 * \detail intersectRaySphere_DiffRays 
 *
 * calculates the intersection of a differential ray with a spherical surface
 *
 * \param[in] double3 rayPosition, double3 rayDirection, SphericalSurface_DiffRays_ReducedParams params
 * 
 * \return double t
 * \sa 
 * \remarks this function is defined inline so it can be used on GPU and CPU
 * \author Mauch
 */
inline RT_HOSTDEVICE double intersectRaySphere_DiffRays(double3 rayPosition, double3 rayDirection, SphericalSurface_DiffRays_ReducedParams params)
{
	return intersectRaySphere(rayPosition, rayDirection, params);
};

/**
 * \detail calcHitParamsSphere_DiffRays 
 *
 * calculates the parameters of the surface at the intersection point that are needed within the hit-function
 *
 * \param[in] double3 position,SphericalSurface_DiffRays_ReducedParams params
 * 
 * \return Mat_DiffRays_hitParams
 * \sa 
 * \remarks this function is defined inline so it can be used on GPU and CPU
 * \author Mauch
 */
inline RT_HOSTDEVICE Mat_DiffRays_hitParams calcHitParamsSphere_DiffRays(double3 position,SphericalSurface_DiffRays_ReducedParams params)
{
	Mat_DiffRays_hitParams t_hitParams_DiffRays;
	Mat_hitParams t_hitParams;
	double3 test=position;
	t_hitParams=calcHitParamsSphere(position, params);
	t_hitParams_DiffRays.normal=t_hitParams.normal;
	double2 phi=calcAnglesFromVector(position-params.centre,make_double3(0,0,0));
	t_hitParams_DiffRays.mainDirX=createObliqueVec(make_double2(phi.x,phi.y+M_PI/2));
	t_hitParams_DiffRays.mainDirY=createObliqueVec(make_double2(phi.x+M_PI/2,phi.y));
	t_hitParams_DiffRays.mainRad=(params.curvatureRadius);
	return t_hitParams_DiffRays;
};

#endif
