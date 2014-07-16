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

/**\file PlaneSurface_intersect.h
* \brief header file containing definitions of functions and datastructures that can be used on the GPU as well
* 
*           
* \author Mauch
*/

#ifndef PLANESURFACEINTERSECT_H
  #define PLANESURFACEINTERSECT_H

/* include header of basis class */
#include "Geometry_intersect.h"
#include <optixu/optixu_aabb.h>
#include "rayTracingMath.h"

/* declare class */
/**
  *\class   PlaneSurface_ReducedParams 
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
class PlaneSurface_ReducedParams : public Geometry_ReducedParams
{
  public:
   double3 root;
   double3 normal;
   double2 apertureRadius;
//   double rotNormal; // rotation of geometry around its normal
   ApertureType apertureType;
   //int geometryID;
};

/**
 * \detail intersectRayPlaneSurface 
 *
 * calculates the intersection of a ray with a plane surface
 *
 * \param[in] double3 rayPosition, double3 rayDirection, PlaneSurface_ReducedParams params
 * 
 * \return double t
 * \sa 
 * \remarks this function is defined inline so it can be used on GPU and CPU
 * \author Mauch
 */
inline RT_HOSTDEVICE double intersectRayPlaneSurface(double3 rayPosition, double3 rayDirection, PlaneSurface_ReducedParams params)
{
//	double3 rayDir, rayPos, planeRoot, planeNormal;
//	rayDir=rayDirection;
//	rayPos=rayPosition;
//	planeRoot=params.root;
//	planeNormal=params.normal;

	double t = intersectRayPlane(rayPosition, rayDirection, params.root, params.normal);

//	double3 test=rayPosition+t*rayDirection;
//	PlaneSurface_ReducedParams testPar=params;
	// check aperture
	if ( !checkAperture(params.root, params.tilt, rayPosition+t*rayDirection, params.apertureType, params.apertureRadius) )
	{
		return 0;
	}
	return t;
}

inline RT_HOSTDEVICE void planeSurfaceBounds (int primIdx, float result[6], PlaneSurface_ReducedParams params)
{
  optix::Aabb* aabb = (optix::Aabb*)result;
  double3 l_ex=make_double3(1,0,0);
  rotateRay(&l_ex,params.tilt);
  double3 l_ey=make_double3(0,1,0);
  rotateRay(&l_ey,params.tilt);

  float3 t_maxBox=make_float3(params.root+params.apertureRadius.x*l_ex+params.apertureRadius.y*l_ey);
  float3 t_minBox=make_float3(params.root-params.apertureRadius.x*l_ex-params.apertureRadius.y*l_ey);

  float3 maxBox=make_float3(max(t_maxBox.x,t_minBox.x), max(t_maxBox.y,t_minBox.y), max(t_maxBox.z, t_minBox.z));
  float3 minBox=make_float3(min(t_maxBox.x,t_minBox.x), min(t_maxBox.y,t_minBox.y), min(t_maxBox.z, t_minBox.z));
  aabb->set(minBox, maxBox);
}


#endif
