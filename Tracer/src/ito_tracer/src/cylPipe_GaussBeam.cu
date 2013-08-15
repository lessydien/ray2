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

#include <optix.h>
#include <optix_math.h>
#include <optixu/optixu_matrix.h>
#include <optixu/optixu_aabb.h>
#include "rayData.h"
#include "CylPipe_Intersect.h"

rtDeclareVariable(float3, boxmin, , );
rtDeclareVariable(float3, boxmax, , );
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(gaussBeamRayStruct, prd, rtPayload, ); // get per-ray-data structure
rtDeclareVariable(CylPipe_ReducedParams, params, , ); // normal vector to surface. i.e. part of the definition of the plane surface geometry
// variables that are communicate to the hit program via the attribute mechanism
rtDeclareVariable(gaussBeam_geometricNormal, geometric_normal, attribute geometric_normal, ); // normal to the geometry at the hit-point. at a plane surface this will simply be the normal of the definition of the plane surface
rtDeclareVariable(gaussBeam_t, t_hit, attribute t_hit, ); 
rtDeclareVariable(int,               geometryID, attribute geometryID , );

/* calc normal to surface at intersection point */
__forceinline__ __device__ gaussBeam_geometricNormal calcHitParams(gaussBeam_t t)
{
  gaussBeam_geometricNormal normal;
  normal.normal_baseRay=make_double3(0,0,0);
  normal.normal_waistRayX=make_double3(0,0,0);
  normal.normal_waistRayY=make_double3(0,0,0);
  normal.normal_divRayX=make_double3(0,0,0);
  normal.normal_divRayY=make_double3(0,0,0);
  return normal;
}

/* calc intersection of ray with geometry */
RT_PROGRAM void intersect(int)
{
  gaussBeam_t t;

  t.t_baseRay = intersectRayCylPipe(prd.baseRay.position, prd.baseRay.direction, params);

  // check wether intersection lies within valid interval of t_hit
  if( rtPotentialIntersection( (float)t.t_baseRay ) ) 
  {
    // infinite thickness is represented by a value of zero. For these rays we set the aperture to infinity
    params.thickness=0;
	// intersect all the other rays
	t.t_waistRayX = intersectRayCylPipe(prd.waistRayX.position, prd.waistRayX.direction, params);
	t.t_waistRayY = intersectRayCylPipe(prd.waistRayY.position, prd.waistRayY.direction, params);
	t.t_divRayX = intersectRayCylPipe(prd.divRayX.position, prd.divRayX.direction, params);
	t.t_divRayY = intersectRayCylPipe(prd.divRayY.position, prd.divRayY.direction, params);
	/**************************************************************************************************************/
	/* what should we do if one of the rays doesn't hit the geometry, the centre ray hit ?                        */
	/* So far we call the hit functions anyway and terminate the ray with an error in the closest hit function    */
	/**************************************************************************************************************/
	geometric_normal=calcHitParams(t);
	// communicate t_hit to closest_hit function
	t_hit=t;
	// pass geometryID to hit-program
	geometryID=params.geometryID;
	// call any hit function with material indexed zero
	rtReportIntersection(0);
  }

}

RT_PROGRAM void bounds (int, float result[6])
{
  optix::Aabb* aabb = (optix::Aabb*)result;
  aabb->set(boxmin, boxmax);
}
