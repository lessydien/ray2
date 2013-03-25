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
#include "CylLenseSurface_Intersect.h"

rtDeclareVariable(float3, boxmin, , );
rtDeclareVariable(float3, boxmax, , );
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(rayStruct, prd, rtPayload, ); // get per-ray-data structure
rtDeclareVariable(simMode, mode, , );
rtDeclareVariable(CylLenseSurface_ReducedParams, params, , ); // normal vector to surface. i.e. part of the definition of the plane surface geometry
rtDeclareVariable(int, materialListLength, , ); 
// variables that are communicate to the hit program via the attribute mechanism
rtDeclareVariable(Mat_hitParams, hitParams, attribute hitParams, ); // normal to the geometry at the hit-point. at a plane surface this will simply be the normal of the definition of the plane surface
rtDeclareVariable(int,               geometryID, attribute geometryID , );
rtDeclareVariable(double, t_hit, attribute t_hit, ); 

/* calc normal to surface at intersection point */
__forceinline__ __device__ Mat_hitParams calcHitParams(double t)
{
  return calcHitParamsCylLenseSurface(prd.position+t*prd.direction, params);
}

/* calc intersection of ray with geometry */
RT_PROGRAM void intersect(int)
{
  double t;
  // we only calculate the intersection in nonsequential mode or if the current geometry is the next to intersect in "sequential mode"
//  if ( (mode==SIM_GEOMRAYS_NONSEQ) || (prd.currentGeometryID==params.geometryID-1) )
//  {
	  t = intersectRayCylLenseSurface(prd.position, prd.direction, params);

	  // check wether intersection lies within valid interval of t_hit
	  if( rtPotentialIntersection( (float)t ) ) 
	  {
		hitParams=calcHitParams(t);
		// communicate t_hit to closest_hit function
		t_hit=t;
		// pass geometryID to hit-program
		geometryID=params.geometryID;
		// call any hit function of the respective material
		rtReportIntersection( 0 );
//	  }
  }
}

RT_PROGRAM void bounds (int, float result[6])
{
  optix::Aabb* aabb = (optix::Aabb*)result;
  aabb->set(boxmin, boxmax);
}
