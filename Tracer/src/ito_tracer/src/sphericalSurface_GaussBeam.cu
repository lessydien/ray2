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
#include "SphericalSurface_Intersect.h"

rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(gaussBeamRayStruct, prd, rtPayload, ); // get per-ray-data structure
rtDeclareVariable(SphericalSurface_ReducedParams, params, , ); // centre of spherical surface
// variables that are communicated to hit program vie attribute mechanism
rtDeclareVariable(gaussBeam_geometricNormal, geometric_normal, attribute geometric_normal, ); 
rtDeclareVariable(gaussBeam_t, t_hit, attribute t_hit, ); 
rtDeclareVariable(int,               geometryID, attribute geometryID , );

__device__ gaussBeam_geometricNormal calcHitParams(gaussBeam_t t)
{
  gaussBeam_geometricNormal normal;
  double3 hit_point;
  // centre ray
  hit_point=prd.baseRay.position+t.t_baseRay*prd.baseRay.direction;
  normal.normal_baseRay=-((hit_point-params.centre)/params.curvatureRadius.x);
  // waist rays
  hit_point=prd.waistRayX.position+t.t_waistRayX*prd.waistRayX.direction;
  normal.normal_waistRayX=-((hit_point-params.centre)/params.curvatureRadius.x);
  hit_point=prd.waistRayY.position+t.t_waistRayY*prd.waistRayY.direction;
  normal.normal_waistRayY=-((hit_point-params.centre)/params.curvatureRadius.x);
  // div rays
  hit_point=prd.divRayX.position+t.t_divRayX*prd.divRayX.direction;
  normal.normal_divRayX=-((hit_point-params.centre)/params.curvatureRadius.x);
  hit_point=prd.divRayY.position+t.t_divRayY*prd.divRayY.direction;
  normal.normal_divRayY=-((hit_point-params.centre)/params.curvatureRadius.x);
  //return normalize(hit_point-sphereCentre);
  return normal;//-((hit_point-params.centre)/params.curvatureRadius.x);
}

RT_PROGRAM void intersect(int)
{ 
	gaussBeam_t t;
    // if the radius is zero in both direction we use the plane surface intersection instead
	if ( (params.curvatureRadius.x==0) && (params.curvatureRadius.y==0) )
    {
		double3 normal=params.orientation;
		double3 root=params.centre;
		t.t_baseRay = intersectRayPlane(prd.baseRay.position, prd.baseRay.direction, root, normal);
		// check aperture explicitly
		double3 intersection=prd.baseRay.position+t_hit.t_baseRay*prd.baseRay.direction;
		if ( !checkAperture(params.centre, params.orientation, intersection, params.apertureType, params.apertureRadius) ) 
		{
			t.t_baseRay=0;
		}
	}
	else
	{
		t.t_baseRay = intersectRaySphere(prd.baseRay.position, prd.baseRay.direction, params);
	}
	if ( t.t_baseRay!=0 )
	{
	  // check wether intersection is within valid interval of t
	  if( rtPotentialIntersection( (float)t.t_baseRay ) ) 
	  {
	    // intersect all the rays
		// if the radius is zero in both direction we use the plane surface intersection instead
		if ( (params.curvatureRadius.x==0) && (params.curvatureRadius.y==0) )
	    {
			double3 normal=params.orientation;
			double3 root=params.centre;
			t.t_waistRayX=intersectRayPlane(prd.waistRayX.position, prd.waistRayX.direction, root, normal);
			t.t_waistRayY=intersectRayPlane(prd.waistRayY.position, prd.waistRayY.direction, root, normal);
			t.t_divRayX=intersectRayPlane(prd.divRayX.position, prd.divRayX.direction, root, normal);
			t.t_divRayY=intersectRayPlane(prd.divRayY.position, prd.divRayY.direction, root, normal);
		}
		else
		{	
			// set aperture to infinity for these rays
			params.apertureType=AT_INFTY;
			t.t_waistRayX=intersectRaySphere(prd.waistRayX.position, prd.waistRayX.direction, params);
			t.t_waistRayY=intersectRaySphere(prd.waistRayY.position, prd.waistRayY.direction, params);
			t.t_divRayX=intersectRaySphere(prd.divRayX.position, prd.divRayX.direction, params);
			t.t_divRayY=intersectRaySphere(prd.divRayY.position, prd.divRayY.direction, params);
		}
		/**************************************************************************************************************/
		/* what should we do if one of the rays doesn't hit the geometry, the centre ray hit ?                        */
		/* So far we call the hit functions anyway and terminate the ray with an error in the closest hit function    */
		/**************************************************************************************************************/
	    // calc normal in intersections
	    geometric_normal=calcHitParams(t);

	    // pass hit paramter to hit-program
	    t_hit=t;
		// pass geometryID to hit-program
		geometryID=params.geometryID;
	    // call any hit function
	    rtReportIntersection( 0 );
	  }
	}
}

RT_PROGRAM void bounds (int, float result[6])
{
  optix::Aabb* aabb = (optix::Aabb*)result;
  aabb->set(make_float3(0,0,0), make_float3(0,0,0));
}
