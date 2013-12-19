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
#include "cadObject_Intersect.h"

rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(rayStruct, prd, rtPayload, ); // get per-ray-data structure
rtDeclareVariable(CadObject_ReducedParams, params, , ); // normal vector to surface. i.e. part of the definition of the plane surface geometry
//rtDeclareVariable(int, materialListLength, , ); 
// variables that are communicate to the hit program via the attribute mechanism
rtDeclareVariable(Mat_hitParams, hitParams, attribute hitParams, ); // normal to the geometry at the hit-point. at a plane surface this will simply be the normal of the definition of the plane surface
rtDeclareVariable(double, t_hit, attribute t_hit, ); 
rtDeclareVariable(int,               geometryID, attribute geometryID , );

rtBuffer<int3>			index_buffer;
rtBuffer<float3>		vertex_buffer;

///* calc normal to surface at intersection point */
//__forceinline__ __device__ Mat_hitParams calcHitParams(double t)
//{
//  Mat_hitParams t_hitParams;
//  t_hitParams.normal=params.normal;
//  return t_hitParams;
//}

/* calc intersection of ray with geometry */
RT_PROGRAM void intersect(int primIdx)
{
  // Intersect ray with triangle
  double t=intersectRayCadObject(prd.position,prd.direction,params, &(vertex_buffer[0]), &(index_buffer[0]), primIdx);
  //if( my_intersect_triangle( make_float3(prd.direction), make_float3(prd.position), p0, p1, p2, n, t, beta, gamma ) ) 
  //{
//	my_intersect_triangle( make_float3(prd.direction), make_float3(prd.position), p0, p1, p2, n, t, beta, gamma ) ;
    if(  rtPotentialIntersection( t ) ) 
	{
	  //hitParams.normal=make_double3(normalize(n));
		hitParams.normal=make_double3(1,0,0);
//      float3 n0 = vertex_buffer[ v_idx.x ].normal;
//      float3 n1 = vertex_buffer[ v_idx.y ].normal;
//      float3 n2 = vertex_buffer[ v_idx.z ].normal;
//      shading_normal   = normalize( n0*(1.0f-beta-gamma) + n1*beta + n2*gamma );
//      geometric_normal = normalize( n );
		// communicate t_hit to closest_hit function
		t_hit=t;
		// pass geometryID to hit-program
		geometryID=params.geometryID;


        rtReportIntersection(0);
		//prd.running=false;
		//prd.position=make_double3(0, 0, 1000);

    }

//  }
}

RT_PROGRAM void bounds (int primIdx, float result[6])
{
  optix::Aabb* aabb = (optix::Aabb*)result;
  const int3 v_idx = index_buffer[primIdx];

  const float3 v0 = vertex_buffer[ v_idx.x ];
  const float3 v1 = vertex_buffer[ v_idx.y ];
  const float3 v2 = vertex_buffer[ v_idx.z ];
  const float  area = length(cross(v1-v0, v2-v0));

  if(area > 0.0f && !isinf(area)) {
    aabb->m_min = fminf( fminf( v0, v1), v2 );
    aabb->m_max = fmaxf( fmaxf( v0, v1), v2 );
  } else {
    aabb->invalidate();
  }

  //optix::Aabb* aabb = (optix::Aabb*)result;
  //aabb->set(make_float3(0,0,0), make_float3(0,0,0));
}
