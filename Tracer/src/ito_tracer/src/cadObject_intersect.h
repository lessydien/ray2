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

/**\file CadObject_intersect.h
* \brief header file containing definitions of functions and datastructures that can be used on the GPU as well
* 
*           
* \author Mauch
*/

#ifndef CADOBJECTINTERSECT_H
  #define CADOBJECTINTERSECT_H

/* include header of basis class */
#include "Geometry_intersect.h"
#include "rayTracingMath.h"
#include <optixu/optixu_aabb.h>


/* declare class */
/**
  *\class   CadObject_ReducedParams 
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
class CadObject_ReducedParams : public Geometry_ReducedParams
{
  public:
   double3 root;
   double3 normal;
//   double2 apertureRadius;
//   double rotNormal; // rotation of geometry around its normal
//   ApertureType apertureType;
   //int geometryID;
};

/**
 * \detail intersectRayCadObject 
 *
 * calculates the intersection of a ray with a plane surface
 *
 * \param[in] double3 rayPosition, double3 rayDirection, CadObject_ReducedParams params
 * 
 * \return double t
 * \sa 
 * \remarks this function is defined inline so it can be used on GPU and CPU
 * \author Mauch
 */
inline RT_HOSTDEVICE double intersectRayCadObject(double3 rayPosition, double3 rayDirection, CadObject_ReducedParams params, float3 *vertex_buffer, int3 *index_buffer, int primIdx)
{
	// transform ray into local coordinate system
	double3 tmpPos=rayPosition-params.root;
	rotateRayInv(&tmpPos,params.tilt);
	double3 tmpDir=rayDirection;
	rotateRayInv(&tmpDir,params.tilt);

	int3 v_idx = index_buffer[primIdx];

	float3 p0 = vertex_buffer[ v_idx.x ];
	float3 p1 = vertex_buffer[ v_idx.y ];
	float3 p2 = vertex_buffer[ v_idx.z ];

    float3 n;
    float  t, beta, gamma;
	
	if( !my_intersect_triangle( make_float3(tmpDir), make_float3(tmpPos), p0, p1, p2, n, t, beta, gamma ) ) 
		t=0;

	//double3 test=rayPosition+t*rayDirection;
	//CadObject_ReducedParams testPar=params;
	//// check aperture
	//if ( !checkAperture(params.root, params.tilt, rayPosition+t*rayDirection, params.apertureType, params.apertureRadius) )
	//{
	//	return 0;
	//}
	return t;
}

/**
 * \detail calcHitParamsCADObject 
 *
 * calculates the parameters of the surface at the intersection point that are needed within the hit-function
 *
 * \param[in] double3 position,SphericalSurface_ReducedParams params
 * 
 * \return Mat_hitParams
 * \sa 
 * \remarks this function is defined inline so it can be used on GPU and CPU
 * \author Mauch
 */
inline RT_HOSTDEVICE Mat_hitParams calcHitParamsCADObject(CadObject_ReducedParams params, float3* vertex_buffer, int3* index_buffer, int primIdx)
{
	double3 n;

    const int3 v_idx = index_buffer[primIdx];

    const float3 v0 = vertex_buffer[ v_idx.y]-vertex_buffer[v_idx.x ];
    const float3 v1 = vertex_buffer[ v_idx.z]-vertex_buffer[v_idx.x ];
    float3 normal=cross(v0,v1);
 
	Mat_hitParams t_hitParams;
	t_hitParams.normal=make_double3(normal.x, normal.y, normal.z);
    t_hitParams.normal=normalize(t_hitParams.normal);

    rotateRay(&t_hitParams.normal, params.tilt);

	return t_hitParams;
}

/**
 * \detail cadObjectBounds 
 *
 * calculates the bounding boxes of the individual triangles of an CAD object
 *
 * \param[in] int primIdx, float result[6], ApertureStop_ReducedParams params
 * 
 * \return double t
 * \sa 
 * \remarks this function is defined inline so it can be used on GPU and CPU
 * \author Mauch
 */
inline RT_HOSTDEVICE void cadObjectBounds (int primIdx, float result[6], CadObject_ReducedParams params, float3* vertex_buffer, int3* index_buffer)
{
  optix::Aabb* aabb = (optix::Aabb*)result;
  const int3 v_idx = index_buffer[primIdx];

  //const float3 v0 = vertex_buffer[ v_idx.x ];
  //const float3 v1 = vertex_buffer[ v_idx.y ];
  //const float3 v2 = vertex_buffer[ v_idx.z ];

  const float3 v0 = rotateRay(vertex_buffer[ v_idx.x ],params.tilt)+make_float3(params.root);
  const float3 v1 = rotateRay(vertex_buffer[ v_idx.y ],params.tilt)+make_float3(params.root);
  const float3 v2 = rotateRay(vertex_buffer[ v_idx.z ],params.tilt)+make_float3(params.root);

  const float  area = length(cross(v1-v0, v2-v0));

  if(area > 0.0f && area < 99999999999999999999.0f) {
      float3 maxBox=make_float3(max(max(v0.x, v1.x), v2.x), max(max(v0.y, v1.y), v2.y), max(max(v0.z, v1.z), v2.z));
      float3 minBox=make_float3(min(min(v0.x, v1.x), v2.x), min(min(v0.y, v1.y), v2.y), min(min(v0.z, v1.z), v2.z));
      aabb->set(minBox, maxBox);
//    aabb->m_min = fminf( fminf( v0, v1), v2 );
//    aabb->m_max = fmaxf( fmaxf( v0, v1), v2 );
  } else {
    aabb->invalidate();
  }
}

#endif
