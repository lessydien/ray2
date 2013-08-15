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

#endif
