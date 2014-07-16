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

#ifndef CONEPIPEINTERSECT_H
  #define CONEPIPEINTERSECT_H
  
/* include header of basis class */
#include "Geometry_intersect.h"
#include "rayTracingMath.h"
#include <optixu/optixu_aabb.h>

/* declare class */
/**
  *\class   ConePipe_ReducedParams 
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
class ConePipe_ReducedParams : public Geometry_ReducedParams
{
  public:
 	  double3 root; // starting point of the cone segment
	  double3 coneEnd; // end point of the cone. where the sidewalls would meet in one point
	  double3 orientation; // orientation of the symmetrie axis of the cone
	  double2 cosTheta; // half opening angle of the cone in x and y
	  double thickness; // length of the cone segment
	  double radMax; // maximum aperture radius of cone
	  //int geometryID;
};

/**
 * \detail intersectRayConePipe 
 *
 * calculates the intersection of a ray with an cone pipe
 *
 * \param[in] double3 rayPosition, double3 rayDirection, ConePipe_ReducedParams params
 * 
 * \return double t
 * \sa 
 * \remarks this function is defined inline so it can be used on GPU and CPU
 * \author Mauch
 */
inline RT_HOSTDEVICE double intersectRayConePipe(double3 rayPosition, double3 rayDirection, ConePipe_ReducedParams params)
{
	/* see "P.J. Schneider, Geometric Tools for Computer Graphics, pp512" for a derivation of the algorithm */
	//in this source: rayDirection=d, orientation=a, costheta=gamma, rayPosition=P, root=V, A=c2, B= c1, C=c0
	
	double cosThetaSqrd=params.cosTheta.x*params.cosTheta.x;

	double A=rayDirection.x*(rayDirection.x*(pow(params.orientation.x,2)-cosThetaSqrd)+rayDirection.y*params.orientation.x*params.orientation.y+rayDirection.z*params.orientation.x*params.orientation.z)
		+rayDirection.y*(rayDirection.x*params.orientation.y*params.orientation.x+rayDirection.y*(pow(params.orientation.y,2)-cosThetaSqrd)+rayDirection.z*params.orientation.y*params.orientation.z)
		+rayDirection.z*(rayDirection.x*params.orientation.z*params.orientation.x+rayDirection.y*params.orientation.z*params.orientation.y+rayDirection.z*(pow(params.orientation.z,2)-cosThetaSqrd));

	double3 Delta=rayPosition-params.coneEnd;//v
	
	double B=rayDirection.x*(Delta.x*(pow(params.orientation.x,2)-cosThetaSqrd)+Delta.y*params.orientation.x*params.orientation.y+Delta.z*params.orientation.x*params.orientation.z)
		+rayDirection.y*(Delta.x*params.orientation.y*params.orientation.x+Delta.y*(pow(params.orientation.y,2)-cosThetaSqrd)+Delta.z*params.orientation.y*params.orientation.z)
		+rayDirection.z*(Delta.x*params.orientation.z*params.orientation.x+Delta.y*params.orientation.z*params.orientation.y+Delta.z*(pow(params.orientation.z,2)-cosThetaSqrd));
	B=2*B;
	
	double C=Delta.x*(Delta.x*(pow(params.orientation.x,2)-cosThetaSqrd)+Delta.y*params.orientation.x*params.orientation.y+Delta.z*params.orientation.x*params.orientation.z)
		+Delta.y*(Delta.x*params.orientation.y*params.orientation.x+Delta.y*(pow(params.orientation.y,2)-cosThetaSqrd)+Delta.z*params.orientation.y*params.orientation.z)
		+Delta.z*(Delta.x*params.orientation.z*params.orientation.x+Delta.y*params.orientation.z*params.orientation.y+Delta.z*(pow(params.orientation.z,2)-cosThetaSqrd));
	


	// if the ray is contained by the cone we define to have no intersection
	if( (A==0) && (B==0) )
	{
		return 0;
	}
	
	double3 intersect;		//intersection Point
	int nrOfIntersections=0;//number of positiv intersections of ray and cone
	double root=(B*B-4*A*C);

	// if root is positive we have two intersections if not, we have none
	if (root>=0)
	{
		// calc the nearest intersection

		//double t=2*C/(-B+sqrt(root));
		double t1=(-B+sqrt(root))/(2*A);
		double t2=(-B-sqrt(root))/(2*A);
		
		//counting intersections with positive t
		int nrOfIntersections=0;
		if (t1>0) nrOfIntersections++;
		if (t2>0) nrOfIntersections++;

		//we check the intersectionpoint of t1 if ...
		//t1 is positiv and t1 is smaller than t2
		//or
		//t1 is positiv and t2 is negativ or zero
		if (((t1>0) && (t1<=t2))  ||  (t1>0) && (t2<=0))
		{
			intersect=rayPosition+t1*rayDirection;
			if (checkApertureCylinder(params.root,params.orientation,intersect,params.thickness/2))
			{
				//if we return t1. this is the only possible solution
				return t1;
			}
		}
		//if we didn't return yet that means, 
		if (t2>0)
		{
			intersect=rayPosition+t2*rayDirection;
			if (checkApertureCylinder(params.root,params.orientation,intersect,params.thickness/2))
			{
				return t2;
			}

			//if we do not have a valid solution yet, we have to recheck if t1 is the solution. even if it is farer away than t2
			if (nrOfIntersections==2)
			{
				intersect=rayPosition+t1*rayDirection;
				if (checkApertureCylinder(params.root,params.orientation,intersect,params.thickness/2))
				{
					return t1;
				}
			}
		}
	}

	// if we haven't returned yet, there's no intersection which we indicate by returning zero
	return 0;
}

/**
 * \detail calcHitParamsConePipe 
 *
 * calculates the parameters of the surface at the intersection point that are needed within the hit-function
 *
 * \param[in] double3 position,ConePipe_ReducedParams params
 * 
 * \return Mat_hitParams
 * \sa 
 * \remarks this function is defined inline so it can be used on GPU and CPU
 * \author Mauch
 */
inline RT_HOSTDEVICE Mat_hitParams calcHitParamsConePipe(double3 position, ConePipe_ReducedParams params)
{
	// first calculate the intersection of the middle axis of the cone with the plane through position beeing normal to the tangent to the cone in position.
	// this tangent is equal to the vector connecting position with the vertex of the cone
	double t=intersectRayPlane(params.root, params.orientation, position, (position-params.coneEnd) );
	double3 i=params.root+t*params.orientation;
	Mat_hitParams t_hitParams;
	t_hitParams.normal=normalize(i-position);
	return t_hitParams;
}

/**
 * \detail conePipeBounds 
 *
 * calculates the bounding box of a conePipe
 *
 * \param[in] int primIdx, float result[6], ConePipe_ReducedParams params
 * 
 * \return double t
 * \sa 
 * \remarks this function is defined inline so it can be used on GPU and CPU
 * \author Mauch
 */
inline RT_HOSTDEVICE void conePipeBounds (int primIdx, float result[6], ConePipe_ReducedParams params)
{
  optix::Aabb* aabb = (optix::Aabb*)result;
  double3 l_ex=make_double3(1,0,0);
  rotateRay(&l_ex,params.tilt);
  double3 l_ey=make_double3(0,1,0);
  rotateRay(&l_ey,params.tilt);
  double3 l_n=make_double3(0,0,1);
  rotateRay(&l_n,params.tilt);

  float3 maxBox=make_float3(params.root+params.thickness*l_n+params.radMax*l_ex+params.radMax*l_ey);
  float3 minBox=make_float3(params.root-params.radMax*l_ex-params.radMax*l_ey);
  aabb->set(minBox, maxBox);    
}

#endif
