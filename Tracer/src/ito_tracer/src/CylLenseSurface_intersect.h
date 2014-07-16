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

/**\file CylLenseSurface_intersect.h
* \brief header file containing definitions of functions and datastructures that can be used on the GPU as well
* 
*           
* \author Mauch
*/

#ifndef CYLLENSEINTERSECT_H
  #define CYLLENSEINTERSECT_H

#include "Geometry_intersect.h"
#include "rayTracingMath.h"

/* declare class */
/**
  *\class   CylLenseSurface_ReducedParams 
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
class CylLenseSurface_ReducedParams : public Geometry_ReducedParams
{
  public:
 	  double3 root;
	  double3 orientation;
	  double3 cylOrientation;
	  double radius;
//	  double rotNormal; // rotation of geometry around its normal
	  double2 aptHalfWidth;
	  ApertureType aptType;

	  //double thickness;   
	  //int geometryID;
};

/**
 * \detail intersectRayCylLenseSurface 
 *
 * calculates the intersection of a ray with an cylindrical lense surface
 *
 * \param[in] double3 rayPosition, double3 rayDirection, CylLenseSurface_ReducedParams params
 * 
 * \return double t
 * \sa 
 * \remarks this function is defined inline so it can be used on GPU and CPU
 * \author Mauch
 */
inline RT_HOSTDEVICE double intersectRayCylLenseSurface(double3 rayPosition, double3 rayDirection, CylLenseSurface_ReducedParams params)
{
	/* see "http://www.matheboard.de/archive/1188/thread.html (20.07.10)" for a derivation of the algorithm */
	/* "P.J. Schneider, Geometric Tools for Computer Graphics, pp512" might be a more serious source... */
	double3 e=rayDirection-dot(rayDirection,params.cylOrientation)/dot(params.cylOrientation,params.cylOrientation)*params.cylOrientation;
	// in contrast to the cylinder piper (the root is located in the centre of the front cap of the pipe), 
	// the root of the CylLenseSurface is located on the sidewall of the pipe. Therefore we need to calc witha modified root here...
	double3 pipeRoot=params.root+params.radius*params.orientation;
	double3 f=(rayPosition-pipeRoot)-dot((rayPosition-pipeRoot),params.cylOrientation)/dot(params.cylOrientation,params.cylOrientation)*params.cylOrientation;
  
	double A=dot(e,e);
	double B=2*dot(e,f); 
	double C=dot(f,f)-params.radius*params.radius;
	double root=(B*B-4*A*C);
	
	double3 intersect;		//intersection Point		

	// if root is positive we have two intersections if not, we have none
	if (root<=0)
	{
		return 0;
	}
	//**********************************************
	// decide which ( if any ) intersection is physical
	//**********************************************
	// calc parameter of intersection of ray with middle plane of sphere
	double3 normal=params.orientation;
	double3 rootVec=params.root;
	double tmid=intersectRayPlane(rayPosition, rayDirection, rootVec, normal);

	// calc parameter first intersection
	double t=2*C/(-B-sqrt(root));
	double3 intersection=rayPosition+t*rayDirection;

	// we use the zemax sign convention, i.e. a negative radius represents convex surfaces !!
	// if the product of the orientation vector, the vector connecting the root of the cylinder and the radius is negative we have the physical intersection. Otherwise we have to calculate the other intersection
	if (dot(params.orientation,(intersection-params.root))*params.radius>0)
	{
		// calc the parameter of the other intersection
  		t=2*C/(-B+sqrt(root));
	}

	//**********************************************
	// check wether this intersection is inside the aperture
	//**********************************************
	intersection=rayPosition+t*rayDirection;
	if (checkAperture(params.root,params.tilt,intersection,params.aptType, params.aptHalfWidth))
	{
		return t;
	}

	// if we haven't returned yet, there's no intersection which we indicate by returning zero
	return 0;
}

/**
 * \detail calcHitParamsCylLenseSurface 
 *
 * calculates the parameters of the surface at the intersection point that are needed within the hit-function
 *
 * \param[in] double3 position,CylLenseSurface_ReducedParams params
 * 
 * \return Mat_hitParams
 * \sa 
 * \remarks this function is defined inline so it can be used on GPU and CPU
 * \author Mauch
 */
inline RT_HOSTDEVICE Mat_hitParams calcHitParamsCylLenseSurface(double3 position, CylLenseSurface_ReducedParams params)
{
	// first calculate the intersection of the middle axis of the cylinder with the plane through position beeing normal to the middle axis
	double t=intersectRayPlane(params.root, params.orientation, position, params.orientation);
	double3 i=params.root+t*params.orientation;
	// the normal vector we looked for is the normalized version of the vector connecting position and i
	Mat_hitParams t_hitParams;
	t_hitParams.normal=normalize(i-position);
	return t_hitParams;
}

/**
 * \detail cylLenseSurfaceBounds 
 *
 * calculates the bounding box of a cylLense
 *
 * \param[in] int primIdx, float result[6], ApertureStop_ReducedParams params
 * 
 * \return double t
 * \sa 
 * \remarks this function is defined inline so it can be used on GPU and CPU
 * \author Mauch
 */
inline RT_HOSTDEVICE void cylLenseSurfaceBounds (int primIdx, float result[6], CylLenseSurface_ReducedParams params)
{
    
}

#endif
