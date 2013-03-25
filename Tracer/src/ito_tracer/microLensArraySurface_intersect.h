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

/**\file MicroLensArraySurface_intersect.h
* \brief header file containing definitions of functions and datastructures that can be used on the GPU as well
* 
*           
* \author Mauch
*/

#ifndef MICROLENSARRAYSURFACEINTERSECT_H
  #define MICROLENSARRAYSURFACEINTERSECT_H
  
/* include geometry lib */
#include "Geometry_intersect.h"
#include "rayTracingMath.h"
//#include "SphericalSurface_intersect.h"

typedef enum 
{
	MICRORECTANGULAR,
	MICROELLIPTICAL,
	MICROUNKNOWN
} MicroLensAptType;

/* declare class */
/**
  *\class   MicroLensArraySurface_ReducedParams
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
class MicroLensArraySurface_ReducedParams : public Geometry_ReducedParams
{
	public:
 	  double3 root;
	  double3 normal;
	  double microLensRadius;
	  double microLensPitch;
	  double microLensAptRad;
	  MicroLensAptType microLensAptType;
	  ApertureType apertureType;
	  double2 apertureRadius;
};

/**
 * \detail intersectRaySphere 
 *
 * calculates the intersection of a ray with a the surface of a micro lens array
 *
 * \param[in] double3 rayPosition, double3 rayDirection, MicroLensArraySurface_ReducedParams params
 * 
 * \return double t
 * \sa 
 * \remarks this function is defined inline so it can be used on GPU and CPU
 * \author Mauch
 */
inline RT_HOSTDEVICE double intersectRayMicroLensArraySurface(double3 rayPosition, double3 rayDirection, MicroLensArraySurface_ReducedParams params)
{
	MicroLensArraySurface_ReducedParams testPar=params;

	double t = intersectRayPlane(rayPosition, rayDirection, params.root, params.normal);

	// check aperture
	if ( !checkAperture(params.root, params.tilt, rayPosition+t*rayDirection, params.apertureType, params.apertureRadius) )
	{
		return 0;
	}
	else
	{
		// position on micro lens aray surface in local coordinate system 
		double3 tmpPos=rayPosition+t*rayDirection-params.root;
		rotateRayInv(&tmpPos,params.tilt);
		double3 tmpDir=rayDirection;
		rotateRayInv(&tmpDir,params.tilt);

		// see in which subaperture we are
		double fac=floor(tmpPos.x/params.microLensPitch+0.5);
		double3 microSphereCentre;
		microSphereCentre.x=fac*params.microLensPitch;
		fac=floor(tmpPos.y/params.microLensPitch+0.5);
		microSphereCentre.y=fac*params.microLensPitch;

		double lensHeightMax;
		double effAptRadius=min(params.microLensPitch/2,params.microLensAptRad);
		double rMax=sqrt(2*effAptRadius*effAptRadius);
		if (rMax>abs(params.microLensRadius))
			lensHeightMax=sqrt(params.microLensRadius*params.microLensRadius-effAptRadius*effAptRadius);
		else
			lensHeightMax=sqrt(params.microLensRadius*params.microLensRadius-rMax*rMax);
		
		if (params.microLensRadius<0)
			microSphereCentre.z=-lensHeightMax;
		else
			microSphereCentre.z=lensHeightMax;

		// intersect the microsphere
		// calc the two intersection points
		double A=dot(tmpDir, tmpDir);
		double3 origin=tmpPos-microSphereCentre;
		double B=2*dot(tmpDir,origin);
		double C=dot(origin,origin)-(params.microLensRadius*params.microLensRadius);
		if (abs(C)<EPSILON) // if the ray hit right at the intersection of the microsphere and the plane, return the position on the plane...
			return t;
		double root=(B*B-4*A*C);
		// if root is positive we have two intersections if not, we have none and we return the intersection with the plane
		if (root<=0)
		{
			return t;
		}
		//**********************************************
		// decide which ( if any ) intersection is physical
		//**********************************************
		// calc parameter first intersection
		double denominator=(-B-sqrt(root));
	//  double t=0;
	//  if (denominator!=0)
		double tmpT=2*C/(-B-sqrt(root));
		double3 intersection=tmpPos+tmpT*tmpDir;

		// we use the zemax sign convention, i.e. a negative radius represents convex surfaces !!
		if ( params.microLensRadius*dot(make_double3(0,0,1),(intersection-tmpPos))>0 )
		{
			// calc the parameter of the other intersection
  			tmpT=2*C/(-B+sqrt(root));
		}
		//**********************************************
		// check wether this intersection is inside the aperture
		//**********************************************
		intersection=tmpPos+tmpT*tmpDir;

		// distance from local centre
		double3 localDist=intersection-microSphereCentre;
		double r;
		if (params.microLensAptType==MICRORECTANGULAR)
			r=max(abs(localDist.x), abs(localDist.y));
		else
			r=sqrt(localDist.x*localDist.x+localDist.y*localDist.y);

		// if we are outside the micro aperture but inside the pitch, we return the intersection with the plane
		if ( (r>params.microLensAptRad) && (abs(localDist.x)<params.microLensPitch) && (abs(localDist.y)<params.microLensPitch) )
			return t;
		else
		{
			// if we are outised the pitch, we need to calculate the intersection with the next pitch
			if ( (abs(localDist.x)>params.microLensPitch/2) || (abs(localDist.y)>params.microLensPitch/2) )
			{
				// see in which subaperture we are now
				fac=floor(intersection.x/params.microLensPitch+0.5);
				microSphereCentre;
				microSphereCentre.x=fac*params.microLensPitch;
				fac=floor(intersection.y/params.microLensPitch+0.5);
				microSphereCentre.y=fac*params.microLensPitch;

				// intersect the microsphere
				// calc the two intersection points
				A=dot(tmpDir, tmpDir);
				origin=intersection-microSphereCentre;
				B=2*dot(tmpDir,origin);
				C=dot(origin,origin)-(params.microLensRadius*params.microLensRadius);
				if (abs(C)<EPSILON) // if the ray hit right at the intersection of the microsphere and the plane, return the position on the plane...
					return t;
				root=(B*B-4*A*C);
				// if root is positive we have two intersections if not, we have none and we return the intersection with the plane
				if (root<=0)
				{
					return t;
				}
				//**********************************************
				// decide which ( if any ) intersection is physical
				//**********************************************
				// calc parameter first intersection
				denominator=(-B-sqrt(root));
			//  double t=0;
			//  if (denominator!=0)
				double tmpT2=2*C/(-B-sqrt(root));
				double3 newIntersection=intersection+tmpT2*tmpDir;

				// we use the zemax sign convention, i.e. a negative radius represents convex surfaces !!
				if ( params.microLensRadius*dot(make_double3(0,0,1),(newIntersection-tmpPos))>0 )
				{
					// calc the parameter of the other intersection
  					tmpT2=2*C/(-B+sqrt(root));
				}
				//**********************************************
				// check wether this intersection is inside the aperture
				//**********************************************
				newIntersection=intersection+tmpT2*tmpDir;

				// distance from local centre
				localDist=newIntersection-microSphereCentre;
				if (params.microLensAptType==MICRORECTANGULAR)
					r=max(abs(localDist.x), abs(localDist.y));
				else
					r=sqrt(localDist.x*localDist.x+localDist.y*localDist.y);
				// if we are outside the micro aperture we return the itersection with the plane
				if ( (r>params.microLensAptRad) && (abs(localDist.x)<params.microLensPitch) && (abs(localDist.y)<params.microLensPitch) )
					return t;
				// if we are inside the aperture, we return the intersection with the new microsphere
				t=t+tmpT+tmpT2;
			}
			// if we are inside the micro aperture and inside the pitch we return the intersection with the original micro sphere
			else
				t=t+tmpT;//dot(tmpDir,(intersection-tmpPos));
		}
				
	}

	return t;
}

/**
 * \detail calcHitParamsSphere 
 *
 * calculates the parameters of the surface at the intersection point that are needed within the hit-function
 *
 * \param[in] double3 position,MicroLensArraySurface_ReducedParams params
 * 
 * \return Mat_hitParams
 * \sa 
 * \remarks this function is defined inline so it can be used on GPU and CPU
 * \author Mauch
 */
inline RT_HOSTDEVICE Mat_hitParams calcHitParamsMicroLensArraySurface(double3 position,MicroLensArraySurface_ReducedParams params)
{
	double3 n;

	// position on micro lens aray surface in local coordinate system 
	double3 tmpPos=position-params.root;
	rotateRayInv(&tmpPos,params.tilt);

	// see in which subaperture we are
	double fac=floor(tmpPos.x/params.microLensPitch+0.5);
	double3 microSphereCentre;
	microSphereCentre.x=fac*params.microLensPitch;
	fac=floor(tmpPos.y/params.microLensPitch+0.5);
	microSphereCentre.y=fac*params.microLensPitch;

	double lensHeightMax;
	double effAptRadius=min(params.microLensPitch/2,params.microLensAptRad);
	double rMax=sqrt(2*effAptRadius*effAptRadius);
	if (rMax>abs(params.microLensRadius))
		lensHeightMax=sqrt(params.microLensRadius*params.microLensRadius-effAptRadius*effAptRadius);
	else
		lensHeightMax=sqrt(params.microLensRadius*params.microLensRadius-rMax*rMax);
		
	if (params.microLensRadius<0)
		microSphereCentre.z=-lensHeightMax;
	else
		microSphereCentre.z=lensHeightMax;

	double3 localDist=tmpPos-microSphereCentre;
	double r;
	if (params.microLensAptType==MICRORECTANGULAR)
		r=max(abs(localDist.x), abs(localDist.y));
	else
		r=sqrt(localDist.x*localDist.x+localDist.y*localDist.y);
	
	if (r>params.microLensAptRad)
		n=make_double3(0,0,1);
	else
		n=normalize(position-microSphereCentre);

	rotateRay(&n, params.tilt);

	Mat_hitParams t_hitParams;
	t_hitParams.normal=n;
	return t_hitParams;
}

#endif
