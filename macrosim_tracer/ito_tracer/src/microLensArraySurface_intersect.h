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
#include <optixu/optixu_aabb.h>
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
		// position on micro lens array surface in local coordinate system 
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

		double3 localDist; // local distance of intersection to microSphereCentre
		double r;

		double lensHeightMax; // distance of sphereCentre to plane of microLensArraySurface
		double effAptRadius=min(params.microLensPitch/2,params.microLensAptRad);
		double rMax=effAptRadius;
		if (params.microLensAptType==MICRORECTANGULAR) // if aperture is rectangular rMax is the diagonal of the aperture...
			rMax=sqrt(2*effAptRadius*effAptRadius);
		if (rMax>abs(params.microLensRadius))
			lensHeightMax=sqrt(params.microLensRadius*params.microLensRadius-effAptRadius*effAptRadius);
		else
			lensHeightMax=sqrt(params.microLensRadius*params.microLensRadius-rMax*rMax);
		
		if (params.microLensRadius<0)
			microSphereCentre.z=-lensHeightMax;
		else
			microSphereCentre.z=lensHeightMax;

		// intersect the microsphere
		// see http://en.wikipedia.org/wiki/Line–sphere_intersection or David H. Eberly (2006), 3D game engine design: a practical approach to real-time computer graphics, 2nd edition, Morgan Kaufmann for reference
		//double A=dot(rayDirection, rayDirection); equals unity...
		double3 origin=tmpPos-microSphereCentre;
		double B=dot(tmpDir,origin);
		double C=dot(origin,origin)-(params.microLensRadius*params.microLensRadius);
		double root=(B*B-C);
		// if root is positive we have two intersections if not, we have none
		// if root is positive we have two intersections if not, we have none and we return the intersection with the plane
		if (root<=0)
		{
			localDist=tmpPos-microSphereCentre;
			if (params.microLensAptType==MICRORECTANGULAR)
				r=max(abs(localDist.x), abs(localDist.y));
			else
				r=sqrt(localDist.x*localDist.x+localDist.y*localDist.y);
			// if the intersection with the plane is inside the microaperture, it is not possible and we return zero
			if ( r<params.microLensAptRad )
				return 0;
			else
				return t;
		}
		double tmpT=-B+sqrt(root);

		double3 intersection=tmpPos+tmpT*tmpDir;

		// we use the zemax sign convention, i.e. a negative radius represents convex surfaces !!
		if ( (params.microLensRadius*dot(make_double3(0,0,1),(intersection-microSphereCentre))>0) || (abs(tmpT) < 1E-10))
		{
			tmpT=-B-sqrt(root);
			intersection=tmpPos+tmpT*tmpDir;
			if ( (params.microLensRadius*dot(make_double3(0,0,1),(intersection-microSphereCentre))>0) || (abs(tmpT) < 1E-10))
			{
				localDist=tmpPos-microSphereCentre;
				if (params.microLensAptType==MICRORECTANGULAR)
					r=max(abs(localDist.x), abs(localDist.y));
				else
					r=sqrt(localDist.x*localDist.x+localDist.y*localDist.y);
				// if the intersection with the plane is inside the microaperture, it is not possible and we return zero
				if ( r<params.microLensAptRad )
					return 0;
				else
					return t;
			}
		}

		// distance from local centre
		localDist=intersection-microSphereCentre;
		if (params.microLensAptType==MICRORECTANGULAR)
			r=max(abs(localDist.x), abs(localDist.y));
		else
			r=sqrt(localDist.x*localDist.x+localDist.y*localDist.y);

		// if we are outside the micro aperture but inside the pitch, we return the intersection with the plane
		if ( (r>params.microLensAptRad) && (abs(localDist.x)<params.microLensPitch) && (abs(localDist.y)<params.microLensPitch) )
		{
			localDist=tmpPos-microSphereCentre;
			if (params.microLensAptType==MICRORECTANGULAR)
				r=max(abs(localDist.x), abs(localDist.y));
			else
				r=sqrt(localDist.x*localDist.x+localDist.y*localDist.y);
			// if the intersection with the plane is inside the microaperture, it is not possible and we return zero
			if ( r<params.microLensAptRad )
				return 0;
			else
				return t;
		}
		else
		{
			// if we are outside the pitch, we need to calculate the intersection with the next pitch
			if ( (abs(localDist.x)>params.microLensPitch/2) || (abs(localDist.y)>params.microLensPitch/2) )
			{
				// see in which subaperture we are now
				fac=floor(intersection.x/params.microLensPitch+0.5);
				microSphereCentre;
				microSphereCentre.x=fac*params.microLensPitch;
				fac=floor(intersection.y/params.microLensPitch+0.5);
				microSphereCentre.y=fac*params.microLensPitch;

				// intersect the microsphere
				// see http://en.wikipedia.org/wiki/Line–sphere_intersection or David H. Eberly (2006), 3D game engine design: a practical approach to real-time computer graphics, 2nd edition, Morgan Kaufmann for reference
				//double A=dot(rayDirection, rayDirection); equals unity...
				origin=tmpPos-microSphereCentre;
				B=dot(tmpDir,origin);
				C=dot(origin,origin)-(params.microLensRadius*params.microLensRadius);
				root=(B*B-C);
				// if root is positive we have two intersections if not, we have none and we return the intersection with the plane
				if (root<=0)
				{
					localDist=tmpPos-microSphereCentre;
					if (params.microLensAptType==MICRORECTANGULAR)
						r=max(abs(localDist.x), abs(localDist.y));
					else
						r=sqrt(localDist.x*localDist.x+localDist.y*localDist.y);
					// if the intersection with the plane is inside the microaperture, it is not possible and we return zero
					if ( r<params.microLensAptRad )
						return 0;
					else
						return t;
				}
				double tmpT2=-B+sqrt(root);

				double3 newIntersection=tmpPos+tmpT2*tmpDir;

				// we use the zemax sign convention, i.e. a negative radius represents convex surfaces !!
				if ( (params.microLensRadius*dot(make_double3(0,0,1),(newIntersection-microSphereCentre))>0) || (abs(tmpT2) < 1E-10))
				{
					tmpT2=-B-sqrt(root);
					newIntersection=tmpPos+tmpT2*tmpDir;
					if ( (params.microLensRadius*dot(make_double3(0,0,1),(newIntersection-microSphereCentre))>0) || (abs(tmpT2) < 1E-10))
					{
						localDist=tmpPos-microSphereCentre;
						if (params.microLensAptType==MICRORECTANGULAR)
							r=max(abs(localDist.x), abs(localDist.y));
						else
							r=sqrt(localDist.x*localDist.x+localDist.y*localDist.y);
						// if the intersection with the plane is inside the microaperture, it is not possible and we return zero
						if ( r<params.microLensAptRad )
							return 0;
						else
							return t;
					}
				}
				//**********************************************
				// check wether this intersection is inside the aperture
				//**********************************************

				// distance from local centre
				localDist=newIntersection-microSphereCentre;
				if (params.microLensAptType==MICRORECTANGULAR)
					r=max(abs(localDist.x), abs(localDist.y));
				else
					r=sqrt(localDist.x*localDist.x+localDist.y*localDist.y);
				// if we are outside the micro aperture we return the itersection with the plane
				if ( (r>params.microLensAptRad) && (abs(localDist.x)<params.microLensPitch) && (abs(localDist.y)<params.microLensPitch) )
				{
					localDist=tmpPos-microSphereCentre;
					if (params.microLensAptType==MICRORECTANGULAR)
						r=max(abs(localDist.x), abs(localDist.y));
					else
						r=sqrt(localDist.x*localDist.x+localDist.y*localDist.y);
					// if the intersection with the plane is inside the microaperture, it is not possible and we return zero
					if ( r<params.microLensAptRad )
						return 0;
					else
						return t;
				}
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
 * \detail calcHitParamsMicroLensArraySurface 
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
inline RT_HOSTDEVICE Mat_hitParams calcHitParamsMicroLensArraySurface(double3 position, MicroLensArraySurface_ReducedParams params)
{
	double3 n;

	// position on micro lens array surface in local coordinate system 
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
	double rMax=effAptRadius;
	if (params.microLensAptType==MICRORECTANGULAR) // if aperture is rectangular rMax is the diagonal of the aperture...
		rMax=sqrt(2*effAptRadius*effAptRadius);
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
		n=normalize(tmpPos-microSphereCentre);

	rotateRay(&n, params.tilt);

	Mat_hitParams t_hitParams;
	t_hitParams.normal=n;
	return t_hitParams;
}

/**
 * \detail microLenseArrayBounds 
 *
 * calculates the bounding box of a microLenseArray
 *
 * \param[in] int primIdx, float result[6], MicroLensArraySurface_ReducedParams params
 * 
 * \return double t
 * \sa 
 * \remarks this function is defined inline so it can be used on GPU and CPU
 * \author Mauch
 */
inline RT_HOSTDEVICE void microLenseArrayBounds (int primIdx, float result[6], MicroLensArraySurface_ReducedParams params)
{
    optix::Aabb* aabb = (optix::Aabb*)result;
    double3 l_ex=make_double3(1,0,0);
    rotateRay(&l_ex,params.tilt);
    double3 l_ey=make_double3(0,1,0);
    rotateRay(&l_ey,params.tilt);
    double3 l_n=make_double3(0,0,1);
    rotateRay(&l_n,params.tilt);

    double effAptRadius=min(params.microLensPitch/2,params.microLensAptRad);
    double rMax=effAptRadius;
    double lensHeightMax;
    if (params.microLensAptType==MICRORECTANGULAR) // if aperture is rectangular rMax is the diagonal of the aperture...
	    rMax=sqrt(2*effAptRadius*effAptRadius);
    if (rMax>abs(params.microLensRadius))
	    lensHeightMax=sqrt(params.microLensRadius*params.microLensRadius-effAptRadius*effAptRadius);
    else
	    lensHeightMax=sqrt(params.microLensRadius*params.microLensRadius-rMax*rMax);

    lensHeightMax=params.microLensRadius-lensHeightMax;

    float3 maxBox=make_float3(params.root+l_n*lensHeightMax+params.apertureRadius.x*l_ex+params.apertureRadius.y*l_ey);
    float3 minBox=make_float3(params.root-params.apertureRadius.x*l_ex-params.apertureRadius.y*l_ey);
    aabb->set(minBox, maxBox);    
}

#endif
