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

/**\file VolumeScattererBox_intersect.h
* \brief header file containing definitions of functions and datastructures that can be used on the GPU as well
* 
*           
* \author Mauch
*/

#ifndef VOLUMESCATTERERBOXINTERSECT_H
  #define VOLUMESCATTERERBOXINTERSECT_H

/* include header of basis class */
#include "Geometry_intersect.h"
#include "rayTracingMath.h"

/* declare class */
/**
  *\class   VolumeScattererBox_ReducedParams 
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
class VolumeScattererBox_ReducedParams : public Geometry_ReducedParams
{
  public:
   double3 root;
   double3 normal;
   double2 apertureRadius;
//   double rotNormal; // rotation of geometry around its normal
   double thickness;
   //int geometryID;
};

/**
 * \detail intersectRayVolumeScattererBox 
 *
 * calculates the intersection of a ray with a plane surface
 *
 * \param[in] double3 rayPosition, double3 rayDirection, VolumeScattererBox_ReducedParams params
 * 
 * \return double t
 * \sa 
 * \remarks this function is defined inline so it can be used on GPU and CPU
 * \author Mauch
 */
inline RT_HOSTDEVICE double intersectRayVolumeScattererBox(double3 rayPosition, double3 rayDirection, VolumeScattererBox_ReducedParams params)
{
	// position on micro lens array surface in local coordinate system 
	double3 tmpPos=rayPosition-params.root;
	rotateRayInv(&tmpPos,params.tilt);
	double3 tmpDir=rayDirection;
	rotateRayInv(&tmpDir,params.tilt);
	// transform ray into local coordinate system
	// intersect front face of box
	double t=DOUBLE_MAX;
	double t1=intersectRayPlane(tmpPos, tmpDir, make_double3(0,0,0), make_double3(0,0,1));
	if ( (abs(tmpPos.x+t1*tmpDir.x)<params.apertureRadius.x) && (abs(tmpPos.y+t1*tmpDir.y)<params.apertureRadius.y) && (t1>0))
		t=min(t,t1);
	// intersect back face of box
	t1=intersectRayPlane(tmpPos, tmpDir, make_double3(0,0,params.thickness), make_double3(0,0,1));
	if ( ((abs(tmpPos.x+t1*tmpDir.x)<params.apertureRadius.x) && (abs(tmpPos.y+t1*tmpDir.y)<params.apertureRadius.y)) && (t1>0))
		t=min(t,t1);
	// intersect back right side of box
	t1=intersectRayPlane(tmpPos, tmpDir, make_double3(0,params.apertureRadius.y,0.5*params.thickness), make_double3(0,1,0));
	if ( ((abs(tmpPos.x+t1*tmpDir.x)<params.apertureRadius.x) && ( (tmpPos.z+t1*tmpDir.z)>0 && (tmpPos.z+t1*tmpDir.z<params.thickness) )) && (t1>0))
		t=min(t,t1);
	// intersect back left side of box
	t1=intersectRayPlane(tmpPos, tmpDir, make_double3(0,-params.apertureRadius.y,0.5*params.thickness), make_double3(0,-1,0));
	if ( ((abs(tmpPos.x+t1*tmpDir.x)<params.apertureRadius.x) && ( (tmpPos.z+t1*tmpDir.z)>0 && (tmpPos.z+t1*tmpDir.z<params.thickness) )) && (t1>0))
		t=min(t,t1);
	// intersect back front side of box
	t1=intersectRayPlane(tmpPos, tmpDir, make_double3(params.apertureRadius.x,0,0.5*params.thickness), make_double3(1,0,0));
	if ( ((abs(tmpPos.y+t1*tmpDir.y)<params.apertureRadius.y) && ( (tmpPos.z+t1*tmpDir.z)>0 && (tmpPos.z+t1*tmpDir.z<params.thickness) )) && (t1>0))
		t=min(t,t1);
	// intersect back front side of box
	t1=intersectRayPlane(tmpPos, tmpDir, make_double3(-params.apertureRadius.x,0,0.5*params.thickness), make_double3(-1,0,0));
	if ( ((abs(tmpPos.y+t1*tmpDir.y)<params.apertureRadius.y) && ( (tmpPos.z+t1*tmpDir.z)>0 && (tmpPos.z+t1*tmpDir.z<params.thickness) )) && (t1>0))
		t=min(t,t1);

	if (t==DOUBLE_MAX)
		t = 0;

	return t;
}

/**
 * \detail calcHitParamsVolumeScattererBox 
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
inline RT_HOSTDEVICE Mat_hitParams calcHitParamsVolumeScattererBox(double3 position,VolumeScattererBox_ReducedParams params)
{
	Mat_hitParams l_hitParams;

	double3 tmpPos=position-params.root;
	rotateRayInv(&tmpPos,params.tilt);
	double3 l_normal=make_double3(0,0,1);
	if (abs(tmpPos.x-params.apertureRadius.x) <EPSILON)
		l_normal=make_double3(1,0,0);
	if (abs(tmpPos.x+params.apertureRadius.x) <EPSILON)
		l_normal=make_double3(-1,0,0);
	if (abs(tmpPos.y-params.apertureRadius.y) <EPSILON)
		l_normal=make_double3(0,1,0);
	if (abs(tmpPos.y+params.apertureRadius.y) <EPSILON)
		l_normal=make_double3(0,-1,0);
	if (abs(tmpPos.z) <EPSILON)
		l_normal=make_double3(0,0,1);
	if (abs(tmpPos.z+params.thickness) <EPSILON)
		l_normal=make_double3(0,0,-1);
	
	rotateRay(&l_normal,params.tilt);
	l_hitParams.normal=l_normal;
	return l_hitParams;
}

#endif
