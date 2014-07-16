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

/**\file MaterialReflecting.h
* \brief header file containing definitions of functions and datastructures that can be used on the GPU as well
* 
*           
* \author Mauch
*/

#ifndef MATERIALRENDERFRINGEPROJ_HIT_H
#define MATERIALRENDERFRINGEPROJ_HIT_H

#include "..\rayTracingMath.h"
#include "..\Material_hit.h"

typedef enum
{
    FT_GRAYCODE,
    FT_SINUS
} FringeType;

typedef enum
{
    O_X,
    O_Y,
    O_Z
} Orientation;

/* declare class */
/**
  *\class   MatFringeProj_GeomRender_params 
  *\ingroup Material
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
class MatFringeProj_GeomRender_params
{
public:
	double power; // power of light source
//    double3 pupilRoot;
//    double3 pupilTilt;
//    double2 pupilAptRad;

    double fringePeriod;
    double fringePhase;
    int nrBits;
    int codeNr;

    FringeType fringeType;
    Orientation fringeOrientation;

    double3 geomRoot;
    double3 geomTilt;
};

/**
 * \detail hitRenderFringeProj
 *
 * modifies the raydata according to the parameters of the material
 *
 * \param[in] rayStruct &ray, Mat_hitParams hitParams, double t_hit, int geometryID
 * 
 * \return bool
 * \sa 
 * \remarks this function is defined inline so it can be used on GPU and CPU
 * \author Mauch
 */
inline RT_HOSTDEVICE bool hitRenderFringeProj(geomRenderRayStruct &ray, Mat_hitParams hitParams, MatFringeProj_GeomRender_params params, double t_hit, int geometryID)
{
    ray.position = ray.position + t_hit * ray.direction;
	ray.currentGeometryID=geometryID;
	ray.opl=ray.opl+ray.nImmersed*t_hit;

	// transform ray to local coordinate system
	double3 l_rayPos=ray.position-params.geomRoot;
	//rotateRayInv(&l_rayPos,params.geomTilt);
    if (params.fringeType==FT_SINUS)
    {
        if (params.fringeOrientation==O_X)
            ray.cumFlux+=ray.flux*params.power*(cos(2*PI*l_rayPos.x/params.fringePeriod+params.fringePhase)+1);
        else
            ray.cumFlux+=ray.flux*params.power*(cos(2*PI*l_rayPos.y/params.fringePeriod+params.fringePhase)+1);
    }
   
    ray.running=false; // stop ray
	return true;
}

#endif


