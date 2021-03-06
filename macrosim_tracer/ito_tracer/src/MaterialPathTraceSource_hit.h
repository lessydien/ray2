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

#ifndef MATERIALPATHTRACESOURCE_HIT_H
#define MATERIALPATHTRACESOURCE_HIT_H

#include "rayTracingMath.h"
#include "Material_hit.h"

/* declare class */
/**
  *\class   MatPathTraceSource_params 
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
class MatPathTraceSource_params
{
public:
	double2 acceptanceAngleMax; 
	double2 acceptanceAngleMin; 
	double3 tilt;
	double flux;
};

/**
 * \detail hitReflecting 
 *
 * modifies the raydata according to the parameters of the material
 *
 * \param[in] rayStruct_PathTracing &ray, Mat_hitParams hitParams, double t_hit, int geometryID
 * 
 * \return bool
 * \sa 
 * \remarks this function is defined inline so it can be used on GPU and CPU
 * \author Mauch
 */
inline RT_HOSTDEVICE bool hitPathTraceSource(rayStruct &ray, Mat_hitParams hitParams, MatPathTraceSource_params params, double t_hit, int geometryID)
{
	rayStruct_PathTracing* ray_interpreted;
	// we have a path tracing ray here ... hopefully ...
	ray_interpreted=reinterpret_cast<rayStruct_PathTracing*>(&ray);

    ray.position = ray.position + t_hit * ray.direction;
	ray.currentGeometryID=geometryID;
	ray.opl=ray.opl+ray.nImmersed*t_hit;
	//ray.direction=reflect(ray.direction,hitParams.normal);
	//ray.flux=ray.flux;
	// check acceptance angle
	double3 e_x=make_double3(1,0,0);
	double3 e_y=make_double3(0,1,0);
	rotateRay(&e_x, params.tilt);
	rotateRay(&e_y, params.tilt);
	double projX=dot(e_x,ray.direction);
	double projY=dot(e_y,ray.direction);
	// check acceptance angle in x
	if ( (projX < cos(params.acceptanceAngleMax.x)) && (projX>-cos(params.acceptanceAngleMin.x)) && (projY < cos(params.acceptanceAngleMax.y)) && (projY>-cos(params.acceptanceAngleMin.y)) )
		ray_interpreted->result=ray_interpreted->result+ray.flux*params.flux;
	return true;
}

#endif


