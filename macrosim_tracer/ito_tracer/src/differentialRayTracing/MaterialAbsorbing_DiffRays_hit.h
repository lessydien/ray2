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

/**\file MaterialAbsorbing_hit.h
* \brief header file containing definitions of functions and datastructures that can be used on the GPU as well
* 
*           
* \author Mauch
*/

#ifndef MATERIALABSORBING_DIFFRAYS_HIT_H
#define MATERIALABSORBING_DIFFRAYS_HIT_H

#include "internal/optix_declarations.h"  // For RT_HOSTDEVICE
#include "Material_DiffRays_hit.h"
#include "../MaterialAbsorbing_hit.h"
#include "../Material_hit.h"

/* declare class */
/**
  *\class   MatAbsorbing_Params 
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
class MatAbsorbing_DiffRays_params : public MatAbsorbing_params
{
public:

};

/**
 * \detail hitAbsorbing_DiffRays 
 *
 * modifies the raydata according to the parameters of the material
 *
 * \param[in] diffRayStruct &ray, Mat_DiffRays_hitParams hitParams, double t_hit, int geomID
 * 
 * \return bool
 * \sa 
 * \remarks this function is defined inline so it can be used on GPU and CPU
 * \author Mauch
 */
inline RT_HOSTDEVICE bool hitAbsorbing_DiffRays(diffRayStruct &ray, Mat_DiffRays_hitParams hitParams, double t_hit, int geomID)
{
	if (!hitAbsorbing(ray, hitParams, t_hit, geomID) )
		return false;
	// do the stuff specific for differential rays
	// calc new wavefront radii
	double2 newWavefrontRad;
	newWavefrontRad.x=(ray.wavefrontRad.x-t_hit*ray.nImmersed);
	newWavefrontRad.y=(ray.wavefrontRad.y-t_hit*ray.nImmersed);
	// calc local flux from wavefront radii
	// see Diss. O. Stolz: "Differentielles Ray Tracing für spezielle Beleuchtungssysteme", Uni Erlangen 2010 (eq.3.10 on page 19)
	ray.flux=ray.flux*abs(ray.wavefrontRad.x/newWavefrontRad.x)*abs(ray.wavefrontRad.y/newWavefrontRad.y);
	// save new wavefront radii
	ray.wavefrontRad=newWavefrontRad;

	return true;
};

#endif


