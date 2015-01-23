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

#ifndef MATERIALVOLUMEABSORBING_HIT_H
#define MATERIALVOLUMEABSORBING_HIT_H

#include "internal/optix_declarations.h"  // For RT_HOSTDEVICE
#include "Material_hit.h"
#include "rayTracingMath.h"
#include "randomGenerator.h"


/* declare class */
/**
  *\class   MatVolumeAbsorbing_Params 
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
class MatVolumeAbsorbing_params
{
public:
	double absorbCoeff;

};

/**
 * \detail hitAbsorbing 
 *
 * modifies the raydata according to the parameters of the material
 *
 * \param[in] rayStruct &ray, Mat_hitParams hitParams, double t_hit, int geomID
 * 
 * \return bool
 * \sa 
 * \remarks this function is defined inline so it can be used on GPU and CPU
 * \author Mauch
 */
inline RT_HOSTDEVICE bool hitVolumeAbsorbing(rayStruct &ray, Mat_hitParams hitParams, MatVolumeAbsorbing_params params, double t_hit, int geomID, bool coat_reflected)
{
	Mat_hitParams debugHitParams=hitParams;
	MatVolumeAbsorbing_params debugParams=params;
	// if the ray was inside 
	double l_t=0;
	bool l_surfaceHit=true;
	// if the ray already was inside the volumscatterer
	if (ray.currentGeometryID == VOLABSORB_GEOMID)
	{
		uint32_t x1[5];
		RandomInit(ray.currentSeed, x1); // seed random generator
		for (unsigned int i=0; i<50; ++i) // in principle, this should be an endless loop, we inserted a maximum number of iterations for performance reasons
		{
			// randomly generate free path until next scatter
			// the idea is to generate a random distance between zero and 10 times the mean free path
			// then, the probability of this distance is compared to a random number between 0 and 1
			l_t=Random(x1)*10/params.absorbCoeff;
			double l_prob=exp(-l_t*params.absorbCoeff);
			double l_threshold=Random(x1);
			if (l_prob>l_threshold)
			{
				ray.currentSeed=x1[4];
				break;
			}
		}
		// if we have a volume hit
		if (l_t<t_hit)
		{
			t_hit=l_t;
			l_surfaceHit=false;
			ray.running=false; // stop ray
		}
		ray.currentSeed=x1[4]; // save seed for next round
	}
	// update ray position 
	ray.position=ray.position+t_hit*ray.direction;
	ray.currentGeometryID=geomID;
	ray.opl=ray.opl+ray.nImmersed*t_hit;

	if (l_surfaceHit) // if we have a surface hit
	{
		// if the coating wants us to have reflection, we do reflection here instead of refraction
		if (coat_reflected)
			ray.direction=reflect(ray.direction,hitParams.normal);
		else
		{
			// set info wether we are noe inside the absorbing material or not
			if ( ray.currentGeometryID==VOLABSORB_GEOMID)
				ray.currentGeometryID=geomID; // signal that the ray is on the surface of geometry
			else
				ray.currentGeometryID=VOLABSORB_GEOMID; // signal that the ray is inside the volum scatterer
		}
	}


	return true;
};

#endif


