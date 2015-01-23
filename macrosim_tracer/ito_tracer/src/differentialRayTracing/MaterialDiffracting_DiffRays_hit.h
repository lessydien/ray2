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

/**\file MaterialDiffracting_hit.h
* \brief header file containing definitions of functions and datastructures that can be used on the GPU as well
* 
*           
* \author Mauch
*/

#ifndef MATERIALDIFFRACTING_DIFFRAYS_HIT_H
#define MATERIALDIFFRACTING_DIFFRAYS_HIT_H

#include "../rayTracingMath.h"
#include "Material_DiffRays_hit.h"
#include "../MaterialDiffracting_hit.h"


/* declare class */
/**
  *\class   MatDiffracting_DiffRays_params
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
class MatDiffracting_DiffRays_params : public MatDiffracting_params
{
public:
	//double n1; // refractive index1
	//double n2; // refractive index2
	//double2 diffConeAngleMax;
	//double2 diffConeAngleMin;
	//double phiRotZ;
	////double t; // amplitude transmission coefficient
	////double r; // amplitude reflection coefficient
};

/**
 * \detail hitDiffracting_DiffRays 
 *
 * modifies the raydata according to the parameters of the material
 *
 * \param[in] diffRayStruct &prd, Mat_DiffRays_hitParams hitParams, MatDiffracting_DiffRays_params params, double t_hit, int geomID, bool coat_reflected
 * 
 * \return bool
 * \sa 
 * \remarks this function is defined inline so it can be used on GPU and CPU
 * \author Mauch
 */
inline RT_HOSTDEVICE bool hitDiffracting_DiffRays(diffRayStruct &prd, Mat_DiffRays_hitParams hitParams, MatDiffracting_DiffRays_params params, double t_hit, int geomID, bool coat_reflected)
{
	// do the geometric hit
	if (!hitDiffracting(prd, hitParams, params, t_hit, geomID, coat_reflected) )
		return false;
	// do the specific stuff for differential rays
	// calc new wavefront radii
	double2 newWavefrontRad;
	newWavefrontRad.x=(prd.wavefrontRad.x-t_hit);
	newWavefrontRad.y=(prd.wavefrontRad.y-t_hit);
	// calc local flux from wavefront radii
	prd.flux=prd.flux*abs(prd.wavefrontRad.x/newWavefrontRad.x)*abs(prd.wavefrontRad.y/newWavefrontRad.y);
	// set new wavefront radii to zero as we start a new sphercial wave here. But we have to move the ray a small distance out of the caustic of this spherical wave...
	double epsilon=10*prd.lambda;
	prd.wavefrontRad=make_double2(-epsilon,-epsilon);
	prd.position=prd.position+epsilon*prd.direction;
	prd.flux=1/(epsilon*epsilon)*prd.flux; // adjust flux accordingly
	prd.opl=prd.opl+epsilon; // adjust opl accordingly
	// main directionX is oriented perpendicular to global y-axis, has to be perpendicular to rayDircetion and has to be of unit length...
	prd.mainDirX.y=0;
	prd.mainDirY.x=0;
	if (prd.direction.z!=0)
	{
		prd.mainDirX.x=1/sqrt(1-prd.direction.x/prd.direction.z);
		prd.mainDirX.z=-prd.mainDirX.x*prd.direction.x/prd.direction.z;
		prd.mainDirY.y=1/sqrt(1-prd.direction.y/prd.direction.z);
		prd.mainDirY.z=-prd.mainDirY.y*prd.direction.x/prd.direction.z;
	}
	else
	{
		if (prd.direction.x != 0)
		{
			prd.mainDirX.z=1/sqrt(1-prd.direction.z/prd.direction.x);
			prd.mainDirX.x=-prd.mainDirX.z*prd.direction.z/prd.direction.x;
		}
		else
			prd.mainDirX=make_double3(1,0,0);
		if (prd.direction.y != 0)
		{
			prd.mainDirY.z=1/sqrt(1-prd.direction.z/prd.direction.y);
			prd.mainDirY.y=-prd.mainDirY.z*prd.direction.z/prd.direction.y;
		}
		else
			prd.mainDirY=make_double3(0,1,0);
	}

	return true;
}

#endif


