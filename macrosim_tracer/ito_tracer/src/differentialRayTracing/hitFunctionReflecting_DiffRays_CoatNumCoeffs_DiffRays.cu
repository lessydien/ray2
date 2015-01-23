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

#include <optix.h>
#include <optix_math.h>
#include "../rayData.h"
#include "MaterialReflecting_DiffRays_hit.h"
#include "Coating_NumCoeffs_DiffRays_hit.h"

/****************************************************************************/
/*				variable definitions										*/
/****************************************************************************/

rtDeclareVariable(Mat_DiffRays_hitParams, hitParams, attribute hitParams, ); 
rtDeclareVariable(double, t_hit, attribute t_hit, ); 
rtDeclareVariable(int,               max_depth, , );
rtDeclareVariable(float,               min_flux, , );
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(diffRayStruct, prd, rtPayload, );
//rtDeclareVariable(rtObject,          top_object, , );
//rtDeclareVariable(rtObject,          top_shadower, , );
rtDeclareVariable(int,               geometryID, attribute geometryID , );
rtDeclareVariable(Coating_NumCoeffs_ReducedParams, coating_params, , );

/****************************************************************************/
/*				device functions											*/
/****************************************************************************/

__forceinline__ __device__ void reflectingCoatNumCoeffsDiffRays_anyHit_device()
{
  // if we are intersecting the geometry we started from again, we ignore the intersection
//  if (prd.currentGeometryID == geometryID)
//  {
//    rtIgnoreIntersection();
//  }
}

__forceinline__ __device__ void reflectingCoatNumCoeffsDiffRays_closestHit_device( Mat_DiffRays_hitParams hitParams, double t_hit )
{
	  bool coat_reflected=true; // standard, this material is reflecting
	  coat_reflected=hitCoatingNumCoeff_DiffRays(prd, hitParams, coating_params); // see what the coating wants
	  // if we still have reflection, reflect the ray. Otherwise dont alter its direction...
	  if (coat_reflected)
		hitReflecting_DiffRays(prd, hitParams, t_hit, geometryID, coat_reflected);
  if ( (prd.depth>max_depth) || (prd.flux<min_flux) )
	  prd.running=false;
}

/********************************************************************************/
/*					OptiX programs												*/
/********************************************************************************/

RT_PROGRAM void anyHit()
{
  reflectingCoatNumCoeffsDiffRays_anyHit_device();
}


RT_PROGRAM void closestHit()
{
  reflectingCoatNumCoeffsDiffRays_closestHit_device( hitParams, t_hit );
}
