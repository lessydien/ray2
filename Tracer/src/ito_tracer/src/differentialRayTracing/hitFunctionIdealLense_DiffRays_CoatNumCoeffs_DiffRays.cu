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
#include "MaterialIdealLense_DiffRays_hit.h"
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
rtDeclareVariable(MatIdealLense_DiffRays_params,               params, , );

/****************************************************************************/
/*				device functions											*/
/****************************************************************************/

__forceinline__ __device__ void idealLenseCoatNumCoeffsDiffRays_anyHit_device()
{
  // if we are intersecting the geometry we started from again, we ignore the intersection
//  if (prd.currentGeometryID==geometryID)
//  {
//    rtIgnoreIntersection();
//  }
}

__forceinline__ __device__ void idealLenseCoatNumCoeffsDiffRays_closestHit_device( Mat_DiffRays_hitParams hitParams, double t_hit )
{
  rtPrintf("closest hit ID %i \n", geometryID);
  rtPrintf("flux %.20lf \n", prd.flux);
  bool coat_reflected=hitCoatingNumCoeff(prd, hitParams, coating_params);
  hitIdealLense_DiffRays(prd, hitParams, params, t_hit, geometryID, coat_reflected);
  
  if ( (prd.depth>max_depth) || (prd.flux<min_flux) )
	  prd.running=false; 
}

/********************************************************************************/
/*					OptiX programs												*/
/********************************************************************************/

RT_PROGRAM void anyHit()
{
  idealLenseCoatNumCoeffsDiffRays_anyHit_device();
}


RT_PROGRAM void closestHit()
{
  idealLenseCoatNumCoeffsDiffRays_closestHit_device( hitParams, t_hit );
}
