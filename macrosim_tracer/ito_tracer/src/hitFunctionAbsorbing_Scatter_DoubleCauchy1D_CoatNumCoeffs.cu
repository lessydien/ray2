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
#include "rayData.h"
#include "MaterialAbsorbing_hit.h"
#include "Coating_NumCoeffs_hit.h"
#include "Scatter_DoubleCauchy1D_hit.h"

/****************************************************************************/
/*				variable definitions										*/
/****************************************************************************/

rtDeclareVariable(Mat_hitParams, hitParams, attribute hitParams, ); 
rtDeclareVariable(double, t_hit, attribute t_hit, ); 
rtDeclareVariable(int,               max_depth, , );
rtDeclareVariable(float,               min_flux, , );
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(rayStruct, prd, rtPayload, );
//rtDeclareVariable(rtObject,          top_object, , );
//rtDeclareVariable(rtObject,          top_shadower, , );
rtDeclareVariable(int,               geometryID, attribute geometryID , );
rtDeclareVariable(Coating_NumCoeffs_ReducedParams, coating_params, , );
rtDeclareVariable(ScatDoubleCauchy1D_params, scatterParams, , );

/****************************************************************************/
/*				device functions											*/
/****************************************************************************/

__forceinline__ __device__ void absorbingScatDoubleCauchy1DCoatNumCoeffs_anyHit_device()
{
  // if we are intersecting the geometry we started from again, we ignore the intersection
//  if (prd.currentGeometryID==geometryID)
//  {
//    rtIgnoreIntersection();
//  }
}

__forceinline__ __device__ void absorbingScatDoubleCauchy1DCoatNumCoeffs_closestHit_device( Mat_hitParams hitParams, double t_hit )
{
  rtPrintf("closest hit ID %i \n", geometryID);
  rtPrintf("flux %.20lf \n", prd.flux);
  hitAbsorbing(prd, hitParams, t_hit, geometryID);
  // if coating wants a reflection we keep the ray running and reflect it
  if (hitCoatingNumCoeff(prd, hitParams, coating_params))
	  prd.direction=reflect(prd.direction, hitParams.normal);

  hitDoubleCauchy1D(prd, hitParams, scatterParams);
  
  if ( (prd.depth>max_depth) || (prd.flux<min_flux) )
	  prd.running=false; 
}

/********************************************************************************/
/*					OptiX programs												*/
/********************************************************************************/

RT_PROGRAM void anyHit()
{
  absorbingScatDoubleCauchy1DCoatNumCoeffs_anyHit_device();
}


RT_PROGRAM void closestHit()
{
  absorbingScatDoubleCauchy1DCoatNumCoeffs_closestHit_device( hitParams, t_hit );
}