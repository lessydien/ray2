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
#include "../rayTracingMath.h"
#include "Scatter_TorranceSparrow1D_DiffRays_hit.h"
#include "MaterialReflecting_DiffRays_hit.h"
#include "Coating_NumCoeffs_DiffRays_hit.h"

/****************************************************************************/
/*				variable definitions										*/
/****************************************************************************/

rtDeclareVariable(Mat_DiffRays_hitParams, hitParams, attribute hitParams, ); 
rtDeclareVariable(double, t_hit, attribute t_hit, ); 
rtDeclareVariable(ScatTorranceSparrow1D_DiffRays_params, scatterParams, , ); 
rtDeclareVariable(MatReflecting_DiffRays_params, params, , ); 
rtDeclareVariable(int,               max_depth, , );
rtDeclareVariable(float,               min_flux, , );
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(float,             scene_epsilon, , );
rtDeclareVariable(diffRayStruct, prd, rtPayload, );
//rtDeclareVariable(rtObject,          top_object, , );
rtDeclareVariable(int,               geometryID, attribute geometryID , );
rtDeclareVariable(Coating_NumCoeffs_DiffRays_ReducedParams, coating_params, , );

/****************************************************************************/
/*				device functions											*/
/****************************************************************************/

__forceinline__ __device__ void reflectingScatTorrSparr1DDiffRaysCoatNumCoeffsDiffRays_anyHit_device()
{
//  if (prd.currentGeometryID == geometryID)
//  {
//    rtIgnoreIntersection();
//  }

}

__forceinline__ __device__ void reflectingScatTorrSparr1DDiffRaysCoatNumCoeffsDiffRays_closestHit_device( Mat_DiffRays_hitParams hitParams, double t_hit )
{

    bool coat_reflected=hitCoatingNumCoeff_DiffRays(prd, hitParams, coating_params); // see weter coating wants this material to be reflecting
		hitReflecting_DiffRays(prd, hitParams, t_hit, geometryID, coat_reflected); // if coating wants reflections, reflect. otherwise don't alter the rays direction
	
	// scatter ray
	hitTorranceSparrow1D_DiffRays(prd, hitParams, scatterParams);

  if ( (prd.depth>max_depth) || (prd.flux<min_flux) )
	  prd.running=false;  
}

/********************************************************************************/
/*					OptiX programs												*/
/********************************************************************************/

RT_PROGRAM void anyHit()
{
  reflectingScatTorrSparr1DDiffRaysCoatNumCoeffsDiffRays_anyHit_device();
}


RT_PROGRAM void closestHit()
{
  reflectingScatTorrSparr1DDiffRaysCoatNumCoeffsDiffRays_closestHit_device( hitParams, t_hit ); 
}
