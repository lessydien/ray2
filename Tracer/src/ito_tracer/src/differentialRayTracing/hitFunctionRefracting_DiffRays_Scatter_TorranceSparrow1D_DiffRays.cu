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
#include "MaterialRefracting_DiffRays_hit.h"

/****************************************************************************/
/*				variable definitions										*/
/****************************************************************************/

rtDeclareVariable(Mat_DiffRays_hitParams, hitParams, attribute hitParams, ); 
rtDeclareVariable(double, t_hit, attribute t_hit, ); 
rtDeclareVariable(ScatTorranceSparrow1D_DiffRays_params, scatterParams, , ); 
rtDeclareVariable(MatRefracting_DiffRays_params, params, , ); 
rtDeclareVariable(int,               max_depth, , );
rtDeclareVariable(float,               min_flux, , );
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(float,             scene_epsilon, , );
rtDeclareVariable(diffRayStruct, prd, rtPayload, );
//rtDeclareVariable(rtObject,          top_object, , );
rtDeclareVariable(int,               geometryID, attribute geometryID , );

/****************************************************************************/
/*				device functions											*/
/****************************************************************************/

__forceinline__ __device__ void refractingScatTorrSparr1DDiffRays_anyHit_device()
{
//  if (prd.currentGeometryID == geometryID)
//  {
//    rtIgnoreIntersection();
//  }
}

__forceinline__ __device__ void refractingScatTorrSparr1DDiffRays_closestHit_device( Mat_DiffRays_hitParams hitParams, double t_hit )
{

    bool coat_reflected=false;
    // refract ray
    hitRefracting_DiffRays(prd, hitParams, params, t_hit, geometryID, coat_reflected);
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
  refractingScatTorrSparr1DDiffRays_anyHit_device();
}


RT_PROGRAM void closestHit()
{
  refractingScatTorrSparr1DDiffRays_closestHit_device( hitParams, t_hit ); 
}
