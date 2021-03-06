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
#include "MaterialAbsorbing_DiffRays_hit.h"
#include "Scatter_DoubleCauchy1D_DiffRays_hit.h"
#include "MaterialLinearGrating1D_DiffRays_hit.h"

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
rtDeclareVariable(ScatDoubleCauchy1D_params, scatterParams, , ); 
rtDeclareVariable(MatLinearGrating1D_DiffRays_params, params, , ); 


/****************************************************************************/
/*				device functions											*/
/****************************************************************************/

__forceinline__ __device__ void linGrat1DScatDoublyCauchy1DDiffRays_anyHit_device()
{
  // if we are intersecting the geometry we started from again, we ignore the intersection
//  if (prd.currentGeometryID==geometryID)
//  {
//    rtIgnoreIntersection();
//  }
}

__forceinline__ __device__ void linGrat1DScatDoublyCauchy1DDiffRays_closestHit_device( Mat_DiffRays_hitParams hitParams, double t_hit )
{
	  bool reflected=false;
	  if (params.nRefr1==0)
		reflected=true;
	  
	  hitLinearGrating1D_DiffRays(prd, hitParams, params, t_hit, geometryID, reflected);
	  hitDoubleCauchy1D_DiffRays(prd, hitParams, scatterParams);

  if ( (prd.depth>max_depth) || (prd.flux<min_flux) )
	  prd.running=false;  
}

/********************************************************************************/
/*					OptiX programs												*/
/********************************************************************************/

RT_PROGRAM void anyHit()
{
  linGrat1DScatDoublyCauchy1DDiffRays_anyHit_device();
}


RT_PROGRAM void closestHit()
{
  linGrat1DScatDoublyCauchy1DDiffRays_closestHit_device( hitParams, t_hit );
}
