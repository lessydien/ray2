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
#include "rayTracingMath.h"
#include "Scatter_TorranceSparrow1D_hit.h"
#include "MaterialLinearGrating1D_hit.h"
#include "Coating_NumCoeffs_hit.h"

/****************************************************************************/
/*				variable definitions										*/
/****************************************************************************/

rtDeclareVariable(Mat_hitParams, hitParams, attribute hitParams, ); 
rtDeclareVariable(double, t_hit, attribute t_hit, ); 
rtDeclareVariable(ScatTorranceSparrow1D_params, scatterParams, , ); 
rtDeclareVariable(MatLinearGrating1D_params, params, , ); 
rtDeclareVariable(int,               max_depth, , );
rtDeclareVariable(float,               min_flux, , );
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(float,             scene_epsilon, , );
rtDeclareVariable(rayStruct, prd, rtPayload, );
//rtDeclareVariable(rtObject,          top_object, , );
rtDeclareVariable(int,               geometryID, attribute geometryID , );
rtDeclareVariable(Coating_NumCoeffs_ReducedParams, coating_params, , );

/****************************************************************************/
/*				device functions											*/
/****************************************************************************/

__forceinline__ __device__ void linGrat1DScatTorrSparr1DCoatNumCoeffs_anyHit_device()
{
//  if (prd.currentGeometryID == geometryID)
//  {
//    rtIgnoreIntersection();
//  }

}

__forceinline__ __device__ void linGrat1DScatTorrSparr1DCoatNumCoeffs_closestHit_device( Mat_hitParams hitParams, double t_hit )
{
    // standard the grating ist transmitting
    bool reflected=false;
	// n1 equal zero indicates reflecting grating
    if (params.nRefr1==0)
		reflected=true;
	// now see what the coating wants to have
	reflected=hitCoatingNumCoeff(prd, hitParams, coating_params);
    // diffract ray with flag indicating reflection or not
    hitLinearGrating1D(prd, hitParams, params, reflected, t_hit, geometryID);
    // scatter ray diffracted ray
    hitTorranceSparrow1D(prd, hitParams, scatterParams);

  if ( (prd.depth>max_depth) || (prd.flux<min_flux) )
	  prd.running=false;  
}

/********************************************************************************/
/*					OptiX programs												*/
/********************************************************************************/

RT_PROGRAM void anyHit()
{
  linGrat1DScatTorrSparr1DCoatNumCoeffs_anyHit_device();
}


RT_PROGRAM void closestHit()
{
  linGrat1DScatTorrSparr1DCoatNumCoeffs_closestHit_device( hitParams, t_hit ); 
}
