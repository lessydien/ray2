
/*
 * Copyright (c) 2008 - 2009 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and proprietary
 * rights in and to this software, related documentation and any modifications thereto.
 * Any use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from NVIDIA Corporation is strictly
 * prohibited.
 *
 * TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
 * AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
 * INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE.  IN NO EVENT SHALL NVIDIA OR ITS SUPPLIERS BE LIABLE FOR ANY
 * SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
 * LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
 * BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR
 * INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGES
 */

#include <optix.h>
#include <optix_math.h>
//#include <commonStructs.h>
//#include "helpers.h"
#include "rayData.h"
#include "rayTracingMath.h"
#include "MaterialLinearGrating1D_hit.h"
#include "Coating_NumCoeffs_hit.h"

/****************************************************************************/
/*				variable definitions										*/
/****************************************************************************/

rtDeclareVariable(Mat_hitParams, hitParams, attribute hitParams, ); 
rtDeclareVariable(double, t_hit, attribute t_hit, ); 
rtDeclareVariable(MatLinearGrating1D_params, params, , ); 
rtDeclareVariable(int,               max_depth, , );
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(float,             scene_epsilon, , );
rtDeclareVariable(rayStruct, prd, rtPayload, );
rtDeclareVariable(rtObject,          top_object, , );
rtDeclareVariable(int,               geometryID, attribute geometryID , );
rtDeclareVariable(Coating_NumCoeffs_ReducedParams, coating_params, , );

/****************************************************************************/
/*				device functions											*/
/****************************************************************************/

__device__ void linearGrating1D_anyHit_device()
{
  if (prd.currentGeometryID == geometryID)
  {
    rtIgnoreIntersection();
  }
}

__device__ void linearGrating1D_closestHit_device( Mat_hitParams hitParams, double t_hit )
{
  if (prd.depth < max_depth)
  {
    rtPrintf("closest hit ID %i \n", geometryID);
	rtPrintf("flux %.20lf \n", prd.flux);
    // standard the grating ist transmitting
    bool reflected=false;
	// n1 equal zero indicates reflecting grating
    if (params.nRefr1==0)
		reflected=true;
	// now see what the coating wants to have
	reflected=hitCoatingNumCoeff(prd, hitParams, coating_params);
    // diffract ray with flag indicating reflection or not
    hitLinearGrating1D(prd, hitParams, params, t_hit, geometryID, reflected);
  }
  else
  {
    rtPrintf("ray stopped in hitLinGrat1D!!");
    prd.running=false; // stop ray
  }
  
}

/********************************************************************************/
/*					OptiX programs												*/
/********************************************************************************/

RT_PROGRAM void anyHit()
{
  linearGrating1D_anyHit_device();
}


RT_PROGRAM void closestHit()
{
  linearGrating1D_closestHit_device( hitParams, t_hit );
}
