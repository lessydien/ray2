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
#include "randomGenerator.h"
#include "rayTracingMath.h"
#include "geometricRender/GeometricRenderField.h"
#include "geometricRender/GeometricRenderField_hostDevice.h"
#include "time.h"

#ifndef PI
	#define PI ((double)3.141592653589793238462643383279502884197169399375105820)
#endif

rtDeclareVariable(renderFieldParams, params, , );

rtDeclareVariable(long long,        launch_offsetX, , );
rtDeclareVariable(long long,        launch_offsetY, , );

rtDeclareVariable(float,         scene_epsilon, , );

rtBuffer<double, 2>              output_buffer;
rtBuffer<uint, 2>              seed_buffer;

rtDeclareVariable(rtObject,      top_object, , );

rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint2, launch_dim,   rtLaunchDim, );

//#define TIME_VIEW


/**********************************************************/
// device functions for distributing ray positions
/**********************************************************/

RT_PROGRAM void rayGeneration_geomRender()
{
#ifdef TIME_VIEW
  clock_t t0, t1;
  double time;
  t0=clock();
#endif

  output_buffer[launch_index]=0;

#ifdef TIME_VIEW
  t1 = clock(); 
  time = (double)(t1-t0);
  rtPrintf("time elapsed while creating ray: %lf ms\n", time);
#endif 


  for (unsigned int jRay=0; jRay < params.nrRayDirections.x*params.nrRayDirections.y; jRay++)
  {
      geomRenderRayStruct prd=createRay(launch_index.x,launch_index.y,jRay, params, params.nImmersed, seed_buffer[launch_index]);

      float3 ray_origin = make_float3(prd.position);

      float3 ray_direction = make_float3(prd.direction);

      optix::Ray ray = optix::make_Ray(ray_origin, ray_direction, 0, scene_epsilon, RT_DEFAULT_MAX);

      for(;;) 
      {
        rtTrace(top_object, ray, prd);

   	    // update ray
	    ray.origin=make_float3(prd.position);
	    ray.direction=make_float3(prd.direction);

        if(!prd.running) 
        {
           break;
        }
      }
      output_buffer[launch_index] = prd.cumFlux;//.direction;//prd.position;
      seed_buffer[launch_index] = prd.currentSeed; // save seed for nex jRay
  }
  
}

RT_PROGRAM void exception()
{
  const unsigned int code = rtGetExceptionCode();
  rtPrintf( "Caught exception 0x%X at launch index (%d,%d)\n", code, (launch_index.x+launch_offsetX), (launch_index.y+launch_offsetY) );
//  output_buffer[launch_index] = prd.position;
}
