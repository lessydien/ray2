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
#include <optixu/optixu_math_namespace.h>

using namespace optix;

rtBuffer<float3,  2>          result_buffer;

rtDeclareVariable(float, scene_epsilon, , );
rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(rtObject,      object, , );

struct PerRayData_shadow
{
  float attenuation;
};

RT_PROGRAM void checkVisibility()
{
  float3 ray_origin = make_float3(-100.0f, -100.0f, 0);
  float3 ray_direction = make_float3(launch_index.x,launch_index.y,1);
  PerRayData_shadow prd;
  optix::Ray ray = optix::make_Ray(ray_origin, ray_direction, 0, scene_epsilon, RT_DEFAULT_MAX);
  rtTrace(object, ray, prd);

  result_buffer[launch_index] = ray_origin;
}

RT_PROGRAM void exception()
{
}
