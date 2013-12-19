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
#include <optixu/optixu_matrix.h>
#include <optixu/optixu_aabb.h>


//rtDeclareVariable(float3, boxmin, , );
//rtDeclareVariable(float3, boxmax, , );

RT_PROGRAM void bounds (int, float result[6])
{
  optix::Aabb* aabb = (optix::Aabb*)result;
  float3 l_boxmin=make_float3(-100,-100,0);
  float3 l_boxmax=make_float3(100,100,0);
  aabb->set(l_boxmin, l_boxmax);
}
