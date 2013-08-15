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

rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(rayStruct, prd, rtPayload, ); // get per-ray-data structure

RT_PROGRAM void miss()
{
//	rtPrintf("ray stopped in miss!! \n");
	prd.running=false; // stop ray
  //prd_position.position = bg_color;
	//prd_position.position.x=1.0f;
	//prd_position.position.y=2.0f;
	//prd_position.position.z=3.33f;

  //prd.position=make_double3((double)100);
}
