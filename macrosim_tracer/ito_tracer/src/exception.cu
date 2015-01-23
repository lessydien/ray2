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

rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
//rtDeclareVariable( float3, error, , ) = make_float3(1,0,0); 
//rtBuffer<float3, 2> output_buffer; 

RT_PROGRAM void exception_program( void ) 
{ 
	const unsigned int code = rtGetExceptionCode(); 
//	if( code == RT_EXCEPTION_STACK_OVERFLOW )
//		output_buffer[launch_index] = error; 
//	else 
	rtPrintExceptionDetails();
	rtPrintf( "Caught exception at launch index (%d,%d)\n", launch_index.x, launch_index.y);
}
