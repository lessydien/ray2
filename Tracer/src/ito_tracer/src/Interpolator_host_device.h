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

/**\file Interpolator_host_device.h
* \brief header file containing definitions of functions and datastructures that can be used on the GPU as well
* 
*           
* \author Mauch
*/

#ifndef INTERPOLATORHOSTDEVICE_H
  #define INTERPOLATORHOSTDEVICE_H
  
#include "internal/optix_declarations.h"  // For RT_HOSTDEVICE
#include <optix_math.h>

#define WIDTH_HOLO_BUFFER 512
#define HEIGHT_HOLO_BUFFER 512
#define DELTA_X_HOLO 2*0.001953125
#define DELTA_Y_HOLO 2*0.001953125
/**
 * \detail nearestNeighbour 
 *
 * Given array yin containing tabulated function on a regular, i.e. yin=f(x1,x2), 
 * where x1=x1_0+index*delta_x1. The value y of the nearest neighbour to x1_out, x2_out is returned in yout_ptr
 *
 * \param[in]	const double&				x1_0
 *				const double&				x2_0
 *				const double&				delta_x1
 *				const double&				delta_x2
 *				const double*				yin_tpr
 *				const unsigned int			dim_x1
 *				const unsigned int			dim_x2
 *				const double&				x1_out
 *				const double&				x2_out
 *				double*						yout_ptr
 * 
 * \return propErr
 * \sa 
 * \remarks 
 * \author Mauch
 */
inline RT_HOSTDEVICE bool nearestNeighbour_hostdevice(const double& x1_0, const double& x2_0, const double& delta_x1, const double& delta_x2, const double *yin_ptr, const unsigned int dim_x1, const unsigned int dim_x2, const double &x1_out, const double &x2_out, double *yout_ptr)
{
	// calc coordinate index of output positions
	double test=((x1_out-x1_0)/delta_x1+0.5);
	unsigned int ind_x1=floor((x1_out-x1_0)/delta_x1+0.5);
	unsigned int ind_x2=floor((x2_out-x2_0)/delta_x2+0.5);
	// if we're outside the definition range of the input we return the value at the border of that range. Therefore...
	ind_x1=max((unsigned int)0,min(dim_x1-1,ind_x1));
	ind_x2=max((unsigned int)0,min(dim_x2-1,ind_x2));
	*yout_ptr=yin_ptr[ind_x1+ind_x2*dim_x1];
	return true;
}

#endif
