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

/**\file wavefrontIn.h
* \brief collection of functions to read in wavefronts represented by zernnike polynoms and convert them to a geometric rayfield. These functions here are defined inline so they can be used on the GPU as well
* 
*           
* \author Mauch
*/

#ifndef WAVEFRONTINPUT_H
#define WAVEFRONTINPUT_H

#include "rayTracingMath.h"

void wavfronInt_spatial_polynomial_row_vectorized(double matrix[],int sx,int sm, double cx, double cy, double* &valueTR);
inline RT_HOSTDEVICE double wavfronIn_single_polyval_rowc(double pol[],double  x,double y,int m, int n)
{
	double number_coef;
    double wert = 0;
	int i, j;
	int counter = 0;
	number_coef=max(m,n);
	int order  =  ceil(0.5*(-3+sqrt((double)(1+8*number_coef))));
	for (i = 0; i <= order ;  ++i)
    		{
    		 for (j = 0; j <= i; ++j)
    			{
					++counter;
					if (counter < (number_coef+1))
 			 		{    
    				wert += pol[counter-1]*pow(x,(i-j))*pow(y,j);
					}
				}
			
			}
	return wert;
};
void wavfronInt_partialpol_row(double P[],int pixelpol_spatialRows, int c, double * &partialP, int &partialRows, double radius_source_reference);

#endif