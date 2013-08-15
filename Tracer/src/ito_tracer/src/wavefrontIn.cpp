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

/**\file wavefrontIn.cpp
* \brief collection of functions to read in wavefronts represented by zernnike polynoms and convert them to a geometric rayfield
* 
*           
* \author Mauch
*/

//#include "stdafx.h"
#include "wavefrontIn.h"
#include <math.h>
#include <algorithm>
using namespace std;


//double wavfronIn_single_polyval_rowc(double pol[],double  x,double y,int m, int n)
//{
//	double number_coef;
//    double wert = 0;
//	int i, j;
//	int counter = 0;
//	number_coef=max(m,n);
//	int order  =  ceil(0.5*(-3+sqrt((double)(1+8*number_coef))));
//	for (i = 0; i <= order ;  ++i)
//    		{
//    		 for (j = 0; j <= i; ++j)
//    			{
//					++counter;
//					if (counter < (number_coef+1))
// 			 		{    
//    				wert += pol[counter-1]*pow(x,(i-j))*pow(y,j);
//					}
//				}
//			
//			}
//	return wert;
//}

void wavfronInt_spatial_polynomial_row_vectorized(double matrix[],int sx,int sm, double cx, double cy, double* &value)
	{
	double order = ceil(0.5*(-3+sqrt((double)(1+8*sm))));
	value = new double[sx];
	for(int k = 1; k <= sx; ++k)
		value[k-1] = 0;	
	int counter = 0;
	for(int i = 0; i <=order ; ++i)
	{
		for(int j = 0; j <= i; ++j)
		{
			++counter;
			if(counter < (sm+1))
			{
				for(int k = 1; k <= sx; ++k)
				{
					value[k-1] += matrix[(counter-1)*sx + k -1 ]*pow(cx,(i-j))*pow(cy,j);
				}
			}
		}
	}

}

void wavfronInt_partialpol_row(double P[],int pixelpol_spatialRows, int c, double * &partialP, int &partialRows, double reference)
	{
		double s = pixelpol_spatialRows;
		double order=ceil(0.5*(-3+sqrt(1+8*s)));
		partialP = new double[(int)((order+2)*(order+3)/2)+1];
		for(int i = 1; i <= s; ++i)
			partialP[i-1] = P[i-1];
		for(int i = s+1 ; i < ((order+2)*(order+3)/2)+1;++i)
			partialP[i-1]=0;
		int counter = 0;

		switch(c)
		{
		case 1:
			for(int i = 0; i <= order; ++i)
			{
				for(int j = 0; j <=i ; ++j)
					{
					++counter;
					partialP[counter-1] = (partialP[(i+2)*(i+3)/2-i+j-1-1]*(i-j+1))/reference;
					}
			}
			break;
		case 2:
			for(int i = 0; i <= order; ++i)
			{
				for(int j = 0; j <= i; ++j)
					{
					++counter;
					partialP[counter-1] = (partialP[(i+2)*(i+3)/2-i+j-1]*(j+1))/reference;
					}
			}
			break;
			
		}	
		 		partialRows = s-order-c+1;
 }
