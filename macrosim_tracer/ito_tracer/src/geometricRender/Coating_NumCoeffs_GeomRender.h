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

/**\file Coating_NumCoeffs_GeomRender.h
* \brief 
* 
*           
* \author Mauch
*/

#ifndef COATING_NUMCOEFFS_GEOMRENDER_H
  #define COATING_NUMCOEFFS_GEOMRENDER_H

#include <optix.h>
#include "../Coating_NumCoeffs.h"
#include <optix_math.h>
#include "../rayData.h"
#include "../GlobalConstants.h"
#include "Coating_NumCoeffs_GeomRender_hit.h"
#include <stdio.h>

#define PATH_TO_HIT_COAT_NUMCOEFFS_GEOMRENDER "_CoatNumCoeffs_GeomRender"

///* declare class */
///**
//  *\class   Coating_NumCoeffs_GeomRender_FullParams
//  *\brief   base class of full set of params of Coating_NumCoeffs_GeomRender
//  *
//  *         
//  *
//  *         \todo
//  *         \remarks           
//  *         \sa       NA
//  *         \date     04.01.2011
//  *         \author  Mauch
//  *
//  */
//class Coating_NumCoeffs_GeomRender_FullParams: public Coating_NumCoeffs_FullParams
//{
//	public:
//		//CoatingType type;
//		//double t; // amplitude transmission coefficient
//		//double r; // amplitude reflection coefficient
//};

/* declare class */
/**
  *\class   Coating_NumCoeffs_GeomRender
  *\ingroup Coating
  *\brief   
  *
  *         
  *
  *         \todo
  *         \remarks           
  *         \sa       NA
  *         \date     04.01.2011
  *         \author  Mauch
  *
  */
class Coating_NumCoeffs_GeomRender: public Coating_NumCoeffs
{
  protected:
 //   char* path_to_ptx;
	//Coating_NumCoeffs_GeomRender_FullParams *fullParamsPtr;
	//Coating_NumCoeffs_GeomRender_ReducedParams *reducedParamsPtr;
	//CoatingError calcCoatingCoeffs(double lambda, double3 normal, double3 direction);
		
  public:
	 Coating_NumCoeffs_GeomRender() // standard constructor
	 {
		 this->fullParamsPtr=new Coating_NumCoeffs_FullParams(); 
		 this->fullParamsPtr->type = CT_NUMCOEFFS;
		 this->fullParamsPtr->t=1;
		 this->fullParamsPtr->r=0;
		 this->reducedParamsPtr=new Coating_NumCoeffs_ReducedParams();
		 path_to_ptx=(char*)malloc(512*sizeof(char));
		 sprintf( path_to_ptx, "%s", PATH_TO_HIT_COAT_NUMCOEFFS_GEOMRENDER );

	 }
	 ~Coating_NumCoeffs_GeomRender() // destructor
	 {
		 delete this->fullParamsPtr;
		 delete this->reducedParamsPtr;
		 delete path_to_ptx;
	 }
     bool  hit(geomRenderRayStruct &ray, Mat_GeomRender_hitParams hitParams);

	CoatingError createCPUSimInstance(double lambda);
};

#endif

