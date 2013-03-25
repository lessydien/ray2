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

/**\file Coating_DiffRays.h
* \brief 
* 
*           
* \author Mauch
*/

#ifndef COATING_DIFFRAYS_H
  #define COATING_DIFFRAYS_H

#include <optix.h>
#include <optix_math.h>
#include "../rayData.h"
#include "../GlobalConstants.h"
#include "Coating_DiffRays_hit.h"
#include "../Coating.h"
#include <stdlib.h>
//#include "Coating_DiffRaysLib.h"

/* declare class */
/**
  *\class   Coating_DiffRays _FullParams
  *\ingroup Coating
  *\brief   base class of full set of params of coatings
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
class Coating_DiffRays_FullParams : public Coating_FullParams
{
public:
	//CoatingType type;
	//double t; // amplitude transmission coefficient
	//double r; // amplitude reflection coefficient
};

/* declare class */
/**
  *\class   Coating_DiffRays
  *\ingroup Coating
  *\brief   base class of coatings
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
class Coating_DiffRays : public Coating
{
  protected:
	Coating_DiffRays_FullParams *fullParamsPtr;
	Coating_DiffRays_ReducedParams *reducedParamsPtr;
	CoatingError calcCoating_DiffRaysCoeffs(double lambda, double3 normal, double3 direction);
			
  public:
    bool update;

    Coating_DiffRays()
	{
//		fullParamsPtr=new Coating_DiffRays_FullParams();
		fullParamsPtr=NULL;
		reducedParamsPtr=NULL;
		path_to_ptx=(char*)malloc(512*sizeof(char));
	}
	virtual ~Coating_DiffRays()
	{
		if (fullParamsPtr != NULL)
			delete fullParamsPtr;
		if (reducedParamsPtr != NULL)
			delete reducedParamsPtr;
		if (path_to_ptx != NULL)
			delete path_to_ptx;
	}
	virtual CoatingError setFullParams(Coating_DiffRays_FullParams* ptrIn);
	virtual Coating_DiffRays_FullParams* getFullParams(void);
	virtual Coating_DiffRays_ReducedParams* getReducedParams(void);
	virtual bool hit(rayStruct &ray, Mat_hitParams hitParams);
	virtual bool hit(diffRayStruct &ray, Mat_hitParams hitParams);
	virtual bool hit(gaussBeamRayStruct &ray, gaussBeam_geometricNormal normal);
};

#endif

