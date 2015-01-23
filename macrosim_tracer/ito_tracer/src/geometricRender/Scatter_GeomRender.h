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

/**\file Scatter_GeomRender.h
* \brief 
* 
*           
* \author Mauch
*/

#ifndef SCATTER_GEOMRENDER_H
  #define SCATTER_GEOMRENDER_H

#include <optix.h>
#include <optix_math.h>
#include "../rayData.h"
#include "../GlobalConstants.h"
#include "Scatter_GeomRender_hit.h"
#include "../Scatter.h"
#include <stdlib.h>
//#include "Scatter_GeomRenderLib.h"

/* declare class */
/**
  *\class   Scatter_GeomRender_params
  *\ingroup Scatter
  *\brief   base class of the full params of the scatters defined in this application
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
class Scatter_GeomRender_Params : public Scatter_Params
{
public:
//	ScatterType type;
};

/* declare class */
/**
  *\class   Scatter_GeomRender
  *\ingroup Scatter
  *\brief   base class of all scatters defined in this application
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
class Scatter_GeomRender : public Scatter
{
  protected:
	Scatter_GeomRender_Params *fullParamsPtr;
	Scatter_GeomRender_ReducedParams reducedParams;
		
  public:
    bool update;
    
	/* standard constructor */
    Scatter_GeomRender()
	{
		fullParamsPtr=NULL;
		path_to_ptx=(char*)malloc(512*sizeof(char));
	}
	/* destructor */
	virtual ~Scatter_GeomRender()
	{
		if (fullParamsPtr != NULL)
			delete fullParamsPtr;
		if (path_to_ptx != NULL)
			delete path_to_ptx;
	}
	virtual void hit(rayStruct &ray, Mat_hitParams hitParams);
	virtual void hit(diffRayStruct &ray, Mat_hitParams hitParams);
	virtual void hit(gaussBeamRayStruct &ray, gaussBeam_geometricNormal normal);
	virtual ScatterError setFullParams(Scatter_GeomRender_Params* ptrIn);
	virtual Scatter_GeomRender_Params* getFullParams(void);
	virtual ScatterError setReducedParams(Scatter_GeomRender_ReducedParams* ptrIn);
	virtual Scatter_GeomRender_ReducedParams* getReducedParams(void);
};

#endif

