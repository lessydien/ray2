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

/**\file SphericalSurface_GeomRender.h
* \brief 
* 
*           
* \author Mauch
*/

#ifndef SPHERICALSURFACE_GEOMRENDER_H
  #define SPHERICALSURFACE_GEOMRENDER_H
  
/* include header of basis class */
#include "../SphericalSurface.h"
#include "SphericalSurface_GeomRender_intersect.h"
#include <optix.h>

/* declare class */
/**
  *\class   SphericalSurface_GeomRender
  *\ingroup Geometry
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
class SphericalSurface_GeomRender : public SphericalSurface
{
  protected:

  private:

  public:
    /* standard constructor */
    SphericalSurface_GeomRender() :
        SphericalSurface()
	{
	  sprintf( this->path_to_ptx_intersect, "%s" PATH_SEPARATOR "%s", PATH_TO_PTX, "macrosim_tracer_generated_SphericalSurface_GeomRender.cu.ptx" );
	  sprintf( this->path_to_ptx_boundingBox, "%s" PATH_SEPARATOR "%s", PATH_TO_PTX, "macrosim_tracer_generated_SphericalSurface_GeomRender.cu.ptx" );
	}
    /* Constructor with known list length */
    SphericalSurface_GeomRender(int length) :
        SphericalSurface(length)
	{
	  sprintf( this->path_to_ptx_intersect, "%s" PATH_SEPARATOR "%s", PATH_TO_PTX, "macrosim_tracer_generated_SphericalSurface_GeomRender.cu.ptx" );
	  sprintf( this->path_to_ptx_boundingBox, "%s" PATH_SEPARATOR "%s", PATH_TO_PTX, "macrosim_tracer_generated_SphericalSurface_GeomRender.cu.ptx" );
	}
	/* Destruktor */
	~SphericalSurface_GeomRender()
	{
	  if (materialList!=NULL)
	  {
		  // delete all the materials attached to this geometry
		  int i;
		  for (i=0;i<materialListLength;i++)
		  {
			  if (materialList[i]!=NULL)
			  {
				  delete materialList[i];
				  materialList[i]=NULL;
			  }
		  }
		  delete materialList;
		  materialList = NULL;
	  }
	  if (paramsPtr != NULL)
	  {
		delete paramsPtr;
		paramsPtr=NULL;
	  }
	  if (reducedParamsPtr != NULL)
	  {
	    delete reducedParamsPtr;
		reducedParamsPtr = NULL;
	  }
	}

	geometryError createOptixInstance( RTcontext &context, RTgeometrygroup &geometrygroup, int index, SimParams simParams, double lambda );
	geometryError updateOptixInstance( RTcontext &context, RTgeometrygroup &geometrygroup, int index, SimParams simParams, double lambda );

};

#endif
