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

/**\file CompoundGeometry.h
* \brief 
* 
*           
* \author Mauch
*/

/**
 *\defgroup CompoundGeometry
 */

#ifndef COMPOUNDGEOMETRY_H
  #define COMPOUNDGEOMETRY_H

#include "Geometry.h"
#include <optix.h>
#include "Material.h"
#include "DetectorParams.h"
#include "macrosim_functions.h"
#include "rayData.h"
#include "stdio.h"
#include "sampleConfig.h"
#include "pugixml.hpp"
#include <vector>

using namespace std;


/* declare class */
/**
  *\class   CompoundGeometry
  *\ingroup CompoundGeometry
  *\brief   base class of all geometries
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
class CompoundGeometry : public Geometry
{
  protected:

  private:
	

  public:
    /* standard constructor */
    CompoundGeometry()
	{
	  materialList=NULL;
	  materialListLength=0;
	}
    /* Constructor in case, list length is known */
    CompoundGeometry(int length)
	{
	  materialList=new Material*[length];
	  // init the pointers to zero
	  int i=0;
	  for (i=0;i<length;i++)
	  {
		  materialList[i]=NULL;
	  }
	  materialListLength = length;
	}
	/* Destruktor */
	virtual ~CompoundGeometry()
	{
	  if ( materialList != NULL)
	  {
		  int i;
		  for (i=0;i<materialListLength;i++)
		  {
			  if (materialList[i]!=NULL)
			  {
				  delete materialList[i];
			  }
		  }
		  materialList = NULL;
	  }
	}

	virtual geometryError parseXml(pugi::xml_node &geometry, vector<Geometry*> &geomVec);
};


#endif

