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

#ifndef MATERIALITEMLIB
#define MATERIALITEMLIB

#include "materialItem.h"
#include "materialAbsorbingItem.h"
#include "materialDiffractingItem.h"
#include "materialFilterItem.h"
#include "materialIdealLenseItem.h"
#include "materialLinearGrating1DItem.h"
#include "materialPathTraceSourceItem.h"
#include "materialReflectingCovGlassItem.h"
#include "materialReflectingItem.h"
#include "materialRefractingItem.h"
#include "materialDOEItem.h"
#include "materialVolumeScatterItem.h"
#include "materialVolumeAbsorbingItem.h"
//#include "materialAbsorbingItem.h"
//#include "geometryItem.h"

namespace macrosim 
{

/** @class MaterialItemLib
*   @brief class for visualizing objects from MacroSim
*   
*   The AddInModel supplies a widget showing the available objects from MacroSim with their name, filename, version and so on.
*/
class MaterialItemLib 
{

public:

	MaterialItemLib(void) {};
	~MaterialItemLib(void) {};

	MaterialItem* MaterialItemLib::createMaterial(MaterialItem::MaterialType type);

	MaterialItem::MaterialType stringToMaterialType(const QString str) const;
	QString materialTypeToString(const MaterialItem::MaterialType type) const;
	MaterialItem::Mat_ScatterType stringToMatScatterType(const QString str) const;
	QString matScatterTypeToString(const MaterialItem::Mat_ScatterType type) const;
	MaterialItem::Mat_CoatingType stringToMatCoatingType(const QString str) const;
	QString matCoatingTypeToString(const MaterialItem::Mat_CoatingType type) const;

	MaterialItem::MaterialType abstractMatTypeToMatType(const AbstractItem::Abstract_MaterialType type) const;
	AbstractItem::Abstract_MaterialType matTypeToAbstractMatType(const MaterialItem::MaterialType type) const;


private:

};

}; //namespace macrosim

#endif