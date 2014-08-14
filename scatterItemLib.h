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

#ifndef SCATTERITEMLIB
#define SCATTERITEMLIB

#include "scatterPhongItem.h"
#include "scatterLambert2DItem.h"
#include "scatterDispersiveDoubleCauchy1DItem.h"
#include "scatterDoubleCauchy1DItem.h"
#include "scatterNoItem.h"
#include "scatterTorranceSparrow1DItem.h"
#include "scatterTorranceSparrow2DItem.h"
#include "scatterTorranceSparrow2DPathTraceItem.h"
#include "materialItem.h"

namespace macrosim 
{

/** @class ScatterItemLib
*   @brief class for visualizing objects from MacroSim
*   
*   The AddInModel supplies a widget showing the available objects from MacroSim with their name, filename, version and so on.
*/
class ScatterItemLib 
{

public:
	ScatterItemLib() {};
	~ScatterItemLib() {};
	ScatterItem* createScatter(ScatterItem::ScatterType type);

	QString scatterTypeToString(const ScatterItem::ScatterType type) const;
	ScatterItem::ScatterType stringToScatterType(const QString str) const;

    QString scatterPupilTypeToString(const ScatterItem::ScatterPupilType type) const;
    ScatterItem::ScatterPupilType stringToScatterPupilType(const QString) const;

	ScatterItem::ScatterType matScatTypeToScatType(const MaterialItem::Mat_ScatterType type) const;
	MaterialItem::Mat_ScatterType scatTypeToMatScatType(const ScatterItem::ScatterType type) const;


private:

};

}; //namespace macrosim

#endif