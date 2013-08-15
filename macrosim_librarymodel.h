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

#ifndef MACROSIMLIBMODEL_H
#define MACROSIMLIBMODEL_H

#include <qlist.h>
#include "abstractItem.h"
#include <qabstractitemmodel.h>
#include "common/sharedStructures.h"
#include "macrosim_scenemodel.h"


namespace macrosim
{

class LibraryModel : public SceneModel
{
	Q_OBJECT

public:
	LibraryModel(QObject *parent);
	~LibraryModel();

	void appendItem(macrosim::AbstractItem* item, vtkSmartPointer<vtkRenderer> renderer, QModelIndex parentIndex, int rowIn);

};

}; // end namespace macrosim
#endif // MACROSIMLIBMODEL_H
