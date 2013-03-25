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

#include "macrosim_librarymodel.h"
#include "abstractItem.h"

using namespace macrosim;
//----------------------------------------------------------------------------------------------------------------------------------
/** constructor
*
*   contructor, creating column headers for the tree view
*/
LibraryModel::LibraryModel(const QStringList &strings, QObject *parent=0)
	: QAbstractItemModel(parent)
{
	m_headers << tr("Name");
	m_alignment << QVariant(Qt::AlignLeft);
}

LibraryModel::~LibraryModel()
{
   m_headers.clear();
   m_alignment.clear();
   m_data.clear();
   return;
}

int LibraryModel::columnCount ( const QModelIndex &parent ) const
{
	return m_headers.size();
}

QVariant LibraryModel::data ( const QModelIndex &index, int role ) const
{
	if (!index.isValid())
		return QVariant();
	else
	{
		AbstractItem* item = reinterpret_cast<AbstractItem*>(index.internalPointer());
		if (role == Qt::DisplayRole)
		{
			switch(index.column())
			{
			case 1:
				switch (item->getObjectType())
				{
				case AbstractItem::GEOMETRYGROUP:
					return tr("GeometryGroup");
					break;
				case AbstractItem::GEOMETRY:
					return tr("Geometry");
					break;
				case AbstractItem::MATERIAL:
					return tr("Material");
					break;
				case AbstractItem::COATING:
					return tr("Coating");
					break;
				case AbstractItem::SCATTER:
					return tr("Scatter");
					break;
				default:
					return QVariant();
					break;
				}
				break;
			case 0:
				return item->getName();
				break;
			default:
				return QVariant();
				break;
			}
		}
	}
	return QVariant();
}

int	LibraryModel::rowCount ( const QModelIndex &parent ) const
{
	return m_data.count();
}

QModelIndex LibraryModel::index ( int row, int column, const QModelIndex &parent ) const
{
	if(parent.isValid()) //we are not in the root level
	{
		AbstractItem* parentItem = reinterpret_cast<AbstractItem*>(parent.internalPointer());
		if(parentItem != NULL)
		{
			if (row<0 || row>=parentItem->getNumberOfChilds())
				return QModelIndex();
			return createIndex(row, column, reinterpret_cast<void*>(parentItem->getChild(row)));
			//return createIndex(row, column, reinterpret_cast<void*>(&parentItem));
		}
		else
		{
			return QModelIndex();
		}
	}
	else //root level
	{
		if(row < 0 || row >= m_data.size())
		{
			return QModelIndex();
		}
		else
		{
			return createIndex(row, column, reinterpret_cast<void*>(m_data[row]));
		}
	}
}

//----------------------------------------------------------------------------------------------------------------------------------
/** return the header / captions for the tree view model
*
*/
QVariant LibraryModel::headerData(int section, Qt::Orientation orientation, int role) const
{
	if (role != Qt::DisplayRole)
		return QVariant();
	if (orientation == Qt::Horizontal)
		return QString("Object Type").arg(section);
	else
		return QString("Row %1").arg(section);
}

//----------------------------------------------------------------------------------------------------------------------------------
Qt::ItemFlags LibraryModel::flags(const QModelIndex &index) const
{
    if (!index.isValid())
        return 0;
	return Qt::ItemIsEnabled | Qt::ItemIsSelectable && !Qt::ItemIsEditable;
}
