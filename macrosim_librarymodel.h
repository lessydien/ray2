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


namespace macrosim
{

class LibraryModel : public QAbstractItemModel
{
	Q_OBJECT

public:
	LibraryModel(const QStringList &strings, QObject *parent);
	~LibraryModel();

	// functions we need to implement
	int columnCount ( const QModelIndex &parent  ) const;
	QVariant data ( const QModelIndex &index, int role ) const;
	QModelIndex index ( int row, int column, const QModelIndex &parent ) const;
//	QModelIndex	parent ( const QModelIndex & index ) const;
	int	rowCount ( const QModelIndex &parent ) const;

    Qt::ItemFlags flags(const QModelIndex &index) const;
    QVariant headerData(int section, Qt::Orientation orientation, int role = Qt::DisplayRole) const;
//    int update(void) { emit(beginResetModel()); emit(endResetModel()); return 0; };
	void clearModel(void) {m_data.clear();};

protected:

private:
    QList<QString> m_headers;									//!<  string list of names of column headers
    QList<QVariant> m_alignment;								//!<  list of alignments for the corresponding headers
	QList<macrosim::AbstractItem*> m_data;		//!<  list containing the individual model items
	
};

}; // end namespace macrosim
#endif // MACROSIMLIBMODEL_H
