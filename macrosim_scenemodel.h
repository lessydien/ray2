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

#ifndef MACRROSIM_SCENEMODEL_H
#define MACRROSIM_SCENEMODEL_H

#include <qabstractitemmodel.h>
#include <qlist.h>
#include "common/sharedStructures.h"
#include "abstractItem.h"
#include "detectorItem.h"

#include <vtkSmartPointer.h>
#include <vtkRenderer.h>
//#include <QtOpenGL\qglfunctions.h>

namespace macrosim
{

class SceneModel : public QAbstractItemModel
{
	Q_OBJECT

public:
	SceneModel(QObject *parent);
	~SceneModel();

	// functions we need to implement
	int columnCount ( const QModelIndex &parent  ) const;
	int getNrOfItems () const {return m_data.size();};
	AbstractItem* getItem(int index) const {return m_data.at(index);};
	QVariant data ( const QModelIndex &index, int role ) const;
	QModelIndex index ( int row, int column, const QModelIndex &parent ) const;
	QModelIndex	parent ( const QModelIndex & index ) const;
	int	rowCount ( const QModelIndex &parent ) const;

    Qt::ItemFlags flags(const QModelIndex &index) const;
    QVariant headerData(int section, Qt::Orientation orientation, int role = Qt::DisplayRole) const;
//    int update(void) { emit(beginResetModel()); emit(endResetModel()); return 0; };
	virtual void appendItem(macrosim::AbstractItem* item, vtkSmartPointer<vtkRenderer> renderer, QModelIndex parentIndex=QModelIndex(), int rowIn=0) ;
	void removeItem(QModelIndex index);
	void removeFocusedItem();
	void clearModel(void);
//	AbstractItem* getItem(int index) {return m_data.at(index);};
	AbstractItem* getItem(const QModelIndex &index) const;
	DetectorItem* getDetectorItem() const;
	void signalDataChange(const QModelIndex &topLeft, const QModelIndex &bottomRight);
	QModelIndex getRootIndex(const QModelIndex &index);
	QModelIndex getBaseIndex(const QModelIndex &index);
	QModelIndex getFocusedItemIndex() const {return m_focusedItemIndex;};
	bool removeRows(int row, int count, const QModelIndex & parentInd = QModelIndex() );
	void render(QMatrix4x4 &matrix, RenderOptions &options);
	void renderVtk(vtkSmartPointer<vtkRenderer> renderer);
	void updateVtk();
	void setRenderOptions(RenderOptions options);


protected:
	QModelIndex getItemWithSpecificChild (AbstractItem* rootItem, const AbstractItem* child) const;
//	QModelIndex test() const { return QModelIndex(); };
	QList<macrosim::AbstractItem*> m_data;						//!<  list containing the individual model items

private:
    QList<QString> m_headers;									//!<  string list of names of column headers
    QList<QVariant> m_alignment;								//!<  list of alignments for the corresponding headers
	
	int m_geomID;
	int m_geomGroupNr;
//	QMap<QString, ito::tParam> m_data;
	RenderOptions m_renderOptions;

	QModelIndex m_focusedItemIndex;
	QModelIndex m_currentGeomGroupIndex;

signals:
	void itemFocusChanged(const QModelIndex &topLeft);
//	void itemDataChanged(const QModelIndex &topLeft, const QModelIndex &bottomRight);

public slots:
	void changeItemFocus(const QModelIndex &topLeft);
	void changeItemData(const QModelIndex &topLeft, const QModelIndex &bottomRight);
	void exchangeItem(const QModelIndex &parentIndex, const int row, const int coloumn, macrosim::AbstractItem &item);
};

}; // end namespace macrosim
#endif // H
