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

#ifndef MYDELEGATE_H
#define MYDELEGATE_H

#include <QItemDelegate>
#include <QModelIndex>
#include <QObject>
#include <QtCore/QSize>
#include <QSpinBox>

class MyDelegate :
	public QItemDelegate
{
	Q_OBJECT

public:
	explicit MyDelegate(QObject *parent = 0);

	QWidget *createEditor ( QWidget * parent, const QStyleOptionViewItem & option, const QModelIndex & index ) const;
	void	setEditorData ( QWidget * editor, const QModelIndex & index ) const;
	void	setModelData ( QWidget * editor, QAbstractItemModel * model, const QModelIndex & index ) const;
	void	updateEditorGeometry ( QWidget * editor, const QStyleOptionViewItem & option, const QModelIndex & index ) const;

signals:

public slots:

	//MyDelegate(void);
	//~MyDelegate(void);
};

#endif
