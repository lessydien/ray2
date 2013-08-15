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

#include "MyDelegate.h"


MyDelegate::MyDelegate(QObject *parent) :
	QItemDelegate(parent)
{
}
//
//
//MyDelegate::~MyDelegate(void)
//{
//}

QWidget *MyDelegate::createEditor ( QWidget * parent, const QStyleOptionViewItem & option, const QModelIndex & index ) const
{
	QSpinBox *editor = new QSpinBox(parent);
	editor->setMinimum(0);
	editor->setMaximum(100);
	return editor;
}

void MyDelegate::setEditorData ( QWidget * editor, const QModelIndex & index ) const
{
	// get data out of model
	int value = index.model()->data(index,Qt::EditRole).toInt();

	QSpinBox *spinbox = static_cast<QSpinBox*>(editor);
	spinbox->setValue(value);
}

void MyDelegate::setModelData ( QWidget * editor, QAbstractItemModel * model, const QModelIndex & index ) const
{
	QSpinBox *spinbox = static_cast<QSpinBox*>(editor);
	spinbox->interpretText();
	int value = spinbox->value();
	model->setData(index,value,Qt::EditRole);

}

void MyDelegate::updateEditorGeometry ( QWidget * editor, const QStyleOptionViewItem & option, const QModelIndex & index ) const
{
	editor->setGeometry(option.rect);
}
