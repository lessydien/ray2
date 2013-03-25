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

#ifndef DOCKWIDGETMacroSim_H
#define DOCKWIDGETMacroSim_H

#include "common/sharedStructures.h"

#include <QtGui>
#include <qwidget.h>
#include <qmap.h>
#include <qstring.h>

#include "ui_dockWidgetMacroSim.h"

class DockWidgetMacroSim : public QWidget
{
    Q_OBJECT

    public:
        DockWidgetMacroSim(QMap<QString, ito::Param> params, int uniqueID);
        ~DockWidgetMacroSim() {};

    private:
        Ui::DockWidgetMacroSim ui;

	signals:
		void dataPropertiesChanged(int sizex, int sizey, int bpp);

	public slots:
		void valuesChanged(QMap<QString, ito::Param> params);


    private slots:
		void on_spinWidth_valueChanged(int i);
		void on_spinHeight_valueChanged(int i);
		void on_spinBpp_valueChanged(int i);
};

#endif
