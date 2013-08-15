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

#include "dockWidgetMacroSim.h"

 DockWidgetMacroSim::DockWidgetMacroSim(QMap<QString, ito::Param> Params, int uniqueID)
 {
	ui.setupUi(this); 
    
    char* temp = Params["name"].getVal<char*>();
//	ui.lblName->setText(temp);
	ui.lblID->setText(QString::number(uniqueID));
    free(temp);

	valuesChanged(Params);
 }

 void DockWidgetMacroSim::valuesChanged(QMap<QString, ito::Param> Params)
 {
	ui.spinBpp->setMinimum(static_cast<int>(Params["bpp"].getMin()));
	ui.spinBpp->setMaximum(static_cast<int>(Params["bpp"].getMax()));
	ui.spinBpp->setValue(Params["bpp"].getVal<int>());

	ui.spinWidth->setMinimum(static_cast<int>(Params["sizex"].getMin()));
	ui.spinWidth->setMaximum(static_cast<int>(Params["sizex"].getMax()));
	ui.spinWidth->setValue(Params["sizex"].getVal<int>());

	ui.spinHeight->setMinimum(static_cast<int>(Params["sizey"].getMin()));
	ui.spinHeight->setMaximum(static_cast<int>(Params["sizey"].getMax()));
	ui.spinHeight->setValue(Params["sizey"].getVal<int>());
 }

void DockWidgetMacroSim::on_spinWidth_valueChanged(int i)
{
	emit dataPropertiesChanged( ui.spinWidth->value(), ui.spinHeight->value(), ui.spinBpp->value() );
}

void DockWidgetMacroSim::on_spinHeight_valueChanged(int i)
{
	emit dataPropertiesChanged( ui.spinWidth->value(), ui.spinHeight->value(), ui.spinBpp->value() );
}

void DockWidgetMacroSim::on_spinBpp_valueChanged(int i)
{
	emit dataPropertiesChanged( ui.spinWidth->value(), ui.spinHeight->value(), ui.spinBpp->value() );
}