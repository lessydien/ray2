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

#include "scatterDoubleCauchy1DItem.h"

using namespace macrosim;

ScatterDoubleCauchy1DItem::ScatterDoubleCauchy1DItem(QString name, QObject *parent) :
	ScatterItem(DOUBLECAUCHY1D, name, parent)
{
}


ScatterDoubleCauchy1DItem::~ScatterDoubleCauchy1DItem()
{
}

bool ScatterDoubleCauchy1DItem::writeToXML(QDomDocument &document, QDomElement &root) const
{
	QDomElement scatter = document.createElement("scatter");

	// call base class function
	if (!ScatterItem::writeToXML(document, scatter))
		return false;

	scatter.setAttribute("scatterType", "DOUBLECAUCHY1D");
	scatter.setAttribute("gammaSl", QString::number(m_gammaSl));
	scatter.setAttribute("gammaSp", QString::number(m_gammaSp));
	scatter.setAttribute("kSl", QString::number(m_kSl));
	scatter.setAttribute("kSp", QString::number(m_kSp));
	scatter.setAttribute("scatAxis.x", QString::number(m_scatAxis.X));
	scatter.setAttribute("scatAxis.y", QString::number(m_scatAxis.Y));
	scatter.setAttribute("scatAxis.z", QString::number(m_scatAxis.Z));

	root.appendChild(scatter);
	return true;
}
