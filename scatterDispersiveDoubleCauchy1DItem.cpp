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

#include "scatterDispersiveDoubleCauchy1DItem.h"

using namespace macrosim;

ScatterDispersiveDoubleCauchy1DItem::ScatterDispersiveDoubleCauchy1DItem(QString name, QObject *parent) :
	ScatterItem(DISPDOUBLECAUCHY1D, name, parent)
{
}


ScatterDispersiveDoubleCauchy1DItem::~ScatterDispersiveDoubleCauchy1DItem()
{
}

bool ScatterDispersiveDoubleCauchy1DItem::writeToXML(QDomDocument &document, QDomElement &root) const
{
	QDomElement scatter = document.createElement("scatter");

	// call base class function
	if (!ScatterItem::writeToXML(document, scatter))
		return false;

	scatter.setAttribute("scatterType", "DISPDOUBLECAUCHY1D");
	scatter.setAttribute("aGammaSl", QString::number(m_aGammaSl));
	scatter.setAttribute("aGammaSp", QString::number(m_aGammaSp));
	scatter.setAttribute("cGammaSl", QString::number(m_cGammaSl));
	scatter.setAttribute("cGammaSp", QString::number(m_cGammaSp));
	scatter.setAttribute("aKSl", QString::number(m_aKSl));
	scatter.setAttribute("aKSp", QString::number(m_aKSp));
	scatter.setAttribute("cKSl", QString::number(m_cKSl));
	scatter.setAttribute("cKSp", QString::number(m_cKSp));
	scatter.setAttribute("scatAxis.x", QString::number(m_scatAxis.X));
	scatter.setAttribute("scatAxis.y", QString::number(m_scatAxis.Y));
	scatter.setAttribute("scatAxis.z", QString::number(m_scatAxis.Z));

	root.appendChild(scatter);
	return true;
}
