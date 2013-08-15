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

#include "scatterTorranceSparrow1DItem.h"

using namespace macrosim;

ScatterTorrSparr1DItem::ScatterTorrSparr1DItem(QString name, QObject *parent) :
	ScatterItem(TORRSPARR1D, name, parent)
{
}


ScatterTorrSparr1DItem::~ScatterTorrSparr1DItem()
{
}

bool ScatterTorrSparr1DItem::writeToXML(QDomDocument &document, QDomElement &root) const
{
	QDomElement scatter = document.createElement("scatter");

	// call base class function
	if (!ScatterItem::writeToXML(document, scatter))
		return false;

	scatter.setAttribute("scatterType", "TORRSPARR2D");
	scatter.setAttribute("kDl", QString::number(m_kDl));
	scatter.setAttribute("kSl", QString::number(m_kSp));
	scatter.setAttribute("kSp", QString::number(m_kSp));
	scatter.setAttribute("sigmaSp", QString::number(m_sigmaSp));
	scatter.setAttribute("sigmaSl", QString::number(m_sigmaSl));
	scatter.setAttribute("scatAxis.x", QString::number(m_scatAxis.X));
	scatter.setAttribute("scatAxis.y", QString::number(m_scatAxis.Y));
	scatter.setAttribute("scatAxis.z", QString::number(m_scatAxis.Z));

	root.appendChild(scatter);
	return true;
}
