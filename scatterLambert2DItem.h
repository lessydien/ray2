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

#ifndef SCATTERLAMBERT2DITEM
#define SCATTERLAMBERT2DITEM

#include <qicon.h>

#include "ScatterItem.h"

#include "QPropertyEditor/CustomTypes.h"


namespace macrosim 
{

/** @class ScatterItem
*   @brief class for visualizing objects from MacroSim
*   
*   The AddInModel supplies a widget showing the available objects from MacroSim with their name, filename, version and so on.
*/
class ScatterLambert2DItem :
	public ScatterItem
{
	Q_OBJECT

	Q_PROPERTY(double m_tis READ getTotalIntegratedScatter WRITE setTotalIntegratedScatter DESIGNABLE true USER true);


public:

	ScatterLambert2DItem(double tis=1, QString name="Lambert2D", QObject *parent=0);
	~ScatterLambert2DItem(void);

	// functions for property editor
	double getTotalIntegratedScatter() const {return m_Tis;};
	void setTotalIntegratedScatter(const double tis) {m_Tis=tis; emit itemChanged(m_index, m_index);};

	bool writeToXML(QDomDocument &document, QDomElement &root) const ;
    bool readFromXML(const QDomElement &node);

private:

	double m_Tis;
};

}; //namespace macrosim

#endif