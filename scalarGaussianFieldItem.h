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

#ifndef SCALARGAUSSIANFIELDITEM
#define SCALARGAUSSIANFIELDITEM

#include <qicon.h>

#include "QPropertyEditor/CustomTypes.h"
#include "scalarFieldItem.h"

using namespace macrosim;

namespace macrosim 
{

/** @class FieldItem
*   @brief class for visualizing objects from MacroSim
*   
*   The AddInModel supplies a widget showing the available objects from MacroSim with their name, filename, version and so on.
*/
class ScalarGaussianFieldItem :
	public ScalarFieldItem
{
	Q_OBJECT

	Q_PROPERTY(Vec2d focusWidth READ getFocusWidth WRITE setFocusWidth DESIGNABLE true USER true);
	Q_PROPERTY(Vec2d distToFocus READ getDistToFocus WRITE setDistToFocus DESIGNABLE true USER true);

public:

	ScalarGaussianFieldItem(QString name="ScalarGaussianField", QObject *parent=0);
	~ScalarGaussianFieldItem(void);

	// functions for property editor
	Vec2d getFocusWidth() const {return m_focusWidth;};
	void setFocusWidth(Vec2d in) {m_focusWidth=in;};
	Vec2d getDistToFocus() const {return m_distToFocus;};
	void setDistToFocus(Vec2d in) {m_distToFocus=in;};

	bool signalDataChanged();

	bool writeToXML(QDomDocument &document, QDomElement &root) const;
	bool readFromXML(const QDomElement &node);

private:
	Vec2d m_focusWidth;
	Vec2d m_distToFocus;

}; // class RayFieldItem

}; //namespace macrosim

#endif