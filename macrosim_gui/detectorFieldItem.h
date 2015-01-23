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

#ifndef DETECTORFIELDITEM
#define DETECTORFIELDITEM

#include <qicon.h>

#include "QPropertyEditor/CustomTypes.h"
#include "DetectorItem.h"

using namespace macrosim;

namespace macrosim 
{

/** @class DetectorFieldItem
*   @brief class for visualizing objects from MacroSim
*   
*   The AddInModel supplies a widget showing the available objects from MacroSim with their name, filename, version and so on.
*/
class DetectorFieldItem :
	public DetectorItem
{
	Q_OBJECT

	Q_PROPERTY(Vec2d detPixel READ getDetPixel WRITE setDetPixel DESIGNABLE true USER true);
	Q_PROPERTY(DetOutFormat outFormat READ getDetOutFormat WRITE setDetOutFormat DESIGNABLE true USER true);

public:

	DetectorFieldItem(QString name="DetField", QObject *parent=0);
	~DetectorFieldItem(void);

	// functions for property editor
	const Vec2d getDetPixel()  {return m_detPixel;};
	void setDetPixel(const Vec2d in) {m_detPixel=in;};

	virtual bool writeToXML(QDomDocument &document, QDomElement &root) const;
	virtual bool readFromXML(const QDomElement &node);

private:

	DetOutFormat m_detOutFormat;
	Vec2d m_detPixel;
};


}; //namespace macrosim

#endif