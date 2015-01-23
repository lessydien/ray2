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

#ifndef DETECTORRAYDATA
#define DETECTORRAYDATA

#include <qicon.h>

#include "QPropertyEditor/CustomTypes.h"
#include "DetectorItem.h"

using namespace macrosim;

namespace macrosim 
{

/** @class DetectorRayDataItem
*   @brief gui class for detectorRayData from MacroSim
*   
*   The AddInModel supplies a widget showing the available objects from MacroSim with their name, filename, version and so on.
*/
class DetectorRayDataItem :
	public DetectorItem
{
	Q_OBJECT

	Q_PROPERTY(bool listAllRays READ getlistAllRays WRITE setListAllRays DESIGNABLE true USER true);
	Q_PROPERTY(bool reduceData READ getReduceData WRITE setReduceData DESIGNABLE true USER true);

public:

	DetectorRayDataItem(QString name="DetRayData", QObject *parent=0);
	~DetectorRayDataItem(void);

	// functions for property editor
	const bool getlistAllRays() {return m_listAllRays;};
	void setListAllRays(bool in) {m_listAllRays=in;}; 
	const bool getReduceData() {return m_reduceData;};
	void setReduceData(bool in) {m_reduceData=in;}; 


	virtual bool writeToXML(QDomDocument &document, QDomElement &root) const;
	virtual bool readFromXML(const QDomElement &node);

private:
	bool m_listAllRays;
	bool m_reduceData;

};

}; //namespace macrosim

#endif