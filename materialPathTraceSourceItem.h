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

#ifndef MATERIALPATHTRACESOURCEITEM
#define MATERIALPATHTRACESOURCEITEM

#include <qicon.h>

#include "QPropertyEditor/CustomTypes.h"
#include "AbstractItem.h"
#include "GeometryItem.h"
#include "MaterialItem.h"


namespace macrosim 
{

/** @class MaterialItem
*   @brief class for visualizing objects from MacroSim
*   
*   The AddInModel supplies a widget showing the available objects from MacroSim with their name, filename, version and so on.
*/
class MaterialPathTraceSourceItem :
	public MaterialItem
{
	Q_OBJECT

	Q_PROPERTY(Vec2d acceptanceAngleMin READ getAcceptanceAngleMin WRITE setAcceptanceAngleMin DESIGNABLE true USER true);
	Q_PROPERTY(Vec2d acceptanceAngleMax READ getAcceptanceAngleMax WRITE setAcceptanceAngleMax DESIGNABLE true USER true);
	Q_PROPERTY(double flux READ getFlux WRITE setFlux DESIGNABLE true USER true);

public:
	
	MaterialPathTraceSourceItem(Vec2d acceptanceAngleMax=Vec2d(0.0,0.0), Vec2d=Vec2d(0.0,0.0), double flux=0,  QString name="MaterialPathTraceSource", QObject *parent=0);
	~MaterialPathTraceSourceItem(void);

	// functions for property editor
	void setAcceptanceAngleMax(const Vec2d in) {m_acceptanceAngleMax=in; emit itemChanged(m_index, m_index);};
	Vec2d getAcceptanceAngleMax() const {return m_acceptanceAngleMax;};
	void setAcceptanceAngleMin(const Vec2d in) {m_acceptanceAngleMin=in; emit itemChanged(m_index, m_index);};
	Vec2d getAcceptanceAngleMin() const {return m_acceptanceAngleMin;};

	void setFlux(const double in) {m_flux=in; emit itemChanged(m_index, m_index);};
	double getFlux() const {return m_flux;};

	bool writeToXML(QDomDocument &document, QDomElement &root) const;
	bool readFromXML(const QDomElement &node);


private:

	Vec2d m_acceptanceAngleMax;
	Vec2d m_acceptanceAngleMin;
	double m_flux;
};

}; //namespace macrosim

#endif