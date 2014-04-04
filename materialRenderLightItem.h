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

#ifndef MATERIALRENDERLIGHTITEM
#define MATERIALRENDERLIGHTITEM

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
class MaterialRenderLightItem :
	public MaterialItem
{
	Q_OBJECT

    Q_PROPERTY(double power READ getPower WRITE setPower DESIGNABLE true USER true);
    Q_PROPERTY(Vec3d pupilRoot READ getPupilRoot WRITE setPupilRoot DESIGNABLE true USER true);
    Q_PROPERTY(Vec3d pupilTilt READ getPupilTilt WRITE setPupilTilt DESIGNABLE true USER true);
    Q_PROPERTY(Vec2d pupilAptRad READ getPupilAptRad WRITE setPupilAptRad DESIGNABLE true USER true);

public:
	
	MaterialRenderLightItem(double power=1, QString name="MaterialRenderLight", QObject *parent=0);
	~MaterialRenderLightItem(void);

	// functions for property editor
    void setPower(const double in) {m_power=in; emit itemChanged(m_index, m_index);};
    double getPower() const {return m_power;}
    void setPupilRoot(const Vec3d in) {m_pupilRoot=in; emit itemChanged(m_index, m_index);};
    Vec3d getPupilRoot() const {return m_pupilRoot;}
    void setPupilTilt(const Vec3d in) {m_pupilTilt=in; emit itemChanged(m_index, m_index);};
    Vec3d getPupilTilt() const {return m_pupilTilt;}
    void setPupilAptRad(const Vec2d in) {m_pupilAptRad=in; emit itemChanged(m_index, m_index);};
    Vec2d getPupilAptRad() const {return m_pupilAptRad;}


	bool writeToXML(QDomDocument &document, QDomElement &root) const;
	bool readFromXML(const QDomElement &node);

	double getTransparency() {return 0.5;};


private:

	Vec3d m_pupilRoot;
	Vec3d m_pupilTilt;
    Vec2d m_pupilAptRad;
    double m_power;
};

}; //namespace macrosim

#endif