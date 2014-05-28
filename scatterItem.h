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

#ifndef SCATTERITEM
#define SCATTERITEM

#include <qicon.h>

#include "AbstractItem.h"

#include "QPropertyEditor/CustomTypes.h"


namespace macrosim 
{

/** @class ScatterItem
*   @brief class for visualizing objects from MacroSim
*   
*   The AddInModel supplies a widget showing the available objects from MacroSim with their name, filename, version and so on.
*/
class ScatterItem :
	public AbstractItem
{
	Q_OBJECT

	Q_ENUMS(ScatterType);
    Q_ENUMS(ScatterPupilType);

	Q_PROPERTY(ScatterType ScatType READ getScatterType DESIGNABLE true USER true);
	Q_PROPERTY(Mat_ScatterType ScatType DESIGNABLE true USER true); // overwrite scatterType-Property of materialItem, so it can not be changed in propertyEditor of this item
	Q_PROPERTY(Abstract_MaterialType materialType DESIGNABLE true USER true); // overwrite materialType-Property of abstractItem, so it can not be changed in propertyEditor of this item
    Q_PROPERTY(Vec3d PupilRoot WRITE setPupRoot READ getPupRoot DESIGNABLE true USER true);
    Q_PROPERTY(Vec3d PupilTilt WRITE setPupTilt READ getPupTilt DESIGNABLE true USER true);
    Q_PROPERTY(Vec2d PupilApertureRadius WRITE setPupAptRad READ getPupAptRad DESIGNABLE true USER true);
    Q_PROPERTY(ScatterPupilType PupilType WRITE setPupAptType READ getPupAptType DESIGNABLE true USER true);	


public:
	enum ScatterType {NOSCATTER, LAMBERT2D, TORRSPARR1D, TORRSPARR2D, TORRSPARR2DPATHTRACE, DISPDOUBLECAUCHY1D, DOUBLECAUCHY1D};
    enum ScatterPupilType {NOPUPIL, RECTPUPIL, ELLIPTPUPIL};

	ScatterItem(ScatterType type=NOSCATTER, QString name="base scatter", QObject *parent=0);
	~ScatterItem(void);

	// functions for property editor
	ScatterType getScatterType() const {return m_scatterType;};
	void setScatterType(const ScatterType type) {m_scatterType=type; emit itemChanged(m_index, m_index);};
    Vec3d getPupRoot() const {return m_pupRoot;};
    void setPupRoot(const Vec3d in) {m_pupRoot=in;};
    Vec3d getPupTilt() const {return m_pupTilt;};
    void setPupTilt(const Vec3d in) {m_pupTilt=in;};
    Vec2d getPupAptRad() const {return m_pupAptRad;};
    void setPupAptRad(const Vec2d in) {m_pupAptRad=in;};
    ScatterPupilType getPupAptType() const {return m_pupAptType;};
    void setPupAptType(const ScatterPupilType in) {m_pupAptType=in;};

	bool writeToXML(QDomDocument &document, QDomElement &root) const;
	bool readFromXML(const QDomElement &node);

private:

	ScatterType m_scatterType;
    Vec3d m_pupRoot;
    Vec3d m_pupTilt;
    Vec2d m_pupAptRad;
    ScatterPupilType m_pupAptType;


signals:
	void itemChanged(const QModelIndex &topLeft, const QModelIndex &bottomRight);

public slots:
	void changeItem(const QModelIndex &topLeft, const QModelIndex &bottomRight);

};

}; //namespace macrosim

#endif