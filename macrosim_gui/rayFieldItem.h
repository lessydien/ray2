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

#ifndef RAYFIELDITEM
#define RAYFIELDITEM

#include <qicon.h>

#include "QPropertyEditor/CustomTypes.h"
#include "fieldItem.h"

using namespace macrosim;

namespace macrosim 
{

/** @class FieldItem
*   @brief class for visualizing objects from MacroSim
*   
*   The AddInModel supplies a widget showing the available objects from MacroSim with their name, filename, version and so on.
*/
class RayFieldItem :
	public FieldItem
{
	Q_OBJECT

//	Q_PROPERTY(Vec3d root READ getRoot WRITE setRoot DESIGNABLE true USER true);
	Q_PROPERTY(Vec3d tilt READ getTilt WRITE setTilt DESIGNABLE true USER true);
	Q_PROPERTY(Vec3d rayDirection READ getRayDirection WRITE setRayDirection DESIGNABLE true USER true);
	Q_PROPERTY(double coherence READ getCoherence WRITE setCoherence DESIGNABLE true USER true);
	Q_PROPERTY(Vec2d alphaMax READ getAlphaMax WRITE setAlphaMax DESIGNABLE true USER true);
	Q_PROPERTY(Vec2d alphaMin READ getAlphaMin WRITE setAlphaMin DESIGNABLE true USER true);
	Q_PROPERTY(double power READ getPower WRITE setPower DESIGNABLE true USER true);
	Q_PROPERTY(unsigned long width READ getWidth WRITE setWidth DESIGNABLE true USER true);
	Q_PROPERTY(unsigned long height READ getHeight WRITE setHeight DESIGNABLE true USER true);
	Q_PROPERTY(unsigned long widthLayout READ getWidthLayout WRITE setWidthLayout DESIGNABLE true USER true);
	Q_PROPERTY(unsigned long heightLayout READ getHeightLayout WRITE setHeightLayout DESIGNABLE true USER true);
	Q_PROPERTY(RayDirDistrType rayDirDistrType READ getRayDirDistrType WRITE setRayDirDistrType DESIGNABLE true USER true);
	Q_PROPERTY(RayPosDistrType rayPosDistrType READ getRayPosDistrType WRITE setRayPosDistrType DESIGNABLE true USER true);

	Q_CLASSINFO("tilt", "decimals=10;");
	Q_CLASSINFO("rayDirection", "decimals=10;");
	Q_CLASSINFO("alphaMax", "decimals=10;");
	Q_CLASSINFO("alphaMin", "decimals=10;");

	Q_ENUMS(RayDirDistrType);
	Q_ENUMS(RayPosDistrType);

public:
	enum RayDirDistrType {RAYDIR_RAND_RECT, RAYDIR_RAND_RAD, RAYDIR_RANDNORM_RECT, RAYDIR_RANDIMPAREA, RAYDIR_UNIFORM, RAYDIR_GRID_RECT, RAYDIR_GRID_RECT_FARFIELD, RAYDIR_GRID_RAD, RAYDIR_UNKNOWN};
	enum RayPosDistrType {RAYPOS_RAND_RECT, RAYPOS_RAND_RECT_NORM, RAYPOS_GRID_RECT, RAYPOS_RAND_RAD, RAYPOS_RAND_RAD_NORM, RAYPOS_GRID_RAD, RAYPOS_UNKNOWN};

	RayFieldItem(QString name="name", FieldType type=UNDEFINED, QObject *parent=0);
	~RayFieldItem(void);

	// functions for property editor
	//Vec3d getRoot() const {return m_root;};
	//void setRoot(const Vec3d in) {m_root=in;};
	Vec3d getTilt() const {return m_tilt;};
	void setTilt(const Vec3d in) {m_tilt=in;};
	Vec3d getRayDirection() const {return m_rayDirection;};
	void setRayDirection(const Vec3d in) {m_rayDirection=in;};
	Vec2d getAlphaMax() const {return m_alphaMax;};
	void setAlphaMax(const Vec2d in) {m_alphaMax=in;};
	Vec2d getAlphaMin() const {return m_alphaMin;};
	void setAlphaMin(const Vec2d in) {m_alphaMin=in;};
	double getPower() const {return m_power;};
	void setPower(const double in) {m_power=in;};
	double getCoherence() const {return m_coherence;};
	void setCoherence(const double in) {m_coherence=in;};
	unsigned long getWidth() const {return m_width;};
	void setWidth(const unsigned long in) {m_width=in;};
	unsigned long getHeight() const {return m_height;};
	void setHeight(const unsigned long in) {m_height=in;};
	unsigned long getWidthLayout() const {return m_widthLayout;};
	void setWidthLayout(const unsigned long in) {m_widthLayout=in;};
	unsigned long getHeightLayout() const {return m_heightLayout;};
	void setHeightLayout(const unsigned long in) {m_heightLayout=in;};
	RayDirDistrType getRayDirDistrType() const {return m_rayDirDistrType;};
	void setRayDirDistrType(const RayDirDistrType in) {m_rayDirDistrType = in;};
	RayPosDistrType getRayPosDistrType() const {return m_rayPosDistrType;};
	void setRayPosDistrType(const RayPosDistrType in) {m_rayPosDistrType = in;};

	QString rayDirDistrTypeToString(const RayDirDistrType type) const;
	RayDirDistrType stringToRayDirDistrType(const QString str) const;
	QString rayPosDistrTypeToString(const RayPosDistrType type) const;
	RayPosDistrType stringToRayPosDistrType(const QString str) const;

	virtual bool signalDataChanged() {return true;};

	virtual bool writeToXML(QDomDocument &document, QDomElement &root) const;
	virtual bool readFromXML(const QDomElement &node);

protected:

//	Vec3d m_root;
	Vec3d m_tilt;
	Vec3d m_rayDirection;
	Vec2d m_alphaMax;
	Vec2d m_alphaMin;
	double m_coherence;
	double m_power;
	unsigned long m_width;
	unsigned long m_height;
	unsigned long m_widthLayout;
	unsigned long m_heightLayout;
	RayDirDistrType m_rayDirDistrType;
	RayPosDistrType m_rayPosDistrType;
}; // class RayFieldItem

}; //namespace macrosim

#endif