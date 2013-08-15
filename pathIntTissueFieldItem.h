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

#ifndef PATHINTTISSUEFIELDITEM
#define PATHINTTISSUEFIELDITEM

#include <qicon.h>

#include "QPropertyEditor/CustomTypes.h"
#include "rayFieldItem.h"

using namespace macrosim;

namespace macrosim 
{

/** @class PathIntTissueFieldItem
*   @brief class for visualizing objects from MacroSim
*   
*   The AddInModel supplies a widget showing the available objects from MacroSim with their name, filename, version and so on.
*/
class PathIntTissueFieldItem :
	public FieldItem
{
	Q_OBJECT

	Q_PROPERTY(Vec3d tilt READ getTilt WRITE setTilt DESIGNABLE true USER true);
	Q_PROPERTY(double power READ getPower WRITE setPower DESIGNABLE true USER true);
	Q_PROPERTY(unsigned long width READ getWidth WRITE setWidth DESIGNABLE true USER true);
	Q_PROPERTY(unsigned long height READ getHeight WRITE setHeight DESIGNABLE true USER true);
	Q_PROPERTY(unsigned long widthLayout READ getWidthLayout WRITE setWidthLayout DESIGNABLE true USER true);
	Q_PROPERTY(unsigned long heightLayout READ getHeightLayout WRITE setHeightLayout DESIGNABLE true USER true);
	Q_PROPERTY(Vec3d volumeWidth READ getVolumeWidth WRITE setVolumeWidth DESIGNABLE true USER true);
	Q_PROPERTY(Vec3d sourcePos READ getSourcePos WRITE setSourcePos DESIGNABLE true USER true);
	Q_PROPERTY(double meanFreePath READ getMeanFreePath WRITE setMeanFreePath DESIGNABLE true USER true);
	Q_PROPERTY(double anisotropy READ getAnisotropy WRITE setAnisotropy DESIGNABLE true USER true);


public:

	PathIntTissueFieldItem(QString name="PathIntTissueRayField", QObject *parent=0);
	~PathIntTissueFieldItem(void);

	// functions for property editor
	Vec3d getTilt() const {return m_tilt;};
	void setTilt(const Vec3d in) {m_tilt=in;};
	double getPower() const {return m_power;};
	void setPower(const double in) {m_power=in;};
	unsigned long getWidth() const {return m_width;};
	void setWidth(const unsigned long in) {m_width=in;};
	unsigned long getHeight() const {return m_height;};
	void setHeight(const unsigned long in) {m_height=in;};
	unsigned long getWidthLayout() const {return m_widthLayout;};
	void setWidthLayout(const unsigned long in) {m_widthLayout=in;};
	unsigned long getHeightLayout() const {return m_heightLayout;};
	void setHeightLayout(const unsigned long in) {m_heightLayout=in;};
	Vec3d getVolumeWidth() const {return m_volumeWidth;};
	void setVolumeWidth(const Vec3d in) {m_volumeWidth=in;};
	Vec3d getSourcePos() const {return m_sourcePos;};
	void setSourcePos(const Vec3d in) {m_sourcePos=in;};
	double getMeanFreePath() const {return m_meanFreePath;};
	void setMeanFreePath(const double in) {m_meanFreePath=in;};
	double getAnisotropy() const {return m_anisotropy;};
	void setAnisotropy(const double in) {m_anisotropy=in;};



	bool signalDataChanged();

	bool writeToXML(QDomDocument &document, QDomElement &root) const;
	bool readFromXML(const QDomElement &node);
	void render(QMatrix4x4 &m, RenderOptions &options);

private:
	Vec3d m_tilt;
	double m_power;
	unsigned long m_width;
	unsigned long m_height;
	unsigned long m_widthLayout;
	unsigned long m_heightLayout;
	Vec3d m_volumeWidth;
	Vec3d m_sourcePos;
	double m_meanFreePath;
	double m_anisotropy;


}; // class RayFieldItem


}; //namespace macrosim

#endif