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

#ifndef GEOMRENDERFIELDITEM
#define GEOMRENDERFIELDITEM

#include <qicon.h>

#include "QPropertyEditor/CustomTypes.h"
#include "rayFieldItem.h"

#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>


using namespace macrosim;

namespace macrosim 
{

/** @class GeomRenderFieldItem
*   @brief class for visualizing objects from MacroSim
*   
*   The AddInModel supplies a widget showing the available objects from MacroSim with their name, filename, version and so on.
*/
class GeomRenderFieldItem :
	public FieldItem
{
	Q_OBJECT

	Q_PROPERTY(Vec3d tilt READ getTilt WRITE setTilt DESIGNABLE true USER true);
	Q_PROPERTY(Vec3d rayDirection READ getRayDirection WRITE setRayDirection DESIGNABLE true USER true);
	Q_PROPERTY(double coherence READ getCoherence WRITE setCoherence DESIGNABLE true USER true);
	Q_PROPERTY(Vec3d pupRoot READ getPupRoot WRITE setPupRoot DESIGNABLE true USER true);
	Q_PROPERTY(Vec3d pupTilt READ getPupTilt WRITE setPupTilt DESIGNABLE true USER true);
    Q_PROPERTY(Vec2d pupAptRad READ getPupAptRad WRITE setPupAptRad DESIGNABLE true USER true);
	Q_PROPERTY(unsigned long width READ getWidth WRITE setWidth DESIGNABLE true USER true);
	Q_PROPERTY(unsigned long height READ getHeight WRITE setHeight DESIGNABLE true USER true);
	Q_PROPERTY(unsigned long widthLayout READ getWidthLayout WRITE setWidthLayout DESIGNABLE true USER true);
	Q_PROPERTY(unsigned long heightLayout READ getHeightLayout WRITE setHeightLayout DESIGNABLE true USER true);
    Q_PROPERTY(Vec2d raysPerPixel READ getRaysPerPixel WRITE setRaysPerPixel DESIGNABLE true USER true);
    Q_PROPERTY(QString FileName READ getFileName WRITE setFileName DESIGNABLE true USER true);

	Q_CLASSINFO("tilt", "decimals=10;");
	Q_CLASSINFO("rayDirection", "decimals=10;");
	Q_CLASSINFO("alphaMax", "decimals=10;");
	Q_CLASSINFO("alphaMin", "decimals=10;");


public:

	GeomRenderFieldItem(QString name="GeomRenderField", QObject *parent=0);
	~GeomRenderFieldItem(void);

	Vec3d getTilt() const {return m_tilt;};
	void setTilt(const Vec3d in) {m_tilt=in; this->updateVtk(); emit itemChanged(m_index, m_index);};
	Vec3d getRayDirection() const {return m_rayDirection;};
	void setRayDirection(const Vec3d in) {m_rayDirection=in;};
	Vec3d getPupRoot() const {return m_pupRoot;};
	void setPupRoot(const Vec3d in) {m_pupRoot=in;};
	Vec3d getPupTilt() const {return m_pupTilt;};
	void setPupTilt(const Vec3d in) {m_pupTilt=in;};
	Vec2d getPupAptRad() const {return m_pupAptRad;};
	void setPupAptRad(const Vec2d in) {m_pupAptRad=in;};
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
    Vec2d getRaysPerPixel() const {return m_raysPerPixel;};
    void setRaysPerPixel(const Vec2d in) {m_raysPerPixel=in;};
	void setFileName(const QString in) {m_fileName=in;};
	const QString getFileName() const {return m_fileName;};


	// functions for property editor
	bool signalDataChanged();

	bool writeToXML(QDomDocument &document, QDomElement &root) const;
	bool readFromXML(const QDomElement &node);
	void render(QMatrix4x4 &m, RenderOptions &options);
	void renderVtk(vtkSmartPointer<vtkRenderer> renderer);

protected:

	Vec3d m_tilt;
	Vec3d m_rayDirection;
	Vec2d m_pupAptRad;
	Vec3d m_pupRoot;
    Vec3d m_pupTilt;
	double m_coherence;
	unsigned long m_width;
	unsigned long m_height;
	unsigned long m_widthLayout;
	unsigned long m_heightLayout;
    Vec2d m_raysPerPixel;
    QString m_fileName;

private:
	void updateVtk();

	vtkSmartPointer<vtkPolyData> m_pPolydata;
	vtkSmartPointer<vtkPolyDataMapper> m_pMapper;

}; // class RayFieldItem


}; //namespace macrosim

#endif