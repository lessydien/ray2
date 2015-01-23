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

#ifndef DETECTORVOLUMEINTENSITYITEM
#define DETECTORVOLUMEINTENSITYITEM

#include <qicon.h>

#include "QPropertyEditor/CustomTypes.h"
#include "DetectorItem.h"

using namespace macrosim;

namespace macrosim 
{

/** @class FieldItem
*   @brief class for visualizing objects from MacroSim
*   
*   The AddInModel supplies a widget showing the available objects from MacroSim with their name, filename, version and so on.
*/
class DetectorVolumeIntensityItem :
	public DetectorItem
{
	Q_OBJECT

	Q_PROPERTY(Vec3d detPixel READ getDetPixel WRITE setDetPixel DESIGNABLE true USER true);	
	Q_PROPERTY(Vec3d apertureHalfWidth READ getApertureHalfWidth WRITE setApertureHalfWidth DESIGNABLE true USER true);
	Q_PROPERTY(int ignoreDepth READ getIgnoreDepth WRITE setIgnoreDepth DESIGNABLE true USER true);	

public:

	DetectorVolumeIntensityItem(QString name="DetVolumeIntensity", QObject *parent=0);
	~DetectorVolumeIntensityItem(void);

	// functions for property editor
	const Vec3d getDetPixel()  {return m_detPixel;};
	void setDetPixel(const Vec3d in) {m_detPixel=in;};
	const Vec3d getApertureHalfWidth()  {return m_apertureHalfWidth;};
	void setApertureHalfWidth(const Vec3d in) {m_apertureHalfWidth=in;};
	const int getIgnoreDepth() {return m_ignoreDepth;};
	void setIgnoreDepth(const int in) {m_ignoreDepth=in;};

	void render(QMatrix4x4 &m, RenderOptions &options);


	virtual bool writeToXML(QDomDocument &document, QDomElement &root) const;
	virtual bool readFromXML(const QDomElement &node);

private:
	Vec3d m_detPixel;
	Vec3d m_apertureHalfWidth;
	int m_ignoreDepth;

};

}; //namespace macrosim

#endif