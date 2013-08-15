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

#ifndef VOLUMESCATTERERITEM
#define VOLUMESCATTERERITEM

#include <qicon.h>

#include "QPropertyEditor/CustomTypes.h"
#include "GeometryItem.h"
//#include <QtOpenGL\qglfunctions.h>
//#include "glut.h"

#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>


using namespace macrosim;

namespace macrosim 
{

/** @class SphericalLenseItem
*   @brief class for visualizing objects from MacroSim
*   
*   The AddInModel supplies a widget showing the available objects from MacroSim with their name, filename, version and so on.
*/
class VolumeScattererItem :
	public GeometryItem
{
	Q_OBJECT

	Q_PROPERTY(double meanFreePath READ getMeanFreePath WRITE setMeanFreePath DESIGNABLE true USER true);
	Q_PROPERTY(double thickness READ getThickness WRITE setThickness DESIGNABLE true USER true);
	Q_PROPERTY(unsigned int maxNrBounces READ getMaxNrBounces WRITE setMaxNrBounces DESIGNABLE true USER true);
	Q_PROPERTY(bool showRayPaths READ getShowRayPaths WRITE setShowRayPaths DESIGNABLE true USER true);

	Q_CLASSINFO("meanFreePath", "decimals=10;");
	Q_CLASSINFO("thickness", "decimals=10;");


public:

	VolumeScattererItem(QString name="VolumeScatterer", QObject *parent=0);
	~VolumeScattererItem(void);

	// functions for property editor
	double getMeanFreePath() const {return m_meanFreePath;};
	void setMeanFreePath(const double in) {m_meanFreePath=in; this->updateVtk(); emit itemChanged(m_index, m_index);};
	double getThickness() const {return m_thickness;};
	void setThickness(const double in) {m_thickness=in; this->updateVtk(); emit itemChanged(m_index, m_index);};
	unsigned int getMaxNrBounces() const {return m_maxNrBounces;};
	void setMaxNrBounces(const unsigned int in) {m_maxNrBounces=in; this->updateVtk(); emit itemChanged(m_index, m_index);};
	bool getShowRayPaths() const {return m_showRayPaths;};
	void setShowRayPaths(const unsigned int in) {m_showRayPaths=in; this->updateVtk(); emit itemChanged(m_index, m_index);};

	bool writeToXML(QDomDocument &document, QDomElement &root) const;
	bool readFromXML(const QDomElement &node);

	Vec3f calcNormal(Vec3f vertex, Vec3f* neighbours, int nr);
	Vec3f calcNormal(Vec3f vertex);
	void renderVtk(vtkSmartPointer<vtkRenderer> renderer);
	void updateVtk();


private:

//	MaterialItem::MaterialType m_materialType;
	double m_meanFreePath;
	double m_thickness;
	unsigned int m_maxNrBounces;
	bool m_showRayPaths;

	vtkSmartPointer<vtkPolyData> m_pPolydata;
	vtkSmartPointer<vtkPolyDataMapper> m_pMapper;
};

}; //namespace macrosim

#endif