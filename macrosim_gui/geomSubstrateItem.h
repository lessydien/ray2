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

#ifndef SUBSTRATEITEM
#define SUBSTRATEITEM

#include <qicon.h>

#include "QPropertyEditor/CustomTypes.h"
#include "GeometryItem.h"

#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>

using namespace macrosim;

namespace macrosim 
{

/** @class MicroLensArrayItem
*   @brief class for visualizing objects from MacroSim
*   
*   The AddInModel supplies a widget showing the available objects from MacroSim with their name, filename, version and so on.
*/
class SubstrateItem :
	public GeometryItem
{
	Q_OBJECT

	Q_PROPERTY(double thickness READ getThickness WRITE setThickness DESIGNABLE true USER true);

	Q_CLASSINFO("thickness", "decimals=10;");


public:

	SubstrateItem(QString name="Substrate", QObject *parent=0);
	~SubstrateItem(void);

	// functions for property editor
	double getThickness() const {return m_thickness;};
	void setThickness(const double in) {m_thickness=in; this->updateVtk(); emit itemChanged(m_index, m_index);};

	bool writeToXML(QDomDocument &document, QDomElement &root) const;
	bool readFromXML(const QDomElement &node);
	void renderVtk(vtkSmartPointer<vtkRenderer> renderer);

	Vec3f calcNormal(Vec3f vertex);


private:
	void updateVtk();

	vtkSmartPointer<vtkPolyData> m_pPolydata;
	vtkSmartPointer<vtkPolyDataMapper> m_pMapper;

	double m_thickness;

};

}; //namespace macrosim

#endif