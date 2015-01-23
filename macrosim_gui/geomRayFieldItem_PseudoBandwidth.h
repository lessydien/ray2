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

#ifndef GEOMRAYFIELDITEM_PSEUDOBANDWIDTH
#define GEOMRAYFIELDITEM_PSEUDOBANDWIDTH

#include <qicon.h>

#include "QPropertyEditor/CustomTypes.h"
#include "rayFieldItem.h"

using namespace macrosim;

namespace macrosim 
{

/** @class FieldItem
*   @brief class for visualizing objects from MacroSim
*   
*   The AddInModel supplies a widget showing the available objects from MacroSim with their name, filename, version and so on.
*/
class GeomRayFieldItem_PseudoBandwidth :
	public RayFieldItem
{
	Q_OBJECT
		Q_PROPERTY(double PseudoBandwidth READ getPseudoBandwidth WRITE setPseudoBandwidth DESIGNABLE true USER true);
		Q_PROPERTY(int nrPseudoLambdas READ getNrPseudoLambdas WRITE setNrPseudoLambdas DESIGNABLE true USER true);
public:

	GeomRayFieldItem_PseudoBandwidth(QString name="GeomRayField_PseudoBandwidth", QObject *parent=0);
	~GeomRayFieldItem_PseudoBandwidth(void);

	// functions for property editor
	double getPseudoBandwidth() const {return m_pseudoBandwidth;}
	void setPseudoBandwidth(const double in) {m_pseudoBandwidth=in;};
	int getNrPseudoLambdas() const {return m_nrPseudoLambdas;}
	void setNrPseudoLambdas(const int in) {m_nrPseudoLambdas=in;};

	bool writeToXML(QDomDocument &document, QDomElement &root) const;
	bool readFromXML(const QDomElement &node);
	void render(QMatrix4x4 &m, RenderOptions &options);
	void renderVtk(vtkSmartPointer<vtkRenderer> renderer);

private:
	double m_pseudoBandwidth;
	int m_nrPseudoLambdas;

}; // class RayFieldItem


}; //namespace macrosim

#endif