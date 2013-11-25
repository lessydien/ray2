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

#include "diffRayField_RayAiming_Item.h"
#include "macrosim_scenemodel.h"
#include "detectorIntensityItem.h"
#include <vtkVersion.h>
#include <vtkPlaneSource.h> 
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkSphereSource.h>

using namespace macrosim;

DiffRayField_RayAiming_Item::DiffRayField_RayAiming_Item(QString name, QObject *parent) :
	RayFieldItem(name, DIFFRAYFIELDRAYAIM, parent)
{
	this->setRender(false); //per default we dont render the ray field
}

DiffRayField_RayAiming_Item::~DiffRayField_RayAiming_Item()
{
	m_childs.clear();
}

bool DiffRayField_RayAiming_Item::signalDataChanged() 
{

	return true;
};

bool DiffRayField_RayAiming_Item::writeToXML(QDomDocument &document, QDomElement &root) const 
{
	QDomElement node = document.createElement("field");

	if (!RayFieldItem::writeToXML(document, node))
		return false;

	// write parameters of the detector of the scene
	const SceneModel *l_pModel;
	l_pModel=reinterpret_cast<const SceneModel*>(this->m_index.model());
	DetectorItem *l_det=l_pModel->getDetectorItem();
	if (l_det == NULL)
	{
		cout << "error in DiffRayFIeld_RayAiming_Item.writeToXml(): scene doesn't seem to have a detector" << endl;
		return false;
	}
	if ( (l_det->getDetType() != DetectorItem::INTENSITY) )
	{
		cout << "error in DiffRayFIeld_RayAiming_Item.writeToXml(): detector is not of type INTENSITY" << endl;
		return false;
	}

	DetectorIntensityItem *l_detInt=reinterpret_cast<DetectorIntensityItem*>(l_det);
	node.setAttribute("detPixel.x", QString::number(l_detInt->getDetPixel().X));
	node.setAttribute("detPixel.y", QString::number(l_detInt->getDetPixel().Y));
	node.setAttribute("detApertureHalfWidth.x", QString::number(l_detInt->getApertureHalfWidth().X));
	node.setAttribute("detApertureHalfWidth.y", QString::number(l_detInt->getApertureHalfWidth().Y));
	node.setAttribute("detTilt.x", QString::number(l_detInt->getTilt().X));
	node.setAttribute("detTilt.y", QString::number(l_detInt->getTilt().Y));
	node.setAttribute("detTilt.z", QString::number(l_detInt->getTilt().Z));
	node.setAttribute("detRoot.x", QString::number(l_detInt->getRoot().X));
	node.setAttribute("detRoot.y", QString::number(l_detInt->getRoot().Y));
	node.setAttribute("detRoot.z", QString::number(l_detInt->getRoot().Z));

	root.appendChild(node);
	return true;
}

bool DiffRayField_RayAiming_Item::readFromXML(const QDomElement &node)
{
	if (!RayFieldItem::readFromXML(node))
		return false;


	return true;
}

void DiffRayField_RayAiming_Item::render(QMatrix4x4 &m, RenderOptions &options)
{

}

void DiffRayField_RayAiming_Item::renderVtk(vtkSmartPointer<vtkRenderer> renderer)
{

}