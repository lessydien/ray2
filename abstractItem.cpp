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

#include "abstractItem.h"
#include "geometryItemLib.h"
#include "materialItemLib.h"
#include "macrosim_scenemodel.h"
#include <vtkProperty.h>
#include <iostream>
using namespace std;

using namespace macrosim;

bool AbstractItem::writeToXML(QDomDocument &document, QDomElement &root) const 
{ 
	root.setAttribute("objectType", objectTypeToString(m_objectType)); 
	root.setAttribute("name", m_name);
	return true;
};

bool AbstractItem::readFromXML(const QDomElement &node) 
{	
	QString str = node.attribute("objectType" );
	m_objectType=stringToObjectType(str);
	m_root.X=(node.attribute("root.x")).toDouble();
	m_root.Y=(node.attribute("root.y")).toDouble();
	m_root.Z=(node.attribute("root.z")).toDouble();
	m_name=node.attribute("name");
	return true;
};

void AbstractItem::setMaterialType(const Abstract_MaterialType type) 
{
	// if materialtype changed
	if (m_materialType != type)
	{
		m_materialType = type; 
		MaterialItemLib l_matLib;
		MaterialItem::MaterialType l_newMatType=l_matLib.abstractMatTypeToMatType(type);
		MaterialItem* l_pNewMat=l_matLib.createMaterial(l_newMatType);
		AbstractItem* l_pOldMaterial=this->getChild(0);
		QModelIndex l_oldParentIndex;
		if (l_pOldMaterial!=NULL)
		{
			l_oldParentIndex=this->getModelIndex();
			// free old material
			this->m_childs.replace(0, l_pNewMat);
			emit itemExchanged(l_oldParentIndex, 0, 0, *l_pNewMat);
		}
		emit itemChanged(m_index, m_index);

		if (this->m_renderOptions.m_renderMode==RENDER_TRANSPARENCY )
			m_pActor->GetProperty()->SetOpacity(this->getChild(0)->getTransparency());
		else
			m_pActor->GetProperty()->SetOpacity(1.0);

		this->updateVtk();
	}
};

QModelIndex AbstractItem::hasActor(void *actor) const
{
	if (actor==this->m_pActor)
	{
		return this->getModelIndex();
	}
	return QModelIndex();
}

void AbstractItem::createModelIndex(QModelIndex &parent, int row, int coloumn)
{
	const SceneModel *l_pModel;
	l_pModel=reinterpret_cast<const SceneModel*>(parent.model());
	QModelIndex l_index;
	//l_pModel->beginInsertRows(parent, row, coloumn);
	l_index=l_pModel->index(row, coloumn, parent);
	this->setModelIndex(l_index);
	// loop over childs
	for (int i=0; i<this->getNumberOfChilds(); i++)
	{
		AbstractItem* l_pChild=this->getChild(i);
		l_pChild->createModelIndex(this->getModelIndex(), i, 0);
	}
};



void AbstractItem::setRender(const bool in)
{
	m_render=in;
	if (m_render)
		m_pActor->SetVisibility(1);
	else
		m_pActor->SetVisibility(0);

	this->updateVtk();
};

void AbstractItem::setRenderOptions(RenderOptions options)
{
	m_renderOptions=options;
	switch (options.m_renderMode)
	{
	case RENDER_TRANSPARENCY:
		if (this->m_childs.size() == 1)
			m_pActor->GetProperty()->SetOpacity(this->getChild(0)->getTransparency());
		else
			m_pActor->GetProperty()->SetOpacity(1.0);
		m_pActor->GetProperty()->SetRepresentationToSurface();
		break;
	case RENDER_SOLID:
		m_pActor->GetProperty()->SetOpacity(1.0);
		m_pActor->GetProperty()->SetRepresentationToSurface();
		break;
	case RENDER_WIREGRID:
		m_pActor->GetProperty()->SetOpacity(1.0);
		m_pActor->GetProperty()->SetRepresentationToWireframe();
		break;
	}
	m_pActor->GetProperty()->SetAmbient(options.m_ambientInt);
	m_pActor->GetProperty()->SetDiffuse(options.m_diffuseInt);
	m_pActor->GetProperty()->SetSpecular(options.m_specularInt);
};