import io

from dig.xgraph.dataset.mol_dataset import x_map, e_map
from dig.xgraph.method import SubgraphX, PGExplainer, GNNExplainer, GradCAM, DeepLIFT
from dig.xgraph.method.base_explainer import ExplainerBase
import numpy as np
import torch
from torch_geometric.data import Data

from PIL import Image
from PIL.PngImagePlugin import PngImageFile
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D


def data_from_smiles(smiles: str) -> Data:
    mol = Chem.MolFromSmiles(smiles)

    x = [
        [
            x_map['atomic_num'].index(atom.GetAtomicNum()),
            x_map['chirality'].index(str(atom.GetChiralTag())),
            x_map['degree'].index(atom.GetTotalDegree()),
            x_map['formal_charge'].index(atom.GetFormalCharge()),
            x_map['num_hs'].index(atom.GetTotalNumHs()),
            x_map['num_radical_electrons'].index(atom.GetNumRadicalElectrons()),
            x_map['hybridization'].index(str(atom.GetHybridization())),
            x_map['is_aromatic'].index(atom.GetIsAromatic()),
            x_map['is_in_ring'].index(atom.IsInRing())
        ]
        for atom in mol.GetAtoms()
    ]
    x = torch.tensor(x, dtype=torch.float32).view(-1, 9)

    edge_index = []
    edge_attr = []

    for bond in mol.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        edge_index.extend([[begin_idx, end_idx], [end_idx, begin_idx]])

        e = [
            e_map['bond_type'].index(str(bond.GetBondType())),
            e_map['stereo'].index(str(bond.GetStereo())),
            e_map['is_conjugated'].index(bond.GetIsConjugated())
        ]
        edge_attr.extend([e, e])

    edge_index = torch.tensor(edge_index).t().to(torch.long).view(2, -1)
    edge_attr = torch.tensor(edge_attr, dtype=torch.long).view(-1, 3)

    if edge_index.numel() > 0:
        perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
        edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        smiles=smiles,
        mol=mol
    )

def attr2clr(a: float) -> tuple:
    CLR_MIN = (0.5, 0.5, 1.)
    CLR_MAX = (1., 0.5, 0.5)
    CLR_WHITE = (1., 1., 1.)

    c_max = np.array(CLR_MAX)
    c_min = np.array(CLR_MIN)
    c_white = np.array(CLR_WHITE)

    grad_pos = c_max - c_white
    grad_neg = c_min - c_white
    
    clr = c_white + grad_pos * a if a > 0 else c_white - grad_neg * a
    return tuple(clr)

def attrs_image(
    mol: Chem.rdchem.Mol,
    attr: np.ndarray,
    size: int=300,
    norm_scale: float=None
) -> PngImageFile:

    scale = norm_scale if norm_scale else np.max(np.abs(attr))
    attr /= scale

    atom_clrs = {i: attr2clr(at) for i, at in enumerate(attr)}
    hit_ats = list(range(len(attr)))
    hit_bonds = list(range(len(mol.GetBonds())))

    bond_clrs = {}
    for i, bond in enumerate(mol.GetBonds()):
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        bond_clrs[i] = attr2clr((attr[begin_idx] + attr[end_idx]) / 2)

    d = rdMolDraw2D.MolDraw2DCairo(size, size)
    rdMolDraw2D.PrepareAndDrawMolecule(
        d,
        mol,
        highlightAtoms=hit_ats,
        highlightAtomColors=atom_clrs,
        highlightBonds=hit_bonds,
        highlightBondColors=bond_clrs
    )
    d.FinishDrawing()
    img_bytes = d.GetDrawingText()

    return Image.open(io.BytesIO(img_bytes))

def get_explanation_image(data: Data, edge_mask: torch.Tensor, size: int=400) -> PngImageFile:
    attrs = torch.ones(data.x.shape[0])

    for i in range(data.x.shape[0]):
        l = []

        for e in range(data.edge_index.shape[1]):
            if data.edge_index[0, e] == i or data.edge_index[1, e] == i:
                l.append(edge_mask[e])

        attrs[i] = sum(l) / len(l) if len(l) else float("inf")

    return attrs_image(data.mol, -attrs.numpy(), size=size)

def explain(data: Data, explainer: ExplainerBase, prediction: int) -> PngImageFile:
    if type(explainer) == SubgraphX:
        return None
    if type(explainer) == PGExplainer:
        return None
    if type(explainer) in (GNNExplainer, GradCAM, DeepLIFT):
        _, hard_edge_masks, _ = explainer(
            data.x,
            data.edge_index,
            num_classes=2,
        )
        return get_explanation_image(data, hard_edge_masks[prediction])
    raise TypeError(f"{type(explainer)} is not a supported explainer.")
