import base64
import tempfile
from typing import Dict

from django.shortcuts import render

from .apps import MainConfig as conf
from .ml.explanations import *
from torch_geometric.data import Data


def get_explanations(data: Data) -> Dict[str, str]:
    prediction = int(conf.gcn_model.predict(data.x, data.edge_index))
    explanations = {}

    for xai_method in conf.xai_methods:
        if type(xai_method.explainer) in [SubgraphX, PGExplainer]:
            continue
        explanation_img = explain(data, xai_method.explainer, prediction)
        with tempfile.NamedTemporaryFile() as f:
            explanation_img.save(f.name, format="PNG")
            explanations[xai_method.name] = base64.b64encode(f.read()).decode()

    return explanations

def index(request):
    if request.method == 'GET':
        smiles = request.GET.get('smiles')

        if not smiles:
            return render(request, "index.html")

        try:
            data = data_from_smiles(smiles)
        except AttributeError:
            return render(request, "index.html", context={"error": True})

        return render(request, "index.html", context=get_explanations(data))
