from django.http import HttpResponse
from django.shortcuts import render

from .apps import MainConfig as conf
from .ml.explanations import *

import base64

import tempfile


def index(request):
    if request.method == 'GET':
        smiles = request.GET.get('smiles')
        data = data_from_smiles(smiles)
        prediction = int(conf.gcn_model.predict(data.x, data.edge_index))
        response = {}

        for xai_method in conf.xai_methods:
            if type(xai_method.explainer) in [SubgraphX, PGExplainer]:
                continue
            explanation_img = explain(data, xai_method.explainer, prediction)
            with tempfile.NamedTemporaryFile() as f:
                explanation_img.save(f.name, format="PNG")
                response[xai_method.name] = base64.b64encode(f.read()).decode()

        return render(request, "index.html", context=response)
