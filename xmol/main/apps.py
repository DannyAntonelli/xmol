from dataclasses import dataclass

from django.apps import AppConfig
from django.conf import settings

import torch

from dig.xgraph.method import SubgraphX, PGExplainer, GNNExplainer, GradCAM, DeepLIFT
from dig.xgraph.method.base_explainer import ExplainerBase

from .ml.gcn import GCN


@dataclass
class XaiMethod:
    name: str
    explainer: ExplainerBase


class XaiGnnConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'main'

    gcn_model = GCN(
        input_dim=settings.NUM_FEATURES,
        output_dim=settings.NUM_CLASSES
    )
    gcn_model.load_state_dict(
        torch.load(
            settings.GCN_MODEL_PATH,
            map_location=torch.device(settings.DEVICE_TYPE),
        )
    )

    subgraphx = SubgraphX(
        gcn_model,
        num_classes=settings.NUM_CLASSES,
        device=torch.device(settings.DEVICE_TYPE),
        explain_graph=True,
    )

    pgexplainer = PGExplainer(
        gcn_model,
        in_channels=settings.PGEXPLAINER_IN_CHANNELS,
        device=torch.device(settings.DEVICE_TYPE),
        explain_graph=True,
    )
    pgexplainer.load_state_dict(
        torch.load(
            settings.PGEXPLAINER_MODEL_PATH,
            map_location=torch.device(settings.DEVICE_TYPE),
        )
    )

    gnnexplainer = GNNExplainer(gcn_model, explain_graph=True)
    gradcam = GradCAM(gcn_model, explain_graph=True)
    deeplift = DeepLIFT(gcn_model, explain_graph=True)

    xai_methods = [
        XaiMethod("SubgraphX", subgraphx),
        XaiMethod("PGExplainer", pgexplainer),
        XaiMethod("GNNExplainer", gnnexplainer),
        XaiMethod("GradCAM", gradcam),
        XaiMethod("DeepLIFT", deeplift),
    ]
