from ray.rllib.models import ModelCatalog
from .models import CustomFeedForwardModel, CustomFeedForwardModel3D


# register models
ModelCatalog.register_custom_model('CustomFeedForwardModel', CustomFeedForwardModel)
ModelCatalog.register_custom_model('CustomFeedForwardModel3D', CustomFeedForwardModel3D)

