"""Explicit versioned goal estimators; checkpoints select their observation recipe."""
from pathlib import Path

def goal_estimator(recipe):
    import rgbd_object_features as v1
    import rgbd_goal_geometry_v2 as v2
    for module in [v1,v2]:
        if recipe==module.GOAL_RECIPE:return module.GoalMemory,module.load_goal_features,Path(module.__file__)
    raise ValueError('Unsupported goal observation recipe '+str(recipe))
