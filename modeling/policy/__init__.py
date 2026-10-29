from .denoise_actor_3d import DenoiseActor as DenoiseActor3D


def fetch_model_class(model_type):
    if model_type == 'denoise3d':  # standard 3DFA
        return DenoiseActor3D
    return None
