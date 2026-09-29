_MODELS = {}

def register_model(cls=None, *, name=None):
    """A decorator for registering model classes."""

    def _register(cls):
        model_name = cls.__name__ if name is None else name
        assert model_name not in _MODELS, ValueError(f'Already registered model with name: {model_name}')
        _MODELS[model_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)