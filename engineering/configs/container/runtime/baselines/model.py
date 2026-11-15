if __package__:
    from .core_model import Model as BaseModel
    from .complete_model import load_checkpoint
else:
    from core_model import Model as BaseModel
    from complete_model import load_checkpoint

class Model(BaseModel):
    @classmethod
    def load(cls, path):
        return load_checkpoint(path, BaseModel)
