__all__ = ["Pixal3DMultiviewTo3DPipeline"]


def __getattr__(name):
    if name == "Pixal3DMultiviewTo3DPipeline":
        from .pipeline import Pixal3DMultiviewTo3DPipeline

        return Pixal3DMultiviewTo3DPipeline
    raise AttributeError(name)
