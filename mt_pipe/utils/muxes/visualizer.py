from typing import Any, Dict, Callable

from ...visualizers import Visualizer


class VisualizerMux:

    def __init__(
        self, visualizers: Dict[str, Visualizer], input_mappers: Dict[str, Callable]
    ) -> None:
        self.visualizers = visualizers
        self.input_mappers = input_mappers

    def __call__(self, batch, model_out, epoch, loop) -> Any:
        for dp, visualizer in self.visualizers.items():
            mapper = self.input_mappers[dp]
            visual_in = mapper(
                batch=batch[dp], model_out=model_out[dp], epoch=epoch, loop=loop
            )
            (
                visualizer(**visual_in)
                if isinstance(visual_in, dict)
                else visualizer(*visual_in)
            )
