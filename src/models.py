import logging
from model.baseline_ann import baseline_ann
from model.baseline_ann_one_layer import baseline_ann_one_layer
from model.baseline_cnn import baseline_cnn
from model.baseline_lstm import baseline_lstm
from model.baseline_transformer import baseline_transformer


log = logging.getLogger(__name__)


class ModelBuilder(object):
    def __init__(
        self,
        model_id,
        input_shape,
        output_shape,
        output_activation,
        model_config,
    ):
        log.info(
            f"Initialized ModelBuilder with model_id={model_id}, "
            f"input_shape={input_shape}, output_shape={output_shape}, "
            f"output_activation={output_activation}, "
            f"model_config={model_config}"
        )
        self.model_id = model_id
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.output_activation = output_activation
        self.model_config = model_config

    def get_model(self):
        if self.model_id == "baseline-ann":
            model = baseline_ann
        elif self.model_id == "baseline-cnn":
            model = baseline_cnn
        elif self.model_id == "baseline-lstm":
            model = baseline_lstm
        elif self.model_id == "baseline-transformer":
            model = baseline_transformer
        elif self.model_id == "baseline-ann-one-layer":
            model = baseline_ann_one_layer
        else:
            raise ValueError(f"Unknown model_id {self.model_id}")
        return model(
            input_shape=self.input_shape,
            output_shape=self.output_shape,
            output_activation=self.output_activation,
            model_config=self.model_config,
        )
