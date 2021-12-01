import torch
import torch.nn as nn
import torch.nn.functional as F

from detr.modeling.backbone import Backbone, Joiner
from detr.utils.tensor import NestedTensor, nested_tensor_from_tensor_list
from detr.modeling.layers.positional_encoding import PositionEmbeddingLearned, PositionEmbeddingSine
from detr.modeling.layers.transformer import Transformer


def build_position_encoding(hidden_dim, positional_embedding="v2"):
    N_steps = hidden_dim // 2
    if positional_embedding in ("v2", "sine"):
        # TODO find a better way of exposing other arguments
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    elif positional_embedding in ("v3", "learned"):
        position_embedding = PositionEmbeddingLearned(N_steps)
    else:
        raise ValueError(f"not supported {positional_embedding}")
    return position_embedding


def build_transformer(
    hidden_dim,
    dropout=0.1,
    num_heads=2,
    dim_feedforward=512,
    num_encoder_layers=1,
    num_decoder_layers=1,
    pre_norm=False,
):
    return Transformer(
        d_model=hidden_dim,
        dropout=dropout,
        nhead=num_heads,
        dim_feedforward=dim_feedforward,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        normalize_before=pre_norm,
        return_intermediate_dec=True,
    )


def build_backbone(backbone, hidden_dim, positional_embedding="sine", lr_backbone=0.0, masks=False, dilation=False):
    position_embedding = build_position_encoding(hidden_dim, positional_embedding)
    train_backbone = lr_backbone > 0
    backbone = Backbone(backbone, train_backbone, masks, dilation)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class DETR(nn.Module):
    """This is the DETR module that performs object detection"""

    def __init__(self, backbone, transformer, num_classes, num_queries, aux_loss=False, onnx_export=False):
        """Initializes the model.

        Args:
            backbone (torch.nn.Module): torch module of the backbone to be used. See backbone.py
            transformer (torch.nn.Module): torch module of the transformer architecture. See transformer.py
            num_classes (int): number of object classes
            num_queries (int): number of object queries, ie detection slot. This is the maximal number of objects
                DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss (bool): `True` if auxiliary decoding losses (loss at each decoder layer) are to be used.
                Default is `False`.
            onnx_export (bool): flag to use model in export mode.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss
        self._onnx_export = False

    def onnx_export(self, flag=False):
        """Enable or disable onnx export mode.

        Args:
            flag (bool): enable/disable onnx export mode.
                Default is False.
        """
        self._onnx_export = flag

    def onnx_forward(self, inputs):
        if isinstance(inputs, (list, torch.Tensor)):
            inputs = nested_tensor_from_tensor_list(inputs)

        features, pos = self.backbone(inputs)

        src, mask = features[-1].decompose()
        assert mask is not None
        hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]

        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()

        return outputs_coord, outputs_class

    def basic_forward(self, samples: NestedTensor):
        """The forward expects a NestedTensor, which consists of:
        - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
        - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

        It returns a dict with the following elements:
            - "pred_logits": the classification logits (including no-object) for all queries.
                            Shape= [batch_size x num_queries x (num_classes + 1)]
            - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                (center_x, center_y, height, width). These values are normalized in [0, 1],
                relative to the size of each individual image (disregarding possible padding).
                See PostProcess for information on how to retrieve the unnormalized bounding box.
            - "aux_outputs": Optional, only returned when auxiliary losses are activated. It is a list of
                            dictionaries containing the two above keys for each decoder layer.
        """
        outputs_coord, outputs_class = self.onnx_forward(samples)

        out = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}
        if self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss(outputs_class, outputs_coord)
        return out

    def forward(self, samples: NestedTensor):
        if self._onnx_export:
            return self.onnx_forward(samples)
        else:
            return self.basic_forward(samples)

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{"pred_logits": a, "pred_boxes": b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    @staticmethod
    def save_onnx(model, inputs, path):
        """Export model to .onnx file.

        Args:
            model (DETR): model to export
            inputs (torch.Tensor): model example input.
            path (str or Path): path to a file where model will be stored.
        """
        prev_state = model.training
        model.eval()

        # save model
        torch.onnx.export(
            model,
            inputs,
            str(path),
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["boxes", "logits"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "boxes": {0: "batch_size"},
                "logits": {0: "batch_size"},
            },
            enable_onnx_checker=True,
            training=torch.onnx.TrainingMode.EVAL,  # it will keep bn layers as it is
        )

        # switch to previous mode
        model.onnx_export(False)
        model.train(prev_state)

    @staticmethod
    def save_tensorflow(model, inputs, path):
        """Export model to .tf file.

        Args:
            model (DETR): model to export.
            inputs (torch.Tensor): model example input.
            path (str or pathlib.Path): path to a file where will be stored a model.
        """
        import os
        from tempfile import TemporaryDirectory

        import onnx
        from onnx_tf.backend import prepare

        with TemporaryDirectory() as tmp_dir:
            onnx_file = os.path.join(tmp_dir, "model.onnx")
            DETR.save_onnx(model, inputs, onnx_file)

            # export onnx to tensorflow
            onnx_model = onnx.load(onnx_file)
            tf_representation = prepare(onnx_model)
            tf_representation.export_graph(str(path))

    @staticmethod
    def save_tflite(model, inputs, path):
        """Export model to .tflite file.

        Args:
            model (DETR): model to export.
            inputs (torch.Tensor): model example input.
            path (str or pathlib.Path): path to a file where will be stored model.
        """
        import os
        from tempfile import TemporaryDirectory

        import onnx
        import tensorflow as tf
        from onnx_tf.backend import prepare

        with TemporaryDirectory() as tmp_dir, open(str(path), "wb") as tflite_file:
            # step 1: export to ONNX
            onnx_file = os.path.join(tmp_dir, "model.onnx")
            DETR.save_onnx(model, inputs, onnx_file)

            # step 2: export to TensorFlow
            tf_file = os.path.join(tmp_dir, "model.tf")
            onnx_model = onnx.load(onnx_file)
            tf_representation = prepare(onnx_model)
            tf_representation.export_graph(tf_file)

            # step 3: export to TFLite
            converter = tf.lite.TFLiteConverter.from_saved_model(tf_file)
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
                tf.lite.OpsSet.SELECT_TF_OPS,  # enable TensorFlow ops.
            ]
            tflite_model = converter.convert()
            tflite_file.write(tflite_model)

    @staticmethod
    def save_coreml(model, inputs, path):
        """Export model to CoreML format (.mlmodel).

        Args:
            model (DETR): model to export.
            inputs (torch.Tensor): model examole input.
            path (str or pathlib.Path): path to a file where will be stored a model.
        """
        import coremltools as ct

        from alto_ai.utils.misc import type_as  # noqa: F401

        prev_state = model.training
        model.eval()
        model.onnx_export(True)

        # initialize grids if they aren't initialized
        with torch.no_grad():
            boxes, logits = model(inputs)

        traced_model = torch.jit.trace(model, inputs)
        traced_model.eval()

        input_image = ct.ImageType(name="inputs", shape=(1, *inputs.shape[1:]), scale=1 / 255, channel_first=True)
        coreml_model = ct.convert(traced_model, inputs=[input_image])

        spec = coreml_model.get_spec()

        ct.utils.rename_feature(spec, spec.description.output[0].name, "boxes")
        ct.utils.rename_feature(spec, spec.description.output[1].name, "logits")

        for output_idx, shapes in enumerate([boxes.shape[1:], logits.shape[1:]]):
            spec.description.output[output_idx].type.multiArrayType.shape.append(1)  # batch dimension
            for dim_size in shapes:
                spec.description.output[output_idx].type.multiArrayType.shape.append(dim_size)

        coreml_model = ct.models.MLModel(spec)
        coreml_model.save(str(path))

        # switch to previous mode
        model.train(prev_state)
        model.onnx_export(False)


def build_detr(
    num_classes,
    backbone="resnet18",
    hidden_dim=256,  # size of the embeddings, dimension size of transformer
    positional_embedding="sine",
    lr_backbone=1e-5,
    dilation=False,  # option to replace stride with dilation in the last convolution layer (DC5)
    dropout=0.1,
    num_heads=8,
    dim_feedforward=2048,
    num_encoder_layers=6,
    num_decoder_layers=6,
    pre_norm=False,
    n_objects=100,  # max number of objects on image
    *args,
    **kwargs,
):
    masks = False
    use_aux_loss = False

    backbone_model = build_backbone(backbone, hidden_dim, positional_embedding, lr_backbone, masks, dilation)

    transformer_model = build_transformer(
        hidden_dim, dropout, num_heads, dim_feedforward, num_encoder_layers, num_decoder_layers, pre_norm
    )

    model = DETR(
        backbone_model,
        transformer_model,
        num_classes=num_classes,
        num_queries=n_objects,
        aux_loss=use_aux_loss,
    )

    return model
