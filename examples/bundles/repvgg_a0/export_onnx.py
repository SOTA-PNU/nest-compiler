"""Export RepVGG-A0 (timm) to the ONNX file used by this bundle.

RepVGG trains with 3x3, 1x1 and identity branches per block. Before export
the branches are merged (structural re-parameterization), so the ONNX graph
only has Conv, Relu, GlobalAveragePool, Flatten and Gemm.

    pip install torch timm onnx onnxruntime pillow
    python export_onnx.py ../../../glow/tests/images/imagenet/dog_207.png

The result is uploaded to the models-v1 release (NESTC_MODEL_URL).
"""
import sys

import numpy as np
import onnx
import onnxruntime as ort
import timm
import torch
from PIL import Image
from timm.utils.model import reparameterize_model

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

model = reparameterize_model(timm.create_model("repvgg_a0.rvgg_in1k", pretrained=True).eval())

img = np.asarray(Image.open(sys.argv[1]).convert("RGB"), np.float32) / 255.0
x = ((img - MEAN) / STD).transpose(2, 0, 1)[None]
with torch.no_grad():
    ref = model(torch.from_numpy(x)).numpy()

torch.onnx.export(model, torch.from_numpy(x), "repvgg_a0.onnx",
                  input_names=["input"], output_names=["output"],
                  opset_version=11, do_constant_folding=True, dynamo=False)
onnx.checker.check_model(onnx.load("repvgg_a0.onnx"))

out = ort.InferenceSession("repvgg_a0.onnx").run(None, {"input": x})[0]
print("top1 torch/onnx:", ref.argmax(), out.argmax(), "max abs diff:", np.abs(ref - out).max())
