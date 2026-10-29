"""Export RepVGG-A0 (timm) to the ONNX file used by this bundle.

RepVGG trains with 3x3, 1x1 and identity branches per block. Before export
the branches are merged (structural re-parameterization), so the ONNX graph
only has Conv, Relu, GlobalAveragePool, Flatten and Gemm.

Merging the branches leaves a few output channels with weights 10-15x
larger than the rest of the layer. VTA quantizes weights per tensor with a
power-of-2 scale, so those channels squash everything else and int8 results
are garbage. Cross-layer equalization (Nagel et al., ICCV 2019) fixes this:
the network is a plain Conv-ReLU chain, and ReLU(s*x) = s*ReLU(x) for s > 0,
so each channel can be scaled down in one layer and up in the next without
changing the float output.

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



def equalize(model, iters=20):
    """Cross-layer equalization over consecutive Conv/Linear layers."""
    layers = [m for m in model.modules() if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear))]
    for _ in range(iters):
        for a, b in zip(layers[:-1], layers[1:]):
            r1 = a.weight.abs().flatten(1).amax(1)  # range of each output channel of a
            wb = b.weight if b.weight.dim() == 4 else b.weight[:, :, None, None]
            r2 = wb.abs().transpose(0, 1).flatten(1).amax(1)  # range of each input channel of b
            s = torch.where((r1 > 0) & (r2 > 0), torch.sqrt(r1 * r2) / r2, torch.ones_like(r1))
            a.weight.div_(s.view(-1, *[1] * (a.weight.dim() - 1)))
            a.bias.div_(s)
            b.weight.mul_(s.view(1, -1, *[1] * (b.weight.dim() - 2)))
    return model


torch.set_grad_enabled(False)
model = reparameterize_model(timm.create_model("repvgg_a0.rvgg_in1k", pretrained=True).eval())
model = equalize(model)

img = np.asarray(Image.open(sys.argv[1]).convert("RGB"), np.float32) / 255.0
x = ((img - MEAN) / STD).transpose(2, 0, 1)[None]
ref = model(torch.from_numpy(x)).numpy()

torch.onnx.export(model, torch.from_numpy(x), "repvgg_a0.onnx",
                  input_names=["input"], output_names=["output"],
                  opset_version=11, do_constant_folding=True, dynamo=False)
onnx.checker.check_model(onnx.load("repvgg_a0.onnx"))

out = ort.InferenceSession("repvgg_a0.onnx").run(None, {"input": x})[0]
print("top1 torch/onnx:", ref.argmax(), out.argmax(), "max abs diff:", np.abs(ref - out).max())
