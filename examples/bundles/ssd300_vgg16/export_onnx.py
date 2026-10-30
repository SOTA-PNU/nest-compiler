"""Export torchvision SSD300-VGG16 (COCO) to the ONNX file used by this bundle.

Only the network (VGG16 backbone, extra layers and the conv heads) is
exported. Post-processing (softmax, box decoding, NMS) has data-dependent
output sizes and is done in main.cpp instead.

  input   [1, 3, 300, 300]  RGB pixels (0..255) minus 255 * (0.48235, 0.45882,
                            0.40784), resized to 300x300 (bilinear)
  output  [1, 8732, 95]     per default box: 91 class logits (COCO category
                            ids, 0 = background) then the box regression
                            (dx, dy, dw, dh), box coder weights 10, 10, 5, 5

The graph is adjusted for the Glow ONNX loader without changing the result:
- pool3 uses ceil_mode=True; it follows a ReLU (values >= 0), so padding the
  bottom/right with 0 and pooling without ceil_mode is identical.
- F.normalize (L2Norm after conv4_3) exports a Clip with an empty max input,
  which the loader rejects; it is written out as x / max(||x||, eps).
- onnxsim folds the Shape/Gather chains of the head reshapes into constants.

    pip install torch torchvision onnx onnxsim onnxruntime
    python export_onnx.py

The result is uploaded to the models-v1 release (NESTC_MODEL_URL).
"""
import numpy as np
import onnx
import onnxruntime as ort
import onnxsim
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class PadMaxPool(nn.Module):
    """MaxPool2d(2, 2, ceil_mode=True) for inputs >= 0."""

    def forward(self, x):
        return F.max_pool2d(F.pad(x, (0, 1, 0, 1), value=0.0), 2, 2)


class SSDNetwork(nn.Module):
    def __init__(self, ssd):
        super().__init__()
        self.backbone = ssd.backbone
        self.head = ssd.head

    def forward(self, x):
        # SSDFeatureExtractorVGG.forward with the L2 normalization spelled out.
        bb = self.backbone
        x = bb.features(x)
        norm = torch.sqrt(torch.sum(x * x, dim=1, keepdim=True))
        norm = torch.maximum(norm, torch.full((1,), 1e-12))
        feats = [bb.scale_weight.view(1, -1, 1, 1) * (x / norm)]
        for block in bb.extra:
            x = block(x)
            feats.append(x)
        out = self.head(feats)
        return torch.cat([out["cls_logits"], out["bbox_regression"]], dim=2)


torch.manual_seed(0)
torch.set_grad_enabled(False)
ssd = torchvision.models.detection.ssd300_vgg16(weights="COCO_V1").eval()
for i, layer in enumerate(ssd.backbone.features):
    if isinstance(layer, nn.MaxPool2d) and layer.ceil_mode:
        ssd.backbone.features[i] = PadMaxPool()
net = SSDNetwork(ssd).eval()

x = torch.rand(1, 3, 300, 300) * 255 - 115
ref = net(x).numpy()
torch.onnx.export(net, x, "ssd300_vgg16.onnx", input_names=["input"],
                  output_names=["output"], opset_version=11,
                  do_constant_folding=True, dynamo=False)
model, ok = onnxsim.simplify(onnx.load("ssd300_vgg16.onnx"))
assert ok
onnx.save(model, "ssd300_vgg16.onnx")

out = ort.InferenceSession("ssd300_vgg16.onnx").run(None, {"input": x.numpy()})[0]
print("output", out.shape, "max abs diff torch/onnx: %.2e" % np.abs(ref - out).max())
