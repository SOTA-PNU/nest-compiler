#!/usr/bin/env python3
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Merge Glow quantization profiles (-dump-profile YAML) collected on
disjoint image sets into one profile.

Each node's range becomes the union of the ranges, and every histogram is
redistributed onto the union range the same way Glow's rescaleHistogram()
does (each source bin spread over the destination bins it overlaps, in
proportion to the overlap) before the histograms are summed.

usage: merge_profiles.py out.yaml in1.yaml in2.yaml ...

Profiling SSD300-VGG16 takes about 16 s per image, so the profile of the VTA
bundle (vta/bundles/SSD300VGG16Test, ssd300_vgg16_calib500.yaml in the model
release) was collected in 10 shards of 50 images and merged with this script.
The images are the 500 val2017 images with the largest ids, resized by
"ssd300_coco_map.py prepare"; "ssd300_coco_map.py run --exclude" leaves them
out of the mAP. Each shard:

  image-classifier <50 PNGs> -m=ssd300_vgg16.onnx -model-input-name=input \
      -backend=CPU -image-layout=NCHW -image-channel-order=RGB \
      -image-mode=0to255 -mean=122.99925,116.9991,103.9992 -stddev=1,1,1 \
      -minibatch=1 -dump-profile=shard_NN.yaml -num-histogram-bins=2048 \
      -quantization-schema=symmetric_with_power2_scale
"""
import re
import sys

import numpy as np


def parse(path):
    text = open(path).read()
    docs = text.split("\n...\n")
    header = "\n...\n".join(docs[:2]) + "\n...\n"
    body = docs[2]
    entries = []
    for m in re.finditer(r"- NodeOutputName:\s*'([^']*)'\s*\n\s*Min:\s*(\S+)\s*\n"
                         r"\s*Max:\s*(\S+)\s*\n\s*Histogram:\s*\[([^\]]*)\]", body):
        hist = np.array([float(x) for x in m.group(4).replace("\n", " ").split(",")
                         if x.strip()])
        entries.append((m.group(1), float(m.group(2)), float(m.group(3)), hist))
    return header, entries


def rescale(hist, smin, smax, dmin, dmax):
    if smin == dmin and smax == dmax:
        return hist.copy()
    n = len(hist)
    sw = (smax - smin) / n
    dw = (dmax - dmin) / n
    out = np.zeros(n)
    for i, v in enumerate(hist):
        if v == 0:
            continue
        s0, s1 = smin + i * sw, smin + (i + 1) * sw
        j0 = max(0, min(n - 1, int((s0 - dmin) / dw)))
        j1 = max(0, min(n - 1, int((s1 - dmin) / dw)))
        for j in range(j0, j1 + 1):
            d0, d1 = dmin + j * dw, dmin + (j + 1) * dw
            overlap = min(s1, d1) - max(s0, d0)
            if overlap > 0:
                out[j] += v * overlap / sw
    return out


def main(out, *inputs):
    header, base = parse(inputs[0])
    merged = {name: (mn, mx, [(mn, mx, h)]) for name, mn, mx, h in base}
    for path in inputs[1:]:
        h2, entries = parse(path)
        assert h2.split("GraphPreLowerHash")[1] == header.split("GraphPreLowerHash")[1], \
            "profiles of different graphs"
        for name, mn, mx, h in entries:
            pmn, pmx, hs = merged[name]
            merged[name] = (min(pmn, mn), max(pmx, mx), hs + [(mn, mx, h)])
    lines = [header + "---"]
    for name, _, _, _ in base:
        mn, mx, hs = merged[name]
        if mn < mx:
            hist = sum(rescale(h, a, b, mn, mx) if a < b else h for a, b, h in hs)
        else:
            hist = sum(h for _, _, h in hs)
        lines.append("- NodeOutputName:  '%s'\n  Min:             %.17g\n"
                     "  Max:             %.17g\n  Histogram:       [ %s ]"
                     % (name, mn, mx, ", ".join("%.9g" % v for v in hist)))
    open(out, "w").write("\n".join(lines) + "\n...\n")
    print("merged %d profiles, %d nodes -> %s" % (len(inputs), len(base), out))


if __name__ == "__main__":
    main(*sys.argv[1:])
