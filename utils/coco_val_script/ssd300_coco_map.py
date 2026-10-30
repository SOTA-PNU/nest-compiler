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
"""COCO val2017 mAP of SSD300-VGG16 bundles (not part of CI).

  prepare  Resizes every val2017 image to 300x300 the way torchvision's SSD
           transform does (bilinear, aspect ratio not kept) and saves PNGs
           plus meta.json (image id and original size).
  run      Runs the float ONNX model and/or bundle executables on the PNGs.
           Bundles are run once per image with SSD_DUMP_OUTPUT set. The raw
           [8732, 95] output is decoded with torchvision's own SSD
           post-processing and scored with pycocotools.

Example:
  ssd300_coco_map.py prepare --coco-dir coco --out-dir coco300
  ssd300_coco_map.py run --images-dir coco300 --coco-dir coco --work-dir res \\
      --onnx build/examples/bundles/ssd300_vgg16/ssd300_vgg16.onnx \\
      --bundle cpu=build/examples/bundles/ssd300_vgg16/SSD300VGG16Bundle

Raw outputs are kept in --work-dir, so an interrupted run resumes.
"""

import argparse
import concurrent.futures
import contextlib
import io
import json
import os
import subprocess
import tempfile
import time

import numpy as np

NUM_ANCHORS = 8732
NUM_OUT = 95  # 91 class logits + 4 box regression values
MEAN = np.array([0.48235, 0.45882, 0.40784], dtype=np.float32) * 255


def _prepare_one(job):
    import torch
    import torch.nn.functional as F
    from PIL import Image

    src, dst = job
    img = Image.open(src).convert("RGB")
    t = torch.from_numpy(np.asarray(img, dtype=np.float32)).permute(2, 0, 1)
    t = F.interpolate(t[None], size=(300, 300), mode="bilinear",
                      align_corners=False)[0]
    arr = t.round().clamp(0, 255).byte().permute(1, 2, 0).numpy()
    Image.fromarray(arr).save(dst)
    return img.size


def cmd_prepare(args):
    ann = json.load(open(os.path.join(args.coco_dir, "annotations",
                                      "instances_val2017.json")))
    images = sorted(ann["images"], key=lambda im: im["id"])
    os.makedirs(args.out_dir, exist_ok=True)
    jobs = [(os.path.join(args.coco_dir, "val2017", im["file_name"]),
             os.path.join(args.out_dir, "%012d.png" % im["id"])) for im in images]
    with concurrent.futures.ProcessPoolExecutor(args.jobs) as pool:
        list(pool.map(_prepare_one, jobs, chunksize=32))
    meta = [{"id": im["id"], "png": "%012d.png" % im["id"],
             "width": im["width"], "height": im["height"]} for im in images]
    json.dump(meta, open(os.path.join(args.out_dir, "meta.json"), "w"))
    print("wrote %d images to %s" % (len(meta), args.out_dir))


def open_outputs(path, n):
    if os.path.exists(path):
        arr = np.load(path, mmap_mode="r+")
        if arr.shape == (n, NUM_ANCHORS, NUM_OUT):
            return arr
    # float16 keeps 5000 raw outputs at 8 GB instead of 16 GB.
    arr = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16,
                                    shape=(n, NUM_ANCHORS, NUM_OUT))
    arr[:, 0, 0] = np.nan
    arr.flush()
    return arr


def run_onnx(onnx_path, images_dir, meta, out, threads):
    import onnxruntime as ort
    from PIL import Image

    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    sess = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
    start = time.time()
    for i, m in enumerate(meta):
        if not np.isnan(out[i, 0, 0]):
            continue
        img = np.asarray(Image.open(os.path.join(images_dir, m["png"])).convert("RGB"),
                         dtype=np.float32)
        x = (img - MEAN).transpose(2, 0, 1)[None]
        out[i] = sess.run(None, {"input": x})[0][0]
        if (i + 1) % 500 == 0:
            out.flush()
            print("  %d/%d (%.0fs)" % (i + 1, len(meta), time.time() - start), flush=True)
    out.flush()


def run_bundle(exe, images_dir, meta, out, jobs, timeout):
    exe = os.path.abspath(exe)
    todo = [i for i in range(len(meta)) if np.isnan(out[i, 0, 0])]
    tmpdir = tempfile.mkdtemp(prefix="ssd_out_")
    failures = []

    def one(i):
        path = os.path.join(tmpdir, "%d.bin" % i)
        env = dict(os.environ, SSD_DUMP_OUTPUT=path)
        img = os.path.abspath(os.path.join(images_dir, meta[i]["png"]))
        subprocess.run([exe, img], cwd=os.path.dirname(exe), env=env,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       timeout=timeout)
        try:
            v = np.fromfile(path, dtype=np.float32)
            os.remove(path)
        except OSError:
            v = None
        return i, v

    start = time.time()
    with concurrent.futures.ThreadPoolExecutor(jobs) as pool:
        for n, (i, v) in enumerate(pool.map(one, todo), 1):
            if v is None or v.size != NUM_ANCHORS * NUM_OUT:
                failures.append(meta[i]["png"])
            else:
                out[i] = v.reshape(NUM_ANCHORS, NUM_OUT)
            if n % 500 == 0 or n == len(todo):
                out.flush()
                print("  %d/%d (%.0fs)" % (n, len(todo), time.time() - start),
                      flush=True)
    os.rmdir(tmpdir)
    return failures


def default_boxes():
    """SSD300 default boxes (xyxy, 300x300 pixels) from torchvision."""
    import torch
    import torchvision
    from torchvision.models.detection.image_list import ImageList

    model = torchvision.models.detection.ssd300_vgg16(weights=None,
                                                       weights_backbone=None)
    x = torch.zeros(1, 3, 300, 300)
    sizes = [38, 19, 10, 5, 3, 1]
    feats = [torch.zeros(1, 1, s, s) for s in sizes]
    return model.anchor_generator(ImageList(x, [(300, 300)]), feats)[0].numpy()


def detections(outputs, meta, anchors):
    """torchvision's own SSD post-processing on the raw outputs."""
    import torch
    import torchvision

    model = torchvision.models.detection.ssd300_vgg16(weights=None,
                                                       weights_backbone=None)
    # transform.postprocess skips the rescale to the original size in
    # training mode.
    model.eval()
    anchors = torch.from_numpy(anchors)
    results = []
    for i, m in enumerate(meta):
        o = torch.from_numpy(np.array(outputs[i], dtype=np.float32))
        head = {"cls_logits": o[None, :, :91], "bbox_regression": o[None, :, 91:]}
        det = model.postprocess_detections(head, [anchors], [(300, 300)])
        det = model.transform.postprocess(det, [(300, 300)],
                                          [(m["height"], m["width"])])[0]
        for box, score, label in zip(det["boxes"].tolist(), det["scores"].tolist(),
                                     det["labels"].tolist()):
            x1, y1, x2, y2 = box
            results.append({"image_id": m["id"], "category_id": int(label),
                            "bbox": [x1, y1, x2 - x1, y2 - y1], "score": score})
    return results


def coco_map(coco_dir, results, image_ids):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(os.path.join(coco_dir, "annotations", "instances_val2017.json"))
        dt = gt.loadRes(results) if results else None
    if dt is None:
        return [0.0] * 12
    ev = COCOeval(gt, dt, "bbox")
    ev.params.imgIds = image_ids
    with contextlib.redirect_stdout(io.StringIO()):
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return [float(s) for s in ev.stats]


def cmd_run(args):
    meta = json.load(open(os.path.join(args.images_dir, "meta.json")))
    if args.limit:
        meta = meta[:args.limit]
    n = len(meta)
    anchors = default_boxes()
    os.makedirs(args.work_dir, exist_ok=True)
    runs = ([("float", None)] if args.onnx else []) + \
        [tuple(b.split("=", 1)) for b in args.bundle]

    summary = {"images": n, "results": {}}
    exclude = set()
    if args.exclude:
        exclude = {int(x) for x in open(args.exclude).read().split()}
    for name, exe in runs:
        out = open_outputs(os.path.join(args.work_dir, name + "_out.npy"), n)
        print("[%s] %s" % (name, exe or args.onnx), flush=True)
        failures = []
        if exe is None:
            run_onnx(args.onnx, args.images_dir, meta, out, args.jobs)
        else:
            failures = run_bundle(exe, args.images_dir, meta, out, args.jobs,
                                  args.timeout)
        valid = [i for i in range(n) if not np.isnan(out[i, 0, 0])]
        vmeta = [meta[i] for i in valid]
        res = detections(out[valid], vmeta, anchors)
        ids = [m["id"] for m in vmeta]
        stats = coco_map(args.coco_dir, res, ids)
        entry = {"mAP": stats[0], "AP50": stats[1], "AP75": stats[2],
                 "evaluated": len(valid), "failed": failures}
        if exclude:
            held = [i for i in ids if i not in exclude]
            hs = coco_map(args.coco_dir, [r for r in res if r["image_id"] not in exclude],
                          held)
            entry.update({"mAP_heldout": hs[0], "AP50_heldout": hs[1],
                          "heldout_images": len(held)})
        summary["results"][name] = entry
        if failures:
            print("WARNING [%s]: %d of %d images failed (crash or no output); "
                  "mAP below covers only the rest" % (name, len(failures), n),
                  flush=True)
        print("[%s] %d images  mAP %.3f  AP50 %.3f  AP75 %.3f%s" % (
            name, len(valid), stats[0], stats[1], stats[2],
            "  (held-out mAP %.3f AP50 %.3f)" % (entry["mAP_heldout"],
                                                  entry["AP50_heldout"])
            if exclude else ""), flush=True)
    json.dump(summary, open(os.path.join(args.work_dir, "summary.json"), "w"),
              indent=2)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("prepare", help="resize val2017 to 300x300 PNGs")
    pp.add_argument("--coco-dir", required=True,
                    help="contains val2017/ and annotations/")
    pp.add_argument("--out-dir", required=True)
    pp.add_argument("--jobs", type=int, default=16)
    pp.set_defaults(func=cmd_prepare)

    pr = sub.add_parser("run", help="measure mAP")
    pr.add_argument("--images-dir", required=True, help="output of 'prepare'")
    pr.add_argument("--coco-dir", required=True)
    pr.add_argument("--work-dir", required=True)
    pr.add_argument("--onnx", help="float ONNX model to use as the reference")
    pr.add_argument("--bundle", action="append", default=[], metavar="NAME=EXE")
    pr.add_argument("--exclude", help="file with image ids used for "
                    "calibration; mAP is also reported without them")
    pr.add_argument("--jobs", type=int, default=16)
    pr.add_argument("--timeout", type=int, default=600)
    pr.add_argument("--limit", type=int, default=0)
    pr.set_defaults(func=cmd_run)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
