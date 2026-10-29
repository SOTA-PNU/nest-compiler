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
"""ImageNet top-1/top-5 accuracy of the RepVGG-A0 bundles.

Not part of CI: the full validation set takes about an hour per bundle.

  prepare  Converts the ILSVRC2012 validation split (Hugging Face
           ILSVRC/imagenet-1k parquet files) to 224x224 PNGs with the timm
           preprocessing of repvgg_a0.rvgg_in1k (bilinear resize of the short
           side to 256, center crop 224) and writes labels.txt.
  run      Runs each bundle once per image with REPVGG_DUMP_LOGITS set,
           collects the logits and reports top-1/top-5. With --onnx the float
           ONNX model is run on the same PNGs as the reference.

Example:
  repvgg_a0_accuracy.py prepare --parquet-dir hf/data --out-dir val224
  repvgg_a0_accuracy.py run --images-dir val224 --work-dir results \\
      --onnx build/examples/bundles/repvgg_a0/repvgg_a0.onnx \\
      --bundle cpu=build/examples/bundles/repvgg_a0/RepVGGA0Bundle \\
      --bundle vta=build/vta/bundles/RepVGGA0Test/vtaRepVGGA0Bundle \\
      --bundle allvta=build/vta/bundles/RepVGGA0AllVTATest/vtaRepVGGA0AllVTABundle

Results are kept in --work-dir, so an interrupted run resumes where it
stopped.
"""

import argparse
import concurrent.futures
import glob
import io
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

NUM_CLASSES = 1000
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(img, size=224, crop_pct=0.875):
    from PIL import Image

    img = img.convert("RGB")
    scale = int(size / crop_pct)
    w, h = img.size
    if w < h:
        nw, nh = scale, int(scale * h / w)
    else:
        nw, nh = int(scale * w / h), scale
    img = img.resize((nw, nh), Image.BILINEAR)
    left = int(round((nw - size) / 2.0))
    top = int(round((nh - size) / 2.0))
    return img.crop((left, top, left + size, top + size))


def _prepare_one(job):
    from PIL import Image

    data, path = job
    preprocess(Image.open(io.BytesIO(data))).save(path)


def cmd_prepare(args):
    import pyarrow.parquet as pq

    files = sorted(glob.glob(os.path.join(args.parquet_dir, "validation-*.parquet")))
    if not files:
        sys.exit("no validation-*.parquet in %s" % args.parquet_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    labels = []
    idx = 0
    with concurrent.futures.ProcessPoolExecutor(args.jobs) as pool:
        for f in files:
            table = pq.read_table(f, columns=["image", "label"])
            images = table.column("image").to_pylist()
            jobs = []
            for image, label in zip(images, table.column("label").to_pylist()):
                name = "val_%05d.png" % idx
                labels.append("%s %d" % (name, label))
                path = os.path.join(args.out_dir, name)
                if not os.path.exists(path):
                    jobs.append((image["bytes"], path))
                idx += 1
            list(pool.map(_prepare_one, jobs, chunksize=64))
            print("%s: %d images so far" % (os.path.basename(f), idx), flush=True)
    with open(os.path.join(args.out_dir, "labels.txt"), "w") as out:
        out.write("\n".join(labels) + "\n")
    print("wrote %d images to %s" % (idx, args.out_dir))


def load_labels(images_dir, limit):
    names, labels = [], []
    with open(os.path.join(images_dir, "labels.txt")) as f:
        for line in f:
            name, label = line.split()
            names.append(name)
            labels.append(int(label))
    if limit:
        names, labels = names[:limit], labels[:limit]
    return names, np.array(labels)


def open_logits(path, n):
    if os.path.exists(path):
        arr = np.load(path, mmap_mode="r+")
        if arr.shape == (n, NUM_CLASSES):
            return arr
    arr = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32,
                                    shape=(n, NUM_CLASSES))
    arr[:] = np.nan
    arr.flush()
    return arr


def run_onnx(onnx_path, images_dir, names, logits):
    import onnxruntime as ort
    from PIL import Image

    # The model is exported with a fixed batch of 1.
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    for i in range(len(names)):
        if not np.isnan(logits[i, 0]):
            continue
        img = Image.open(os.path.join(images_dir, names[i])).convert("RGB")
        x = (np.asarray(img, dtype=np.float32) / 255.0 - MEAN) / STD
        logits[i] = sess.run(None, {"input": x.transpose(2, 0, 1)[None]})[0][0]
    logits.flush()


def run_bundle(exe, images_dir, names, logits, jobs, timeout):
    exe = os.path.abspath(exe)
    cwd = os.path.dirname(exe)
    todo = [i for i in range(len(names)) if np.isnan(logits[i, 0])]
    tmpdir = tempfile.mkdtemp(prefix="repvgg_logits_")
    failures = []

    def one(i):
        out = os.path.join(tmpdir, "%d.bin" % i)
        env = dict(os.environ, REPVGG_DUMP_LOGITS=out)
        img = os.path.abspath(os.path.join(images_dir, names[i]))
        # The exit code only says whether the image is class 207.
        subprocess.run([exe, img], cwd=cwd, env=env, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, timeout=timeout)
        try:
            v = np.fromfile(out, dtype=np.float32)
            os.remove(out)
        except OSError:
            v = None
        return i, v

    start = time.time()
    with concurrent.futures.ThreadPoolExecutor(jobs) as pool:
        for n, (i, v) in enumerate(pool.map(one, todo), 1):
            if v is None or v.size != NUM_CLASSES:
                failures.append(names[i])
            else:
                logits[i] = v
            if n % 2000 == 0 or n == len(todo):
                logits.flush()
                print("  %d/%d (%.0fs)" % (n, len(todo), time.time() - start),
                      flush=True)
    os.rmdir(tmpdir)
    return failures


def topk(logits, labels):
    top5 = np.argsort(-logits, axis=1)[:, :5]
    top1 = top5[:, 0]
    return top1, (top1 == labels).mean(), (top5 == labels[:, None]).any(1).mean()


def cmd_run(args):
    names, labels = load_labels(args.images_dir, args.limit)
    n = len(names)
    os.makedirs(args.work_dir, exist_ok=True)
    runs = []
    if args.onnx:
        runs.append(("float", None))
    for spec in args.bundle:
        name, exe = spec.split("=", 1)
        runs.append((name, exe))

    results = {}
    for name, exe in runs:
        logits = open_logits(os.path.join(args.work_dir, name + "_logits.npy"), n)
        print("[%s] %s" % (name, exe or args.onnx), flush=True)
        failures = []
        if exe is None:
            run_onnx(args.onnx, args.images_dir, names, logits)
        else:
            failures = run_bundle(exe, args.images_dir, names, logits, args.jobs,
                                  args.timeout)
        results[name] = (np.array(logits), failures)

    ref_top1 = topk(results["float"][0], labels)[0] if "float" in results else None
    summary = {"images": n, "results": {}}
    print("\n%-8s %8s %8s %12s %8s" % ("model", "top1", "top5", "top1==float", "failed"))
    for name, (logits, failures) in results.items():
        valid = ~np.isnan(logits[:, 0])
        top1, acc1, acc5 = topk(logits[valid], labels[valid])
        agree = (top1 == ref_top1[valid]).mean() if ref_top1 is not None else float("nan")
        summary["results"][name] = {"top1": acc1, "top5": acc5,
                                    "top1_agree_float": agree,
                                    "evaluated": int(valid.sum()),
                                    "failed": failures}
        print("%-8s %7.2f%% %7.2f%% %11.2f%% %8d" % (name, 100 * acc1, 100 * acc5,
                                                      100 * agree, len(failures)))
    with open(os.path.join(args.work_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("prepare", help="convert the HF parquet files to PNGs")
    pp.add_argument("--parquet-dir", required=True)
    pp.add_argument("--out-dir", required=True)
    pp.add_argument("--jobs", type=int, default=16)
    pp.set_defaults(func=cmd_prepare)

    pr = sub.add_parser("run", help="measure the bundles")
    pr.add_argument("--images-dir", required=True,
                    help="output of 'prepare' (PNGs and labels.txt)")
    pr.add_argument("--work-dir", required=True,
                    help="where logits and summary.json are kept")
    pr.add_argument("--bundle", action="append", default=[], metavar="NAME=EXE",
                    help="bundle executable; may be repeated")
    pr.add_argument("--onnx", help="float ONNX model to use as the reference")
    pr.add_argument("--jobs", type=int, default=16)
    pr.add_argument("--timeout", type=int, default=120)
    pr.add_argument("--limit", type=int, default=0,
                    help="only the first N images (0 = all)")
    pr.set_defaults(func=cmd_run)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
