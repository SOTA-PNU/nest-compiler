/**
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <assert.h>
#include <inttypes.h>
#include <math.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <string>
#include <vector>

#include "ssd300_vgg16.h"

/// Runs the SSD300-VGG16 object detector (torchvision ssd300_vgg16, COCO).
///
/// The bundle only holds the network (backbone and heads). Everything the
/// compiler cannot handle well, i.e. the dynamic-size post-processing, is done
/// here and follows torchvision's SSD exactly:
///   resize to 300x300 (bilinear) -> bundle -> softmax -> decode boxes against
///   the default boxes -> per-class score threshold and top-k -> per-class NMS
///   -> keep the best 200 -> scale back to the input image size.
///
/// usage: SSD300VGG16Bundle image.png [label x1 y1 x2 y2]
///   With an expected detection, returns 0 only if a box of that COCO label
///   with score >= 0.5 overlaps it with IoU >= 0.5.
///   SSD_DUMP_OUTPUT=<file> writes the raw [8732, 95] network output.

#define INPUT_SIZE 300
#define NUM_ANCHORS 8732
#define NUM_CLASSES 91 // COCO category ids, 0 is background
#define NUM_OUT (NUM_CLASSES + 4)

static const float kMean[3] = {0.48235f * 255, 0.45882f * 255, 0.40784f * 255};
static const float kScoreThresh = 0.01f;
static const size_t kTopKCandidates = 400;
static const float kNmsThresh = 0.45f;
static const size_t kDetectionsPerImg = 200;
static const float kPrintThresh = 0.3f;

static const char *kCocoNames[NUM_CLASSES] = {
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "N/A",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "N/A", "backpack",
    "umbrella", "N/A", "N/A", "handbag", "tie", "suitcase", "frisbee", "skis",
    "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "N/A", "wine glass",
    "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "N/A", "dining table", "N/A",
    "N/A", "toilet", "N/A", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "N/A",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"};

struct Box {
  float x1, y1, x2, y2;
};

struct Detection {
  Box box;
  float score;
  int label;
};

//===----------------------------------------------------------------------===//
//                   Image loading and pre-processing
//===----------------------------------------------------------------------===//

/// Reads an 8-bit RGB, RGBA or gray PNG into \p rgb (row-major HxWx3, 0..255).
/// \returns false on error.
static bool readPng(const char *filename, std::vector<float> &rgb,
                    size_t &width, size_t &height) {
  FILE *fp = fopen(filename, "rb");
  if (!fp) {
    return false;
  }
  unsigned char header[8];
  if (fread(header, 1, 8, fp) != 8 || png_sig_cmp(header, 0, 8)) {
    fclose(fp);
    return false;
  }
  png_structp png =
      png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
  png_infop info = png ? png_create_info_struct(png) : nullptr;
  if (!info || setjmp(png_jmpbuf(png))) {
    png_destroy_read_struct(&png, &info, nullptr);
    fclose(fp);
    return false;
  }
  png_init_io(png, fp);
  png_set_sig_bytes(png, 8);
  png_read_info(png, info);
  width = png_get_image_width(png, info);
  height = png_get_image_height(png, info);
  int colorType = png_get_color_type(png, info);
  if (png_get_bit_depth(png, info) != 8 ||
      png_set_interlace_handling(png) != 1 ||
      (colorType != PNG_COLOR_TYPE_RGB && colorType != PNG_COLOR_TYPE_RGBA &&
       colorType != PNG_COLOR_TYPE_GRAY)) {
    png_destroy_read_struct(&png, &info, nullptr);
    fclose(fp);
    return false;
  }
  png_read_update_info(png, info);
  size_t channels = colorType == PNG_COLOR_TYPE_GRAY  ? 1
                    : colorType == PNG_COLOR_TYPE_RGB ? 3
                                                      : 4;
  std::vector<png_byte> data(png_get_rowbytes(png, info) * height);
  std::vector<png_bytep> rows(height);
  for (size_t y = 0; y < height; y++) {
    rows[y] = &data[y * png_get_rowbytes(png, info)];
  }
  png_read_image(png, rows.data());
  png_read_end(png, info);
  png_destroy_read_struct(&png, &info, nullptr);
  fclose(fp);

  rgb.resize(width * height * 3);
  for (size_t y = 0; y < height; y++) {
    for (size_t x = 0; x < width; x++) {
      const png_byte *p = rows[y] + x * channels;
      for (size_t c = 0; c < 3; c++) {
        rgb[(y * width + x) * 3 + c] = p[channels == 1 ? 0 : c];
      }
    }
  }
  return true;
}

/// Bilinear resize to INPUT_SIZE x INPUT_SIZE (torch interpolate,
/// align_corners=False), mean subtraction (std is 1/255 on 0..1 pixels, i.e. 1
/// on 0..255) and HWC -> NCHW, written to \p out.
static void preprocess(const std::vector<float> &rgb, size_t width,
                       size_t height, float *out) {
  const float sx = float(width) / INPUT_SIZE;
  const float sy = float(height) / INPUT_SIZE;
  for (size_t oy = 0; oy < INPUT_SIZE; oy++) {
    float fy = std::max((oy + 0.5f) * sy - 0.5f, 0.f);
    size_t y0 = std::min(size_t(fy), height - 1);
    size_t y1 = std::min(y0 + 1, height - 1);
    float wy = fy - y0;
    for (size_t ox = 0; ox < INPUT_SIZE; ox++) {
      float fx = std::max((ox + 0.5f) * sx - 0.5f, 0.f);
      size_t x0 = std::min(size_t(fx), width - 1);
      size_t x1 = std::min(x0 + 1, width - 1);
      float wx = fx - x0;
      for (size_t c = 0; c < 3; c++) {
        auto px = [&](size_t y, size_t x) { return rgb[(y * width + x) * 3 + c]; };
        float v = (1 - wy) * ((1 - wx) * px(y0, x0) + wx * px(y0, x1)) +
                  wy * ((1 - wx) * px(y1, x0) + wx * px(y1, x1));
        out[(c * INPUT_SIZE + oy) * INPUT_SIZE + ox] = v - kMean[c];
      }
    }
  }
}

//===----------------------------------------------------------------------===//
//                   Post-processing (torchvision SSD)
//===----------------------------------------------------------------------===//

/// Default boxes of SSD300 in pixels, in the order of the network output:
/// feature map, then row, then column, then box shape.
static std::vector<Box> defaultBoxes() {
  static const int fmap[6] = {38, 19, 10, 5, 3, 1};
  static const float steps[6] = {8, 16, 32, 64, 100, 300};
  static const float scales[7] = {0.07f, 0.15f, 0.33f, 0.51f,
                                  0.69f, 0.87f, 1.05f};
  static const std::vector<float> ratios[6] = {{2}, {2, 3}, {2, 3},
                                               {2, 3}, {2}, {2}};
  std::vector<Box> boxes;
  for (int k = 0; k < 6; k++) {
    float s = scales[k], sp = sqrtf(scales[k] * scales[k + 1]);
    std::vector<std::pair<float, float>> wh = {{s, s}, {sp, sp}};
    for (float r : ratios[k]) {
      float q = sqrtf(r);
      wh.push_back({s * q, s / q});
      wh.push_back({s / q, s * q});
    }
    float f = INPUT_SIZE / steps[k];
    for (int i = 0; i < fmap[k]; i++) {
      for (int j = 0; j < fmap[k]; j++) {
        float cx = (j + 0.5f) / f, cy = (i + 0.5f) / f;
        for (auto &p : wh) {
          float w = std::min(std::max(p.first, 0.f), 1.f);
          float h = std::min(std::max(p.second, 0.f), 1.f);
          boxes.push_back({(cx - w / 2) * INPUT_SIZE, (cy - h / 2) * INPUT_SIZE,
                           (cx + w / 2) * INPUT_SIZE, (cy + h / 2) * INPUT_SIZE});
        }
      }
    }
  }
  assert(boxes.size() == NUM_ANCHORS);
  return boxes;
}

/// BoxCoder.decode_single with weights (10, 10, 5, 5), clipped to the input.
static Box decode(const float *rel, const Box &a) {
  const float clip = logf(1000.f / 16);
  float w = a.x2 - a.x1, h = a.y2 - a.y1;
  float cx = a.x1 + 0.5f * w, cy = a.y1 + 0.5f * h;
  float dx = rel[0] / 10, dy = rel[1] / 10;
  float dw = std::min(rel[2] / 5, clip), dh = std::min(rel[3] / 5, clip);
  float pcx = dx * w + cx, pcy = dy * h + cy;
  float pw = expf(dw) * w, ph = expf(dh) * h;
  auto clamp = [](float v) { return std::min(std::max(v, 0.f), float(INPUT_SIZE)); };
  return {clamp(pcx - 0.5f * pw), clamp(pcy - 0.5f * ph), clamp(pcx + 0.5f * pw),
          clamp(pcy + 0.5f * ph)};
}

static float iou(const Box &a, const Box &b) {
  float iw = std::min(a.x2, b.x2) - std::max(a.x1, b.x1);
  float ih = std::min(a.y2, b.y2) - std::max(a.y1, b.y1);
  if (iw <= 0 || ih <= 0) {
    return 0;
  }
  float inter = iw * ih;
  float areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
  float areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
  return inter / (areaA + areaB - inter);
}

static std::vector<Detection> postprocess(const float *out, size_t width,
                                          size_t height) {
  static const std::vector<Box> anchors = defaultBoxes();
  std::vector<float> probs(NUM_ANCHORS * NUM_CLASSES);
  for (size_t a = 0; a < NUM_ANCHORS; a++) {
    const float *logit = out + a * NUM_OUT;
    float mx = *std::max_element(logit, logit + NUM_CLASSES), sum = 0;
    for (int c = 0; c < NUM_CLASSES; c++) {
      sum += probs[a * NUM_CLASSES + c] = expf(logit[c] - mx);
    }
    for (int c = 0; c < NUM_CLASSES; c++) {
      probs[a * NUM_CLASSES + c] /= sum;
    }
  }

  std::vector<Detection> all;
  std::vector<Box> decoded(NUM_ANCHORS);
  std::vector<bool> isDecoded(NUM_ANCHORS, false);
  for (int c = 1; c < NUM_CLASSES; c++) {
    std::vector<Detection> cand;
    for (size_t a = 0; a < NUM_ANCHORS; a++) {
      float s = probs[a * NUM_CLASSES + c];
      if (s > kScoreThresh) {
        if (!isDecoded[a]) {
          decoded[a] = decode(out + a * NUM_OUT + NUM_CLASSES, anchors[a]);
          isDecoded[a] = true;
        }
        cand.push_back({decoded[a], s, c});
      }
    }
    auto byScore = [](const Detection &x, const Detection &y) {
      return x.score > y.score;
    };
    std::stable_sort(cand.begin(), cand.end(), byScore);
    if (cand.size() > kTopKCandidates) {
      cand.resize(kTopKCandidates);
    }
    // NMS within the class (batched_nms never suppresses across labels).
    std::vector<bool> removed(cand.size(), false);
    for (size_t i = 0; i < cand.size(); i++) {
      if (removed[i]) {
        continue;
      }
      all.push_back(cand[i]);
      for (size_t j = i + 1; j < cand.size(); j++) {
        if (!removed[j] && iou(cand[i].box, cand[j].box) > kNmsThresh) {
          removed[j] = true;
        }
      }
    }
  }
  std::stable_sort(all.begin(), all.end(),
                   [](const Detection &x, const Detection &y) {
                     return x.score > y.score;
                   });
  if (all.size() > kDetectionsPerImg) {
    all.resize(kDetectionsPerImg);
  }
  const float rx = float(width) / INPUT_SIZE, ry = float(height) / INPUT_SIZE;
  for (auto &d : all) {
    d.box = {d.box.x1 * rx, d.box.y1 * ry, d.box.x2 * rx, d.box.y2 * ry};
  }
  return all;
}

//===----------------------------------------------------------------------===//
//                 Wrapper code for executing a bundle
//===----------------------------------------------------------------------===//

static const SymbolTableEntry &getMutableWeightVar(const BundleConfig &config,
                                                   const char *name) {
  for (unsigned i = 0, e = config.numSymbols; i < e; ++i) {
    if (!strcmp(config.symbolTable[i].name, name)) {
      assert(config.symbolTable[i].kind != 0 && "Expected a mutable variable");
      return config.symbolTable[i];
    }
  }
  fprintf(stderr, "Expected to find variable '%s'\n", name);
  exit(1);
}

static uint8_t *alignedAlloc(const BundleConfig &config, size_t size) {
  void *ptr = nullptr;
  if (posix_memalign(&ptr, config.alignment, size) != 0) {
    fprintf(stderr, "posix_memalign failed\n");
    exit(1);
  }
  memset(ptr, 0, size);
  return static_cast<uint8_t *>(ptr);
}

static uint8_t *initConstantWeights(const char *fileName,
                                    const BundleConfig &config) {
  FILE *f = fopen(fileName, "rb");
  if (!f) {
    fprintf(stderr, "Could not open the weights file: %s\n", fileName);
    exit(1);
  }
  fseek(f, 0, SEEK_END);
  size_t size = ftell(f);
  fseek(f, 0, SEEK_SET);
  if (size != config.constantWeightVarsMemSize) {
    fprintf(stderr, "Wrong weights file size: %zu, expected %" PRIu64 "\n",
            size, config.constantWeightVarsMemSize);
    exit(1);
  }
  uint8_t *weights = alignedAlloc(config, size);
  if (fread(weights, size, 1, f) != 1) {
    fprintf(stderr, "Could not read the weights file\n");
    exit(1);
  }
  fclose(f);
  return weights;
}

int main(int argc, char **argv) {
  if (argc != 2 && argc != 7) {
    fprintf(stderr, "usage: %s image.png [label x1 y1 x2 y2]\n", argv[0]);
    return 2;
  }
  std::vector<float> rgb;
  size_t width = 0, height = 0;
  if (!readPng(argv[1], rgb, width, height)) {
    fprintf(stderr, "Could not read the PNG image: %s\n", argv[1]);
    return 2;
  }
  printf("Loaded image: %s (%zux%zu)\n", argv[1], width, height);

  const BundleConfig &config = ssd300_vgg16_config;
  uint8_t *constantWeights = initConstantWeights("ssd300_vgg16.weights.bin", config);
  uint8_t *mutableWeights = alignedAlloc(config, config.mutableWeightVarsMemSize);
  uint8_t *activations = alignedAlloc(config, config.activationsMemSize);

  const SymbolTableEntry &input = getMutableWeightVar(config, "input");
  const SymbolTableEntry &output = getMutableWeightVar(config, "output");
  assert(input.size == 3 * INPUT_SIZE * INPUT_SIZE);
  assert(output.size == NUM_ANCHORS * NUM_OUT);
  preprocess(rgb, width, height,
             reinterpret_cast<float *>(mutableWeights + input.offset));

  int errCode = ssd300_vgg16(constantWeights, mutableWeights, activations);
  if (errCode != GLOW_SUCCESS) {
    fprintf(stderr, "Error running bundle: error code %d\n", errCode);
    return 1;
  }
  const float *out = reinterpret_cast<float *>(mutableWeights + output.offset);

  if (const char *dumpPath = getenv("SSD_DUMP_OUTPUT")) {
    FILE *f = fopen(dumpPath, "wb");
    if (f) {
      fwrite(out, sizeof(float), output.size, f);
      fclose(f);
    }
  }

  std::vector<Detection> dets = postprocess(out, width, height);
  printf("Detections (score >= %.1f):\n", kPrintThresh);
  for (const auto &d : dets) {
    if (d.score >= kPrintThresh) {
      printf("  %-14s label %2d  score %.3f  box %.1f %.1f %.1f %.1f\n",
             kCocoNames[d.label], d.label, d.score, d.box.x1, d.box.y1,
             d.box.x2, d.box.y2);
    }
  }

  int ret = 0;
  if (argc == 7) {
    int label = atoi(argv[2]);
    Box expected = {float(atof(argv[3])), float(atof(argv[4])),
                    float(atof(argv[5])), float(atof(argv[6]))};
    float best = 0;
    for (const auto &d : dets) {
      if (d.label == label && d.score >= 0.5f) {
        best = std::max(best, iou(d.box, expected));
      }
    }
    printf("Expected %s: best IoU %.3f -> %s\n", kCocoNames[label], best,
           best >= 0.5f ? "PASS" : "FAIL");
    ret = best >= 0.5f ? 0 : 1;
  }

  free(activations);
  free(mutableWeights);
  free(constantWeights);
  return ret;
}
