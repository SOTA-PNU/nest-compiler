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
#include "VTARequantRounding.h"

#include "glow/Graph/Graph.h"
#include "glow/Graph/Nodes.h"

#include "llvm/Support/CommandLine.h"

#include <algorithm>
#include <cmath>
#include <limits>

using namespace glow;

// Defined once here and linked into both the VTA and VTAInterpreter backends.
// Registering the same option from each backend would abort at startup.
static llvm::cl::opt<bool> vtaRoundRequantOpt(
    "vta-round-requant",
    llvm::cl::desc("Round to nearest instead of truncating when VTA "
                   "convolutions rescale int32 accumulators to int8, by "
                   "folding 2^(shift-1) into the int32 bias"),
    llvm::cl::init(false));

bool glow::foldVTARequantRounding(Function *F, ConvolutionNode &CN) {
  if (!vtaRoundRequantOpt) {
    return false;
  }

  TypeRef inTy = CN.getInput().getType();
  TypeRef filterTy = CN.getFilter().getType();
  TypeRef outTy = CN.getResult().getType();
  TypeRef biasTy = CN.getBias().getType();
  if (inTy->getElementType() != ElemKind::Int8QTy ||
      filterTy->getElementType() != ElemKind::Int8QTy ||
      outTy->getElementType() != ElemKind::Int8QTy ||
      biasTy->getElementType() != ElemKind::Int32QTy) {
    return false;
  }

  // Same conditions as the VTA path in VTAInterpreterNodes.cpp and
  // VTASave.cpp (NHWC input, group 1, no dilation, channels in blocks of 16,
  // zero offsets, power-of-2 rescale). Scales are compared exactly the way
  // they are there.
  if (CN.getGroup() != 1 || CN.getDilation()[0] != 1 ||
      CN.getDilation()[1] != 1 || inTy->dims()[3] % 16 != 0 ||
      outTy->dims()[3] % 16 != 0) {
    return false;
  }
  if (inTy->getOffset() != 0 || filterTy->getOffset() != 0 ||
      outTy->getOffset() != 0 || biasTy->getOffset() != 0) {
    return false;
  }
  float inScale = 1 / inTy->getScale();
  float filterScale = 1 / filterTy->getScale();
  float biasScale = 1 / biasTy->getScale();
  float outScale = 1 / outTy->getScale();
  float matMulScale = inScale * filterScale;
  float scale = matMulScale / outScale;
  if (biasScale != matMulScale || !(scale > 1)) {
    return false;
  }
  unsigned shift = 0;
  float tempScale = 1.0;
  while (tempScale < scale) {
    tempScale *= 2;
    shift++;
  }
  if (tempScale != scale) {
    return false;
  }

  auto *biasC = llvm::dyn_cast<Constant>(CN.getBias().getNode());
  if (!biasC) {
    return false;
  }

  // Work on a copy: the bias constant may be shared with other nodes.
  Constant *rounded = F->getParent()->createConstant(
      biasC->getType(), biasC->getName().str() + "_vta_round");
  rounded->getPayloadMutable().assign(&biasC->getPayload());
  auto H = rounded->getPayloadMutable().getHandle<int32_t>();
  const int64_t half = int64_t(1) << (shift - 1);
  for (dim_t i = 0, e = H.size(); i < e; i++) {
    int64_t v = int64_t(H.raw(i)) + half;
    H.raw(i) = int32_t(std::min<int64_t>(v, std::numeric_limits<int32_t>::max()));
  }
  CN.setNthInput(ConvolutionNode::BiasIdx, rounded);
  return true;
}

bool glow::alignVTAConvBiasScale(Function *F, ConvolutionNode &CN) {
  TypeRef inTy = CN.getInput().getType();
  TypeRef filterTy = CN.getFilter().getType();
  TypeRef biasTy = CN.getBias().getType();
  if (inTy->getElementType() != ElemKind::Int8QTy ||
      filterTy->getElementType() != ElemKind::Int8QTy ||
      biasTy->getElementType() != ElemKind::Int32QTy ||
      biasTy->getOffset() != 0) {
    return false;
  }
  // The same product quantization uses (specializeBiasQuantizationParams).
  float matMulScale = inTy->getScale() * filterTy->getScale();
  if (biasTy->getScale() == matMulScale) {
    return false;
  }
  auto *biasC = llvm::dyn_cast<Constant>(CN.getBias().getNode());
  if (!biasC) {
    return false;
  }

  // Work on a copy: the bias constant may be shared with other nodes.
  TypeRef newTy = F->getParent()->uniqueType(ElemKind::Int32QTy, biasTy->dims(),
                                             matMulScale, 0);
  Constant *aligned = F->getParent()->createConstant(
      newTy, biasC->getName().str() + "_vta_bias");
  auto src = biasC->getPayload().getHandle<int32_t>();
  auto dst = aligned->getPayloadMutable().getHandle<int32_t>();
  const double ratio = double(biasTy->getScale()) / matMulScale;
  for (dim_t i = 0, e = src.size(); i < e; i++) {
    double v = std::round(src.raw(i) * ratio);
    v = std::min<double>(v, std::numeric_limits<int32_t>::max());
    v = std::max<double>(v, std::numeric_limits<int32_t>::min());
    dst.raw(i) = int32_t(v);
  }
  CN.setNthInput(ConvolutionNode::BiasIdx, aligned);
  return true;
}
