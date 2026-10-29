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
#ifndef NESTC_BACKENDS_VTACOMMON_VTAREQUANTROUNDING_H
#define NESTC_BACKENDS_VTACOMMON_VTAREQUANTROUNDING_H

namespace glow {

class Function;
class ConvolutionNode;

/// VTA rescales the int32 accumulator of a convolution with an arithmetic
/// right shift, which truncates. Every layer then loses half an output step
/// on average, and the bias adds up in deep plain networks such as RepVGG.
///
/// With -vta-round-requant, adds 2^(shift-1) to the int32 bias of \p CN so
/// the same shift rounds to nearest. This costs nothing at run time, works on
/// the hardware unchanged, and keeps VTAInterpreter and VTA bundles in sync.
/// Only convolutions that take the VTA path are changed; the CPU fallback
/// already rounds. \returns true if \p CN was changed.
bool foldVTARequantRounding(Function *F, ConvolutionNode &CN);

} // namespace glow

#endif // NESTC_BACKENDS_VTACOMMON_VTAREQUANTROUNDING_H
