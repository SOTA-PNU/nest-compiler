# RepVGG-A0 on VTA, with a few layers left in float on the CPU.
#
# Plain RepVGG loses a lot of accuracy under int8 post-training quantization,
# and VTA (per-tensor power-of-2 scales, truncating requantization) makes it
# worse. Four things together keep the result correct:
# - repvgg_a0.onnx is exported with cross-layer equalization
#   (examples/bundles/repvgg_a0/export_onnx.py), so per-tensor weight scales fit.
# - The profile keeps a 2048-bin histogram and the compiler uses KL
#   calibration, which clips long activation tails.
# - -vta-round-requant folds rounding into the conv biases.
# - VTASkipQuantizeNodes.txt keeps the three most sensitive conv+relu layers
#   (stages.0.0, stages.2.6, stages.2.7) in float on the CPU.
#
# Calibration uses the same RGB ImageNet mean/std normalization as the driver:
# Glow computes ((pixel - mean) / stddev) * (1 / 255) for -image-mode=0to1.
set(REPVGG_A0_ONNX ${CMAKE_CURRENT_BINARY_DIR}/repvgg_a0.onnx)
set(REPVGG_A0_PROFILE ${CMAKE_CURRENT_BINARY_DIR}/repvgg_a0_calib.yaml)
set(REPVGG_A0_IMAGES
        ${GLOW_SOURCE_DIR}/tests/images/imagenet/cat_285.png
        ${GLOW_SOURCE_DIR}/tests/images/imagenet/dog_207.png
        ${GLOW_SOURCE_DIR}/tests/images/imagenet/zebra_340.png)

add_custom_command(
        OUTPUT
        ${REPVGG_A0_ONNX}
        COMMAND
        curl -fsSL --retry 3 -o ${REPVGG_A0_ONNX} ${NESTC_MODEL_URL}/repvgg_a0.onnx
)

add_custom_command(
        OUTPUT
        ${REPVGG_A0_PROFILE}
        COMMAND
        image-classifier ${REPVGG_A0_IMAGES}
        -m=${REPVGG_A0_ONNX}
        -model-input-name=input
        -backend=VTAInterpreter
        -image-layout=NCHW
        -image-channel-order=RGB
        -image-mode=0to1
        -mean=123.675,116.28,103.53
        -stddev=0.229,0.224,0.225
        -minibatch=1
        -compute-softmax
        -topk=5
        -dump-profile=${REPVGG_A0_PROFILE}
        -num-histogram-bins=2048
        -quantization-schema=symmetric_with_power2_scale
        DEPENDS image-classifier ${REPVGG_A0_ONNX}
)

add_custom_command(
        OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/repvgg_a0.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/repvgg_a0.h
        ${CMAKE_CURRENT_BINARY_DIR}/repvgg_a0.weights.bin
        ${CMAKE_CURRENT_BINARY_DIR}/VTARuntime.h
        COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_CURRENT_SOURCE_DIR}/VTASkipQuantizeNodes.txt
        ${CMAKE_CURRENT_BINARY_DIR}/VTASkipQuantizeNodes.txt
        COMMAND
        model-compiler -g
        -model=${REPVGG_A0_ONNX}
        -backend=VTA
        -emit-bundle=${CMAKE_CURRENT_BINARY_DIR}
        -bundle-api=dynamic
        -model-input=input,float,[1,3,224,224]
        -load-profile=${REPVGG_A0_PROFILE}
        -quantization-schema=symmetric_with_power2_scale
        -quantization-calibration=KL
        -vta-round-requant
        DEPENDS
        model-compiler VTA ${REPVGG_A0_PROFILE}
        ${CMAKE_CURRENT_SOURCE_DIR}/VTASkipQuantizeNodes.txt
)

add_custom_target(vtaRepVGGA0BundleNet DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/repvgg_a0.cpp ${CMAKE_CURRENT_BINARY_DIR}/repvgg_a0.weights.bin)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})
add_executable(vtaRepVGGA0Bundle repvgg_a0BundleMain.cpp ${CMAKE_CURRENT_BINARY_DIR}/repvgg_a0.cpp)
add_dependencies(vtaRepVGGA0Bundle vtaRepVGGA0BundleNet ${BUNDLE_DEPENDS})
target_link_libraries(vtaRepVGGA0Bundle VTABundle png ${BUNDLE_LINK_LIB})
set_target_properties(vtaRepVGGA0Bundle
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        )

add_nestc_test(ZCU102 NAME vtaRepVGGA0Bundle COMMAND ${CMAKE_CURRENT_BINARY_DIR}/vtaRepVGGA0Bundle USE_SH 1 PARAMS ${GLOW_SOURCE_DIR}/tests/images/imagenet/dog_207.png)
