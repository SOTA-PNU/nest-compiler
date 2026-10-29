# RepVGG-A0 with every eligible conv on VTA in int8 (no float layers).
# Same model, calibration and -vta-round-requant as RepVGGA0Test, but without
# VTASkipQuantizeNodes.txt: only the stem conv (3 input channels, not a
# multiple of 16) stays on CPU int8. Kept next to RepVGGA0Test to compare the
# all-VTA result with the mixed VTA/CPU one.
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
)

add_custom_target(vtaRepVGGA0AllVTABundleNet DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/repvgg_a0.cpp ${CMAKE_CURRENT_BINARY_DIR}/repvgg_a0.weights.bin)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})
add_executable(vtaRepVGGA0AllVTABundle ${CMAKE_CURRENT_SOURCE_DIR}/../RepVGGA0Test/repvgg_a0BundleMain.cpp ${CMAKE_CURRENT_BINARY_DIR}/repvgg_a0.cpp)
add_dependencies(vtaRepVGGA0AllVTABundle vtaRepVGGA0AllVTABundleNet ${BUNDLE_DEPENDS})
target_link_libraries(vtaRepVGGA0AllVTABundle VTABundle png ${BUNDLE_LINK_LIB})
set_target_properties(vtaRepVGGA0AllVTABundle
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        )

add_nestc_test(ZCU102 NAME vtaRepVGGA0AllVTABundle COMMAND ${CMAKE_CURRENT_BINARY_DIR}/vtaRepVGGA0AllVTABundle USE_SH 1 PARAMS ${GLOW_SOURCE_DIR}/tests/images/imagenet/dog_207.png)
