# SSD300-VGG16 object detector on VTA (same network and driver as
# examples/bundles/ssd300_vgg16, which runs it in float on the CPU).
#
# - Quantization: per-tensor power-of-2 scales with KL calibration and
#   -vta-round-requant. The profile was collected on 500 COCO val2017 images
#   (utils/coco_val_script/merge_profiles.py describes how) and is
#   downloaded from the model release; collecting it takes hours on the CPU.
# - VTASkipQuantizeNodes.txt keeps the L2 normalization after conv4_3 in
#   float: its int8 division by a zero norm traps.
# - Convolutions VTA cannot run (the first conv, the class heads and the
#   24-channel box heads, whose channel counts are not multiples of 16, and
#   the dilated conv6) run in int8 on the CPU, and the head reshape/concat
#   and the L2 normalization use the CPU operators of vtalib
#   (CPUBundle_common.cpp).
#
# The prebuilt vtalib (NESTC_USE_PRECOMPILED_EVTA_LIBRARY) predates those
# operators, so the bundle is only built with vtalib from source.
if(NESTC_USE_PRECOMPILED_EVTA_LIBRARY)
    return()
endif()

set(SSD300_VGG16_ONNX ${CMAKE_CURRENT_BINARY_DIR}/ssd300_vgg16.onnx)
set(SSD300_VGG16_PROFILE ${CMAKE_CURRENT_BINARY_DIR}/ssd300_vgg16_calib500.yaml)

add_custom_command(
        OUTPUT
        ${SSD300_VGG16_ONNX}
        COMMAND
        curl -fsSL --retry 3 -o ${SSD300_VGG16_ONNX} ${NESTC_MODEL_URL}/ssd300_vgg16.onnx
)

add_custom_command(
        OUTPUT
        ${SSD300_VGG16_PROFILE}
        COMMAND
        curl -fsSL --retry 3 -o ${SSD300_VGG16_PROFILE} ${NESTC_MODEL_URL}/ssd300_vgg16_calib500.yaml
)

add_custom_command(
        OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/ssd300_vgg16.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/ssd300_vgg16.h
        ${CMAKE_CURRENT_BINARY_DIR}/ssd300_vgg16.weights.bin
        ${CMAKE_CURRENT_BINARY_DIR}/VTARuntime.h
        COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_CURRENT_SOURCE_DIR}/VTASkipQuantizeNodes.txt
        ${CMAKE_CURRENT_BINARY_DIR}/VTASkipQuantizeNodes.txt
        COMMAND
        model-compiler -g
        -model=${SSD300_VGG16_ONNX}
        -backend=VTA
        -emit-bundle=${CMAKE_CURRENT_BINARY_DIR}
        -bundle-api=dynamic
        -model-input=input,float,[1,3,300,300]
        -load-profile=${SSD300_VGG16_PROFILE}
        -quantization-schema=symmetric_with_power2_scale
        -quantization-calibration=KL
        -vta-round-requant
        DEPENDS
        model-compiler VTA ${SSD300_VGG16_ONNX} ${SSD300_VGG16_PROFILE}
        ${CMAKE_CURRENT_SOURCE_DIR}/VTASkipQuantizeNodes.txt
)

add_custom_target(vtaSSD300VGG16BundleNet DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/ssd300_vgg16.cpp ${CMAKE_CURRENT_BINARY_DIR}/ssd300_vgg16.weights.bin)
add_executable(vtaSSD300VGG16Bundle ${NESTC_ROOT_DIR}/examples/bundles/ssd300_vgg16/main.cpp ${CMAKE_CURRENT_BINARY_DIR}/ssd300_vgg16.cpp)
target_include_directories(vtaSSD300VGG16Bundle PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
target_compile_definitions(vtaSSD300VGG16Bundle PRIVATE SSD_VTA_BUNDLE)
target_compile_options(vtaSSD300VGG16Bundle PRIVATE -O2)
add_dependencies(vtaSSD300VGG16Bundle vtaSSD300VGG16BundleNet ${BUNDLE_DEPENDS})
target_link_libraries(vtaSSD300VGG16Bundle VTABundle png ${BUNDLE_LINK_LIB})
set_target_properties(vtaSSD300VGG16Bundle
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        )

# Same expectation as SSD300VGG16Bundle: a dog (COCO 18) with score >= 0.5 at
# IoU >= 0.5 with the float detection. PARAMS is one string (see there).
add_nestc_test(ZCU102 NAME vtaSSD300VGG16Bundle COMMAND ${CMAKE_CURRENT_BINARY_DIR}/vtaSSD300VGG16Bundle USE_SH 1 PARAMS "${GLOW_SOURCE_DIR}/tests/images/imagenet/dog_207.png 18 4.5 33.5 204.3 224.0")
