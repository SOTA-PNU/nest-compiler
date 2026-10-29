option(NEST_BUILD_ON_MAC "Set configurations for building in MAC environment." OFF)
# RepVGG-A0 (timm repvgg_a0.rvgg_in1k), re-parameterized for inference.
# Exported to ONNX (opset 11): only Conv, Relu, GlobalAveragePool, Flatten, Gemm.
set(MODEL_INPUT_NAME "input")
set(IMAGES ${GLOW_SOURCE_DIR}/tests/images/imagenet)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${NESTC_BINARY_DIR}/bundles)
set(BUNDLE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

add_custom_target(RepVGGA0BundleDir ALL
  COMMAND ${CMAKE_COMMAND} -E make_directory ${BUNDLE_OUTPUT_DIRECTORY}
)

add_custom_command(
        OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/repvgg_a0.onnx
        COMMAND
        curl -fsSL --retry 3 -o ${CMAKE_CURRENT_BINARY_DIR}/repvgg_a0.onnx ${NESTC_MODEL_URL}/repvgg_a0.onnx
)

add_custom_target(RepVGGA0BundleONNX
        DEPENDS
        ${CMAKE_CURRENT_BINARY_DIR}/repvgg_a0.onnx
        )
# Final Executables
# =================
# Regular
add_executable(RepVGGA0Bundle $<TARGET_OBJECTS:RepVGGA0BundleMain>)
set_target_properties(RepVGGA0Bundle PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${BUNDLE_OUTPUT_DIRECTORY})
if(NOT NEST_BUILD_ON_MAC)
target_link_libraries(RepVGGA0Bundle ${BUNDLE_OUTPUT_DIRECTORY}/repvgg_a0.o png)
else()
target_link_libraries(RepVGGA0Bundle ${BUNDLE_OUTPUT_DIRECTORY}/repvgg_a0.o /usr/local/lib/libpng.a /usr/local/Cellar/zlib/1.2.11/lib/libz.a)
endif()

add_dependencies(RepVGGA0Bundle RepVGGA0BundleMain)

# Glow Bundles
# ============
# Regular Bundle
add_custom_command(
  OUTPUT
    ${BUNDLE_OUTPUT_DIRECTORY}/repvgg_a0.o
  COMMAND
    model-compiler -g -model=${CMAKE_CURRENT_BINARY_DIR}/repvgg_a0.onnx
    -model-input=${MODEL_INPUT_NAME},float,[1,3,224,224]
 #   -target=aarch64 -mcpu=cortex-a53
    -backend=CPU -emit-bundle=${BUNDLE_OUTPUT_DIRECTORY}
    -bundle-api=dynamic
  DEPENDS
    model-compiler RepVGGA0BundleDir RepVGGA0BundleONNX
)
add_custom_target(RepVGGA0BundleNet DEPENDS ${BUNDLE_OUTPUT_DIRECTORY}/repvgg_a0.o ${BUNDLE_OUTPUT_DIRECTORY}/repvgg_a0.o)

# Other
# =====
# Driver program with main function for regular bundle
add_library(RepVGGA0BundleMain OBJECT main.cpp)
if(NOT NEST_BUILD_ON_MAC)
target_compile_options(RepVGGA0BundleMain PRIVATE -std=c++11 -g)
else()
target_compile_options(RepVGGA0BundleMain PRIVATE -std=c++11 -g -I/usr/local/include -L/usr/local/lib -w -lz -lpng)
endif()
target_include_directories(RepVGGA0BundleMain PUBLIC ${BUNDLE_OUTPUT_DIRECTORY})
add_dependencies(RepVGGA0BundleMain RepVGGA0BundleNet)

add_nestc_test(NAME RepVGGA0Bundle COMMAND ${CMAKE_CURRENT_BINARY_DIR}/RepVGGA0Bundle USE_SH 1 PARAMS ${GLOW_SOURCE_DIR}/tests/images/imagenet/dog_207.png)