option(NEST_BUILD_ON_MAC "Set configurations for building in MAC environment." OFF)
# SSD300-VGG16 object detector (torchvision ssd300_vgg16, COCO, mAP 25.1).
# The ONNX holds only the network (see export_onnx.py); main.cpp does the
# resize, box decoding and NMS.
set(MODEL_INPUT_NAME "input")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${NESTC_BINARY_DIR}/bundles)
set(BUNDLE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

add_custom_target(SSD300VGG16BundleDir ALL
  COMMAND ${CMAKE_COMMAND} -E make_directory ${BUNDLE_OUTPUT_DIRECTORY}
)

add_custom_command(
        OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/ssd300_vgg16.onnx
        COMMAND
        curl -fsSL --retry 3 -o ${CMAKE_CURRENT_BINARY_DIR}/ssd300_vgg16.onnx ${NESTC_MODEL_URL}/ssd300_vgg16.onnx
)

add_custom_target(SSD300VGG16BundleONNX
        DEPENDS
        ${CMAKE_CURRENT_BINARY_DIR}/ssd300_vgg16.onnx
        )
# Final Executables
# =================
# Regular
add_executable(SSD300VGG16Bundle $<TARGET_OBJECTS:SSD300VGG16BundleMain>)
set_target_properties(SSD300VGG16Bundle PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${BUNDLE_OUTPUT_DIRECTORY})
if(NOT NEST_BUILD_ON_MAC)
target_link_libraries(SSD300VGG16Bundle ${BUNDLE_OUTPUT_DIRECTORY}/ssd300_vgg16.o png)
else()
target_link_libraries(SSD300VGG16Bundle ${BUNDLE_OUTPUT_DIRECTORY}/ssd300_vgg16.o /usr/local/lib/libpng.a /usr/local/Cellar/zlib/1.2.11/lib/libz.a)
endif()

add_dependencies(SSD300VGG16Bundle SSD300VGG16BundleMain)

# Glow Bundles
# ============
# Regular Bundle
add_custom_command(
  OUTPUT
    ${BUNDLE_OUTPUT_DIRECTORY}/ssd300_vgg16.o
  COMMAND
    model-compiler -g -model=${CMAKE_CURRENT_BINARY_DIR}/ssd300_vgg16.onnx
    -model-input=${MODEL_INPUT_NAME},float,[1,3,300,300]
 #   -target=aarch64 -mcpu=cortex-a53
    -backend=CPU -emit-bundle=${BUNDLE_OUTPUT_DIRECTORY}
    -bundle-api=dynamic
  DEPENDS
    model-compiler SSD300VGG16BundleDir SSD300VGG16BundleONNX
)
add_custom_target(SSD300VGG16BundleNet DEPENDS ${BUNDLE_OUTPUT_DIRECTORY}/ssd300_vgg16.o ${BUNDLE_OUTPUT_DIRECTORY}/ssd300_vgg16.o)

# Other
# =====
# Driver program with main function for regular bundle
add_library(SSD300VGG16BundleMain OBJECT main.cpp)
if(NOT NEST_BUILD_ON_MAC)
target_compile_options(SSD300VGG16BundleMain PRIVATE -std=c++11 -O2 -g)
else()
target_compile_options(SSD300VGG16BundleMain PRIVATE -std=c++11 -g -I/usr/local/include -L/usr/local/lib -w -lz -lpng)
endif()
target_include_directories(SSD300VGG16BundleMain PUBLIC ${BUNDLE_OUTPUT_DIRECTORY})
add_dependencies(SSD300VGG16BundleMain SSD300VGG16BundleNet)

# dog_207.png: expects a dog (COCO 18) with score >= 0.5 at IoU >= 0.5 with
# the box torchvision's SSD finds in float. PARAMS is one string because
# add_nestc_test pastes it into "sh -c" and list separators would split it.
add_nestc_test(NAME SSD300VGG16Bundle COMMAND ${CMAKE_CURRENT_BINARY_DIR}/SSD300VGG16Bundle USE_SH 1 PARAMS "${GLOW_SOURCE_DIR}/tests/images/imagenet/dog_207.png 18 4.5 33.5 204.3 224.0")
