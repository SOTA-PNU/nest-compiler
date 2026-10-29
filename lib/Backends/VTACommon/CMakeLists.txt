# Code shared by the VTA and VTAInterpreter backends. Not a backend itself.
add_library(VTACommon VTARequantRounding.cpp)
target_link_libraries(VTACommon
                      PRIVATE
                        Base
                        Graph
                        LLVMSupport)
