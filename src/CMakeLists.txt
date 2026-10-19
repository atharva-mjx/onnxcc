configure_file(onnxcc/version.h.in onnxcc/version.h @ONLY)

add_library(onnxcc_lib STATIC
    onnxcc/version.cpp
    onnxcc/cli/cli.cpp
    onnxcc/cli/parse.cpp
    onnxcc/cli/dump_command.cpp
    onnxcc/ir/attribute.cpp
    onnxcc/ir/datatype.cpp 
    onnxcc/ir/tensor_shape.cpp
)

add_executable(onnxcc onnxcc/main.cpp)

# Link the onnxcc_lib library to the onnxcc executable
target_link_libraries(onnxcc PRIVATE onnxcc_lib)

target_link_libraries(onnxcc_lib
    PRIVATE
        protobuf::libprotobuf
        onnx_proto
)

# Include the current source directory for the onnxcc_lib target
target_include_directories(onnxcc_lib
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_BINARY_DIR}
)

target_compile_options(onnxcc_lib PRIVATE ${ONNXCC_WARNINGS})
target_compile_options(onnxcc PRIVATE ${ONNXCC_WARNINGS})