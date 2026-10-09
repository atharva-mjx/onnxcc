add_executable(unit_tests
    unit/test_sanity.cpp
    unit/test_cli.cpp
    unit/test_parse.cpp
    unit/test_headers_compile.cpp
    unit/test_status.cpp
)

# link the onnxcc_lib library to the unit_tests executable.
# link the gtest_main library to the unit_tests executable so that the main function for the tests is provided by gtest_main.
target_link_libraries(unit_tests
    PRIVATE
        onnxcc_lib
        GTest::gtest_main
)

# Make the directory path available to the executable as a preprocessor macro because ctest runs from build/ while relative paths are relative to source dir
target_compile_definitions(unit_tests
    PRIVATE
        ONNXCC_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures"
) 

include(GoogleTest)
# run the exe in background and discover all the tests in the unit_tests executable and register them with CTest. 
gtest_discover_tests(unit_tests)

target_compile_options(unit_tests PRIVATE ${ONNXCC_WARNINGS})

# GLOB_RECURSE recurses the folders to find all the headers files and CONFIGURE_DEPENDS watch the folders if any .h file is added or removed
file(GLOB_RECURSE ONNXCC_PUBLIC_HEADERS CONFIGURE_DEPENDS
    ${CMAKE_SOURCE_DIR}/src/onnxcc/ir/*.h
    ${CMAKE_SOURCE_DIR}/src/onnxcc/memory/*.h
    ${CMAKE_SOURCE_DIR}/src/onnxcc/operators/*.h
)

# create a dummy .cpp file and compile it to check
set(ONNXCC_SELFCHECK_SOURCES "")
foreach(_hdr IN LISTS ONNXCC_PUBLIC_HEADERS)
    file(RELATIVE_PATH _rel ${CMAKE_SOURCE_DIR}/src ${_hdr})
    string(MAKE_C_IDENTIFIER ${_rel} _stem)
    set(_gen ${CMAKE_CURRENT_BINARY_DIR}/selfcheck/${_stem}.cpp)
    file(GENERATE OUTPUT ${_gen} CONTENT "#include \"${_rel}\"\n")
    list(APPEND ONNXCC_SELFCHECK_SOURCES ${_gen})
endforeach()

# build all those dummy files, generate object files and skip the linker to expose any hidden dependencies
add_library(header_selfcheck OBJECT ${ONNXCC_SELFCHECK_SOURCES})
# For onnxcc_lib's PUBLIC include directories. Nothing is linked so it never produces a binary
target_link_libraries(header_selfcheck PRIVATE onnxcc_lib)
target_compile_options(header_selfcheck PRIVATE ${ONNXCC_WARNINGS})