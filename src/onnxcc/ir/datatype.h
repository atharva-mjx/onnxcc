#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string_view>


namespace onnxcc {

// std::int32_t ensures the cast from ONNX int is lossless and size is predictable
// Do not renumber, these values must match exactly with ONNX TensorProto::DataType codes
enum class DataType : std::int32_t {
    UNDEFINED = 0, 
    FLOAT32 = 1, 
    INT32 = 6, 
    INT64 = 7, 
    BOOL = 9, 
    FLOAT64 = 11
};

// defined dtype_size in the header for compile time evaluation due to constexpr
// dtype_size never returns 0 for a defined type, TensorShape::num_bytes divides by it
// throws std::logic_error for UNDEFINED or a value outside the enum
constexpr std::size_t dtype_size(DataType dt) {
    switch(dt) {
        case DataType::FLOAT32: return 4;
        case DataType::INT32: return 4;
        case DataType::INT64: return 8;
        case DataType::BOOL: return 1;
        case DataType::FLOAT64: return 8;
        case DataType::UNDEFINED: break;
    }
    // does not require caller to check status
    // if the datatype is not defined/undefined it is programmer's error 
    throw std::logic_error("dtype_size called on UNDEFINED or unknown datatype");
}

// implemented in datatype.cpp to avoid merge conflicts on this core header
// returns a view to a static string literal, valid for the entire program's lifetime
std::string_view dtype_name(DataType dt);

}