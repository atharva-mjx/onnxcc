#include "onnxcc/ir/datatype.h"

#include <stdexcept>    // std::logic_error
#include <string_view>  // std::string_view

// This function is intended for diagnostics, logging and error messages, so it
// must be able to describe every enumerator, including DataType::UNDEFINED.

std::string_view onnxcc::dtype_name(onnxcc::DataType dt){
    switch(dt) {
        case onnxcc::DataType::FLOAT32: return "FLOAT32";
        case onnxcc::DataType::INT32: return "INT32";
        case onnxcc::DataType::INT64: return "INT64";
        case onnxcc::DataType::BOOL: return "BOOL";
        case onnxcc::DataType::FLOAT64: return "FLOAT64";
        case onnxcc::DataType::UNDEFINED: return "UNDEFINED";
    }
    // does not require caller to check status
    // UNDEFINED is handled above, so reaching here means dt holds a value outside the enum, which is a programmer's error
    throw std::logic_error("dtype_name called on an unknown datatype");
}