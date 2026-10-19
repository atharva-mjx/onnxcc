#include "onnxcc/ir/tensor_shape.h"
#include "onnxcc/ir/datatype.h"

// Explicit includes so this file does not rely on transitive includes.
#include <cstddef>    // std::size_t
#include <cstdint>    // std::int64_t
#include <limits>     // std::numeric_limits
#include <stdexcept>  // std::logic_error, std::overflow_error
#include <string>     // std::string, std::to_string

std::size_t onnxcc::TensorShape::num_elements() const
{
    if (is_dynamic())
    {
        throw std::logic_error("cannot determine the number of elements of a dynamic tensor shape");
    }

    // Result is initialized to 1. If dims is empty, the loop never runs and the result would be 1 which is correct for scalar
    std::size_t result = 1;
    // Numeric limits sets max size according to the system.
    std::size_t maxi = std::numeric_limits<std::size_t>::max();

    for (const std::int64_t &dim : dims)
    {
        // We have to check everytime if result*dim > max range and that dim is not 0.
        if (dim != 0 && (maxi / dim) < result)
        {
            throw std::overflow_error("number of tensor elements exceeds the representable range");
        }
        result *= dim;
    }
    return result;
}

std::size_t onnxcc::TensorShape::num_bytes(onnxcc::DataType dt) const
{
    std::size_t elements = num_elements();

    // dtype_size() returns a nonzero size for every defined DataType and throws for UNDEFINED (an undefined type has no size). It never returns 0.
    // The overflow check below divides by data_size. If a zero-sized type is ever added, this would be a division by zero.
    std::size_t data_size = dtype_size(dt);

    // Overflow check in division form: data_size * elements > max implies max / data_size < elements.
    // If elements is 0 this is never true and the result is 0, which is correct.
    if ((std::numeric_limits<std::size_t>::max()) / data_size < elements)
    {
        throw std::overflow_error("number of tensor bytes exceeds the representable range");
    }
    return data_size * elements;
}

std::string onnxcc::TensorShape::to_string() const
{
    std::string result = "[";
    bool first = true;
    for (const std::int64_t &dim : dims)
    {
        if (!first)
        {
            result += ", ";
        }
        first = false;

        if (dim == -1)
        {
            result += "?";
        }
        else
        {
            // Valid non-negative dims, and invalid negatives other than -1, which stay visible.
            result += std::to_string(dim);
        }
    }
    result += "]";
    return result;
}

bool onnxcc::TensorShape::is_dynamic() const noexcept
{
    for (const std::int64_t &dim : dims)
    {
        // If dim < 0, that means that TensorShape is dynamic.
        if (dim < 0)
        {
            return true;
        }
    }
    return false;
}