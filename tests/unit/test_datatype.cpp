#include <gtest/gtest.h>

#include <stdexcept>

#include "onnxcc/ir/datatype.h"

// Test for dtype_size()
// Sizes in bytes of one element for each defined type
TEST(DataTypeTest, DtypeSizeReturnsCorrectSizes) {
    EXPECT_EQ(onnxcc::dtype_size(onnxcc::DataType::FLOAT32), 4);
    EXPECT_EQ(onnxcc::dtype_size(onnxcc::DataType::INT32), 4);
    EXPECT_EQ(onnxcc::dtype_size(onnxcc::DataType::INT64), 8);
    EXPECT_EQ(onnxcc::dtype_size(onnxcc::DataType::BOOL), 1);
    EXPECT_EQ(onnxcc::dtype_size(onnxcc::DataType::FLOAT64), 8);
}

// An undefined type has no meaningful size, so dtype_size() throws
// (TensorShape::num_bytes() relies on it never returning 0)
TEST(DataTypeTest, DtypeSizeUndefinedThrowsException) {
    EXPECT_THROW(
        onnxcc::dtype_size(onnxcc::DataType::UNDEFINED),
        std::logic_error
    );
}


// Test for dtype_name()
// Every defined type maps to its enumerator name
TEST(DataTypeTest, ReturnsCorrectNames) {
    EXPECT_EQ(onnxcc::dtype_name(onnxcc::DataType::FLOAT32), "FLOAT32");
    EXPECT_EQ(onnxcc::dtype_name(onnxcc::DataType::INT32), "INT32");
    EXPECT_EQ(onnxcc::dtype_name(onnxcc::DataType::INT64), "INT64");
    EXPECT_EQ(onnxcc::dtype_name(onnxcc::DataType::BOOL), "BOOL");
    EXPECT_EQ(onnxcc::dtype_name(onnxcc::DataType::FLOAT64), "FLOAT64");
}

// dtype_name() is for diagnostics, so UNDEFINED must return its name instead of throwing
// (contrast with dtype_size(), which throws for UNDEFINED)
TEST(DataTypeTest, DtypeNameReturnsUndefinedForUndefinedType)
{
    EXPECT_NO_THROW(onnxcc::dtype_name(onnxcc::DataType::UNDEFINED));
    EXPECT_EQ(onnxcc::dtype_name(onnxcc::DataType::UNDEFINED), "UNDEFINED");
}