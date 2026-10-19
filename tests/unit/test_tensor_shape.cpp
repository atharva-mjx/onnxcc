
#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <stdexcept>

#include "onnxcc/ir/datatype.h"
#include "onnxcc/ir/tensor_shape.h"

using onnxcc::DataType;
using onnxcc::TensorShape;

// Test for rank 0 tensor(scalar)

TEST(TensorShapeTest, RankZeroIsScalar)
{
    TensorShape shape{};

    EXPECT_TRUE(shape.dims.empty());
    EXPECT_EQ(shape.num_elements(), 1);
    EXPECT_EQ(shape.to_string(), "[]");
    EXPECT_FALSE(shape.is_dynamic());
}

// Rank 1 tensor
TEST(TensorShapeTest, RankOne)
{
    TensorShape shape{{7}};

    EXPECT_EQ(shape.num_elements(), 7);
    EXPECT_EQ(shape.to_string(), "[7]");
    EXPECT_FALSE(shape.is_dynamic());
}

// Rank 4 tensor
TEST(TensorShapeTest, RankFour)
{
    TensorShape shape{{2, 3, 4, 5}};

    EXPECT_EQ(shape.dims.size(), 4);
    EXPECT_EQ(shape.num_elements(), 120);
    EXPECT_EQ(shape.to_string(), "[2, 3, 4, 5]");
    EXPECT_FALSE(shape.is_dynamic());
}

// Every supported datatype: num_bytes()
TEST(TensorShapeTest, NumBytesSupportsEveryDefinedDataType)
{
    TensorShape shape{{2, 3}}; // 6 elements

    EXPECT_EQ(shape.num_bytes(DataType::FLOAT32), 6 * 4);
    EXPECT_EQ(shape.num_bytes(DataType::INT32),   6 * 4);
    EXPECT_EQ(shape.num_bytes(DataType::INT64),   6 * 8);
    EXPECT_EQ(shape.num_bytes(DataType::BOOL),    6 * 1);
    EXPECT_EQ(shape.num_bytes(DataType::FLOAT64), 6 * 8);
}

TEST(TensorShapeTest, NumBytesWithScalar)
{
    TensorShape scalar{};

    EXPECT_EQ(scalar.num_bytes(DataType::FLOAT32), 4);
    EXPECT_EQ(scalar.num_bytes(DataType::BOOL), 1);
    EXPECT_EQ(scalar.num_bytes(DataType::INT64), 8);
}

TEST(TensorShapeTest, UndefinedDataTypeThrows)
{
    TensorShape shape{{2, 3}};

    EXPECT_THROW(
        shape.num_bytes(DataType::UNDEFINED),
        std::logic_error
    );
}

// dtype_size contract: num_bytes() divides by the size, so it must never be 0
// Every defined type must have a nonzero size
TEST(TensorShapeTest, EveryDefinedDataTypeHasNonzeroSize)
{
    for (DataType dt : {DataType::FLOAT32, DataType::INT32, DataType::INT64,
                        DataType::BOOL, DataType::FLOAT64})
    {
        EXPECT_GT(onnxcc::dtype_size(dt), 0u);
    }
}

TEST(TensorShapeTest, DtypeSizeKnownSizes)
{
    EXPECT_EQ(onnxcc::dtype_size(DataType::FLOAT32), 4u);
    EXPECT_EQ(onnxcc::dtype_size(DataType::INT32),   4u);
    EXPECT_EQ(onnxcc::dtype_size(DataType::INT64),   8u);
    EXPECT_EQ(onnxcc::dtype_size(DataType::BOOL),    1u);
    EXPECT_EQ(onnxcc::dtype_size(DataType::FLOAT64), 8u);
}

// UNDEFINED and out-of-range values must throw instead of returning 0
TEST(TensorShapeTest, DtypeSizeThrowsInsteadOfReturningZero)
{
    EXPECT_THROW(onnxcc::dtype_size(DataType::UNDEFINED), std::logic_error);
    EXPECT_THROW(onnxcc::dtype_size(static_cast<DataType>(0x7F)), std::logic_error);
}

// Three-way comparison (<=>)
TEST(TensorShapeTest, EqualShapesCompareEqual)
{
    TensorShape a{{2, 3, 4}};
    TensorShape b{{2, 3, 4}};

    EXPECT_TRUE((a <=> b) == 0);
    EXPECT_TRUE(a == b);
}

TEST(TensorShapeTest, ShapesCompareLexicographically)
{
    TensorShape a{{2, 3}};
    TensorShape b{{2, 4}};
    TensorShape c{{3, 1}};

    EXPECT_TRUE((a <=> b) < 0);
    EXPECT_TRUE((b <=> c) < 0);
    EXPECT_FALSE((a <=> c) > 0);

    EXPECT_TRUE(a < b);
    EXPECT_TRUE(c > b);
}

TEST(TensorShapeTest, DifferentRanksCanBeOrdered)
{
    TensorShape shorter{{2, 3}};
    TensorShape longer{{2, 3, 1}};

    // When one sequence is a prefix of another,
    // the shorter sequence compares less.
    EXPECT_TRUE((shorter <=> longer) < 0);
}

// Dynamic dimensions
TEST(TensorShapeTest, DynamicDimensionIsDetected)
{
    TensorShape shape{{2, -1, 4}};

    EXPECT_TRUE(shape.is_dynamic());
    EXPECT_EQ(shape.to_string(), "[2, ?, 4]");

    EXPECT_THROW(shape.num_elements(), std::logic_error);
}

// Only -1 renders as '?'. Other negative dims are invalid and stay visible
TEST(TensorShapeTest, OtherNegativeDimsStayDistinguishable)
{
    TensorShape shape{{-2, -1, -100}};

    EXPECT_TRUE(shape.is_dynamic());
    EXPECT_EQ(shape.to_string(), "[-2, ?, -100]");
}

// to_string() is a diagnostic and must never throw, even for dynamic shapes
TEST(TensorShapeTest, ToStringDoesNotThrowOnDynamicShape)
{
    TensorShape shape{{-1, 3}};

    EXPECT_NO_THROW(shape.to_string());
    EXPECT_EQ(shape.to_string(), "[?, 3]");
}

// Shapes with one element or zero elements must still print every dim
TEST(TensorShapeTest, ToStringKeepsAllDimsForDegenerateShapes)
{
    TensorShape ones{{1, 1}};
    TensorShape has_zero{{0, 3}};

    EXPECT_EQ(ones.to_string(), "[1, 1]");
    EXPECT_EQ(has_zero.to_string(), "[0, 3]");
}

TEST(TensorShapeTest, ZeroDimensionProducesZeroElements)
{
    TensorShape shape{{3, 0, 4}};

    EXPECT_EQ(shape.num_elements(), 0);
    EXPECT_EQ(shape.num_bytes(DataType::FLOAT32), 0);
}

// Overflow
TEST(TensorShapeTest, NumElementsOverflowThrows)
{
    TensorShape shape{{
        std::numeric_limits<std::int64_t>::max(),
        std::numeric_limits<std::int64_t>::max()
    }};

    EXPECT_THROW(shape.num_elements(), std::overflow_error);
}

TEST(TensorShapeTest, NumBytesOverflowThrows)
{
    // On a 64-bit platform, INT64_MAX elements fit in size_t,
    // but multiplying that count by 8 bytes overflows size_t.
    TensorShape shape{{
        std::numeric_limits<std::int64_t>::max()
    }};

    EXPECT_THROW(
        shape.num_bytes(DataType::FLOAT64),
        std::overflow_error
    );
}