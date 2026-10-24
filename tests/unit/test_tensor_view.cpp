#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "onnxcc/ir/tensor.h"
#include "onnxcc/memory/arena.h"
#include "onnxcc/memory/tensor_view.h"

namespace onnxcc {
namespace {

// a 2x3 float view from the arena should come back as 6 floats with the same shape and dtype
TEST(TensorViewTest, CorrectSpanFromArena) {
    MemoryArena arena(1024);
    TensorShape shape{{2, 3}};
    TensorView view = TensorView::over_arena(arena, shape, DataType::FLOAT32);

    auto span = view.as<float>();
    EXPECT_EQ(span.size(), 6u);
    EXPECT_EQ(view.dtype(), DataType::FLOAT32);
    // works because TensorShape has a defaulted operator<=>
    EXPECT_EQ(view.shape(), shape);
}

// asking for int32 on float data must fail loudly, e.g. 1.5f read as int32 would be 1069547520
TEST(TensorViewTest, RejectsDtypeMismatch) {
    MemoryArena arena(1024);
    TensorShape shape{{2, 3}};
    TensorView view = TensorView::over_arena(arena, shape, DataType::FLOAT32);

    EXPECT_THROW(view.as<std::int32_t>(), std::logic_error);
}

// what you write through the span is really in the arena, a second span reads the same values back
TEST(TensorViewTest, WriteThenReadRoundTrip) {
    MemoryArena arena(1024);
    TensorShape shape{{3}};
    TensorView view = TensorView::over_arena(arena, shape, DataType::INT32);

    auto write_span = view.as<std::int32_t>();
    write_span[0] = 10;
    write_span[1] = 20;
    write_span[2] = 30;

    auto read_span = view.as<std::int32_t>();
    EXPECT_EQ(read_span[0], 10);
    EXPECT_EQ(read_span[1], 20);
    EXPECT_EQ(read_span[2], 30);
}

// TensorView v; is a blank label: no memory, 0 elements, no type
TEST(TensorViewTest, DefaultViewIsEmpty) {
    TensorView view;

    EXPECT_EQ(view.data(), nullptr);
    EXPECT_EQ(view.num_elements(), 0u);
    EXPECT_EQ(view.dtype(), DataType::UNDEFINED);
}

// a shape like [?, 4] has no fixed size, so it can't describe memory that already exists
TEST(TensorViewTest, RejectsDynamicShape) {
    std::byte buffer[16] = {};

    EXPECT_THROW(TensorView(buffer, 16, TensorShape{{-1, 4}}, DataType::FLOAT32), std::logic_error);
}

// "kind: unknown" is refused at the gate, otherwise num_elements() (noexcept) would kill the program
TEST(TensorViewTest, RejectsUndefinedDtype) {
    std::byte buffer[16] = {};

    EXPECT_THROW(TensorView(buffer, 16, TensorShape{{4}}, DataType::UNDEFINED), std::logic_error);
}

#ifndef NDEBUG
// debug builds only: 4 floats need 16 bytes, claiming 12 is a bug and the assert should stop it
// release builds compile the check out, as task 8 asks, so there is nothing to test there
TEST(TensorViewDeathTest, WrongByteCountAssertsInDebug) {
    std::byte buffer[16] = {};

    EXPECT_DEATH(TensorView(buffer, 12, TensorShape{{4}}, DataType::FLOAT32), "byte count");
}
#endif

// a dynamic shape has no fixed byte count, so over_arena rejects it
TEST(TensorViewTest, OverArenaRejectsDynamicShape) {
    MemoryArena arena(1024);

    EXPECT_THROW(
        TensorView::over_arena(arena, TensorShape{{-1, 4}}, DataType::FLOAT32),
        std::logic_error
    );
}

// UNDEFINED has no known element size, so over_arena rejects it
TEST(TensorViewTest, OverArenaRejectsUndefinedDtype) {
    MemoryArena arena(1024);

    EXPECT_THROW(
        TensorView::over_arena(arena, TensorShape{{4}}, DataType::UNDEFINED),
        std::logic_error
    );
}

// shape [0] contains zero elements: the view and its typed span should be empty
TEST(TensorViewTest, OverArenaHandlesZeroElementShape) {
    MemoryArena arena(1024);
    TensorView view =
        TensorView::over_arena(arena, TensorShape{{0}}, DataType::FLOAT32);

    EXPECT_EQ(view.num_elements(), 0u);
    EXPECT_TRUE(view.as<float>().empty());
}

// a 64 byte arena can't fit 100 floats (400 bytes), so over_arena throws instead of handing out null
TEST(TensorViewTest, OverArenaThrowsWhenArenaIsFull) {
    MemoryArena arena(64);

    EXPECT_THROW(TensorView::over_arena(arena, TensorShape{{100}}, DataType::FLOAT32), std::bad_alloc);
}

// every arena backed view starts on a 64 byte boundary, even the second one after a small first
TEST(TensorViewTest, OverArenaIsSixtyFourByteAligned) {
    MemoryArena arena(1024);
    TensorView first = TensorView::over_arena(arena, TensorShape{{3}}, DataType::FLOAT32);
    TensorView second = TensorView::over_arena(arena, TensorShape{{3}}, DataType::FLOAT32);

    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(first.data()) % 64, 0u);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(second.data()) % 64, 0u);
}

// a weight loaded from the model can be read as floats through the view, with no copy made
TEST(TensorViewTest, OverTensorSeesTheInitializerBytes) {
    const std::vector<float> weights = {1.5f, -2.0f, 3.25f, 0.5f};
    Tensor tensor;
    tensor.shape = TensorShape{{2, 2}};
    tensor.dtype = DataType::FLOAT32;
    tensor.data.resize(weights.size() * sizeof(float));
    std::memcpy(tensor.data.data(), weights.data(), tensor.data.size());

    TensorView view = TensorView::over_tensor(tensor);
    auto span = view.as<float>();

    ASSERT_EQ(span.size(), 4u);
    EXPECT_EQ(span[0], 1.5f);
    EXPECT_EQ(span[3], 0.5f);
    // no copy, the view points straight at the vector
    EXPECT_EQ(view.data(), reinterpret_cast<std::byte*>(tensor.data.data()));
}

// a const view hands out a read only span, and the dtype check still works on it
TEST(TensorViewTest, ConstViewGivesReadOnlySpan) {
    MemoryArena arena(1024);
    TensorView view = TensorView::over_arena(arena, TensorShape{{2}}, DataType::INT64);
    view.as<std::int64_t>()[1] = 42;

    const TensorView& read_only = view;
    auto span = read_only.as<std::int64_t>();

    static_assert(std::is_same_v<decltype(span), std::span<const std::int64_t>>);
    EXPECT_EQ(span[1], 42);
    EXPECT_THROW(read_only.as<float>(), std::logic_error);
}

// ONNX bool is 1 byte each, so 5 bools are 5 bytes and every one is reachable
TEST(TensorViewTest, BoolViewIsOneBytePerElement) {
    MemoryArena arena(1024);
    TensorView view = TensorView::over_arena(arena, TensorShape{{5}}, DataType::BOOL);

    auto span = view.as<bool>();
    EXPECT_EQ(span.size(), 5u);
    span[4] = true;
    EXPECT_TRUE(view.as<bool>()[4]);
}

// shape [] is a single number (a scalar), so the view holds exactly 1 element
TEST(TensorViewTest, ScalarHasOneElement) {
    MemoryArena arena(1024);
    TensorView view = TensorView::over_arena(arena, TensorShape{}, DataType::FLOAT64);

    EXPECT_EQ(view.num_elements(), 1u);
    EXPECT_EQ(view.as<double>().size(), 1u);
}

}  // namespace
}  // namespace onnxcc
