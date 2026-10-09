#include <gtest/gtest.h>
#include <cstdint>
#include <stdexcept>
#include "onnxcc/memory/tensor_view.h"
#include "onnxcc/memory/arena.h"
#include "onnxcc/ir/tensor.h"
namespace onnxcc{
namespace{
    TEST(TensorViewTest, CorrectSpanFromArena){
        MemoryArena arena(1024);
        TensorShape shape{{2, 3}};
        TensorView view = TensorView::over_arena(arena, shape, DataType::FLOAT32);
        auto span = view.as<float>();
        EXPECT_EQ(span.size(), 6u);
        EXPECT_EQ(view.dtype(), DataType::FLOAT32);
        EXPECT_EQ(view.shape(), shape);//should work because TensorShape has operator<=> defined

    }


    TEST(TensorViewTest, RejectsDtypeMismatch) {
        MemoryArena arena(1024);
        TensorShape shape{{2, 3}};
        TensorView view = TensorView::over_arena(arena, shape, DataType::FLOAT32);

        EXPECT_THROW(view.as<std::int32_t>(), std::logic_error);
    }

    TEST(TensorViewTest, WriteThenReadRoundTrip){
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

}
}