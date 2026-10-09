#include <gtest/gtest.h>

#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>

#include "onnxcc/cli/dump_command.h"
#include "onnxcc/ir/status.h"

namespace {

std::size_t g_heap_allocations = 0;
bool g_track_allocations = false;

}

void* operator new(std::size_t size) {
    if (g_track_allocations) {
        ++g_heap_allocations;
    }
    return std::malloc(size);
}

void operator delete(void* p) noexcept {
    std::free(p);
}

void operator delete(void* p, std::size_t) noexcept {
    std::free(p);
}

namespace {

using onnxcc::Status;

TEST(StatusTest, OkStatusBasics) {
    const Status s = Status::ok_value();
    EXPECT_TRUE(s.ok());
    EXPECT_TRUE(static_cast<bool>(s));
    EXPECT_TRUE(s.message().empty());
}

TEST(StatusTest, ErrorStatusCapturesMessageAndSourceLocation) {
    const unsigned int line = __LINE__ + 1;
    const Status s = Status::error("validation failed");
    EXPECT_FALSE(s.ok());
    EXPECT_FALSE(static_cast<bool>(s));
    EXPECT_NE(std::string_view::npos, s.message().find("validation failed"));
    EXPECT_NE(std::string_view::npos, s.message().find("test_status.cpp"));
    EXPECT_NE(std::string_view::npos, s.message().find(std::to_string(line)));
}

TEST(StatusTest, SuccessPathIsAllocationFree) {
    g_heap_allocations = 0;
    g_track_allocations = true;

    Status s1 = Status::ok_value();
    Status s2 = s1;
    Status s3 = std::move(s1);
    const bool is_ok = s2.ok();
    const std::string_view msg = s2.message();

    g_track_allocations = false;

    EXPECT_EQ(0u, g_heap_allocations);
    EXPECT_TRUE(is_ok);
    EXPECT_TRUE(msg.empty());
}

TEST(StatusTest, WithContextOnFailure) {
    const Status err = Status::error("root error");
    const Status ctx = err.with_context("subsystem failure");

    EXPECT_FALSE(ctx.ok());
    EXPECT_NE(std::string_view::npos, ctx.message().find("subsystem failure: "));
    EXPECT_NE(std::string_view::npos, ctx.message().find("root error"));
}

TEST(StatusTest, WithContextOnSuccess) {
    const Status ok = Status::ok_value();
    const Status ctx = ok.with_context("subsystem");

    EXPECT_TRUE(ctx.ok());
    EXPECT_TRUE(ctx.message().empty());
}

TEST(StatusTest, ErrorWithContextHelper) {
    const Status s = Status::error_with_context("op_compiler", "unknown operator");

    EXPECT_FALSE(s.ok());
    EXPECT_NE(std::string_view::npos, s.message().find("op_compiler: unknown operator"));
    EXPECT_NE(std::string_view::npos, s.message().find("test_status.cpp"));
}

TEST(StatusTest, StatusEquality) {
    EXPECT_EQ(Status::ok_value(), Status::ok_value());
    EXPECT_NE(Status::ok_value(), Status::error("some error"));
}

TEST(StatusTest, WorkedExampleValidateModelPath) {
    const Status missing = onnxcc::cli::validate_model_path("non_existent_file_xyz.onnx");
    EXPECT_FALSE(missing.ok());
    EXPECT_NE(std::string_view::npos, missing.message().find("non_existent_file_xyz.onnx"));

    const std::filesystem::path dir = std::filesystem::temp_directory_path();
    const Status is_dir = onnxcc::cli::validate_model_path(dir);
    EXPECT_FALSE(is_dir.ok());
    EXPECT_NE(std::string_view::npos, is_dir.message().find("not a regular file"));

    const std::filesystem::path temp_file = dir / "test_status_model.onnx";
    std::ofstream(temp_file) << "fake onnx";
    const Status existing = onnxcc::cli::validate_model_path(temp_file);
    EXPECT_TRUE(existing.ok());
    std::filesystem::remove(temp_file);
}

}
