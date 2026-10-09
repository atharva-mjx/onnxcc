#pragma once

#include <filesystem>
#include <iosfwd>

#include "onnxcc/cli/parse.h"
#include "onnxcc/ir/status.h"

namespace onnxcc::cli {
    Status validate_model_path(const std::filesystem::path& model);
    int run_dump(const DumpOptions& options, std::ostream& out, std::ostream& err);
}
