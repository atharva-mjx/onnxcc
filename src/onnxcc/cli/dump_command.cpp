#include "onnxcc/cli/dump_command.h"

#include <filesystem>
#include <format>
#include <ostream>
#include <system_error>

#include "onnxcc/cli/exit_codes.h"
#include "onnxcc/ir/status.h"
#include "onnxcc/version.h"

namespace onnxcc::cli {

    Status validate_model_path(const std::filesystem::path& model) {
        // non-throwing overloads, so an unreadable parent dir reports instead of aborting
        std::error_code ec;
        if (!std::filesystem::exists(model, ec) || ec) {
            return Status::error(std::format("model file '{}' does not exist", model.string()));
        }
        // exists() is also true for dirs, devices and fifos (named pipes, where a read waits
        // for whoever writes), so a fifo would hang the parser instead of failing
        if (!std::filesystem::is_regular_file(model, ec) || ec) {
            return Status::error(std::format("'{}' is not a regular file", model.string()));
        }
        return Status::ok_value();
    }

    int run_dump(const DumpOptions& options, std::ostream& out, std::ostream& err) {
        const std::filesystem::path model(options.model);

        const Status status = validate_model_path(model);
        if (!status.ok()) {
            err << "onnxcc dump: error: " << status.message() << "\n";
            return kExitFailure;
        }

        out << "model: " << options.model << "\n";

        if (options.verbose) {
            std::error_code ec;
            const std::uintmax_t size = std::filesystem::file_size(model, ec);
            out << "onnxcc: " << get_version() << " (" << get_version_codename() << ")\n";
            if (!ec) {
                out << "size: " << size << " bytes\n";
            }
        }

        if (options.show_graph) {
            out << "graph: not parsed yet, the model is only validated as a path for now\n";
        }

        return kExitOk;
    }
}
