#include "onnxcc/memory/tensor_view.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <new>
#include <span>
#include <stdexcept>
#include <utility>

#include "onnxcc/ir/tensor.h"
#include "onnxcc/memory/arena.h"

namespace onnxcc {
    namespace {

        // maps a C++ type to the DataType its bytes must be labelled with
        // there is no generic version, so any type not listed below is a link error
        template <typename T>
        DataType dtype_for();

        template <>
        DataType dtype_for<float>() {
            return DataType::FLOAT32;
        }

        template <>
        DataType dtype_for<double>() {
            return DataType::FLOAT64;
        }

        template <>
        DataType dtype_for<std::int32_t>() {
            return DataType::INT32;
        }

        template <>
        DataType dtype_for<std::int64_t>() {
            return DataType::INT64;
        }

        template <>
        DataType dtype_for<bool>() {
            return DataType::BOOL;
        }

    }  // anonymous namespace

    // every check lives here, so a view that exists is always valid and the getters can trust it
    TensorView::TensorView(std::byte* data, std::size_t size_bytes, TensorShape shape, DataType dtype) {
        if (shape.is_dynamic()) {
            throw std::logic_error("TensorView cannot be constructed with a dynamic shape");
        }
        // num_elements() is noexcept but dtype_size() throws on UNDEFINED, so stop it here
        if (dtype == DataType::UNDEFINED) {
            throw std::logic_error("TensorView cannot be constructed with an undefined dtype");
        }
        // debug only bounds check (task 8 wants it compiled out in release)
        // a wrong byte count is a caller bug, the view would claim memory it does not cover
        assert(size_bytes == shape.num_bytes(dtype) && "byte count does not match shape and dtype");

        m_data = data;
        m_size_bytes = size_bytes;
        m_shape = std::move(shape);
        m_dtype = dtype;
    }

    const TensorShape& TensorView::shape() const noexcept {
        return m_shape;
    }

    DataType TensorView::dtype() const noexcept {
        return m_dtype;
    }

    std::size_t TensorView::num_elements() const noexcept {
        // a default constructed view has no dtype, so there is nothing to divide by
        if (m_size_bytes == 0) {
            return 0;
        }
        return m_size_bytes / dtype_size(m_dtype);
    }

    std::byte* TensorView::data() noexcept {
        return m_data;
    }

    const std::byte* TensorView::data() const noexcept {
        return m_data;
    }

    // the arena owns the bytes, the view only labels them
    TensorView TensorView::over_arena(MemoryArena& arena, const TensorShape& shape, DataType dtype) {
        const std::size_t bytes = shape.num_bytes(dtype);
        void* ptr = arena.allocate(bytes);
        if (ptr == nullptr) {
            throw std::bad_alloc{};
        }
        return TensorView(static_cast<std::byte*>(ptr), bytes, shape, dtype);
    }

    // initializer bytes already live in the Tensor's vector, so nothing is allocated
    TensorView TensorView::over_tensor(Tensor& tensor) {
        auto* bytes = reinterpret_cast<std::byte*>(tensor.data.data());
        return TensorView(bytes, tensor.data.size(), tensor.shape, tensor.dtype);
    }

    // runtime type guard, reading float bytes as int32 would silently give garbage
    template <typename T>
    std::span<T> TensorView::as() {
        if (m_dtype != dtype_for<T>()) {
            throw std::logic_error("TensorView dtype does not match the requested type");
        }
        // debug only, the span must cover exactly the bytes the view owns, no more no less
        assert(num_elements() * sizeof(T) == m_size_bytes);
        return std::span<T>(reinterpret_cast<T*>(m_data), num_elements());
    }

    template <typename T>
    std::span<const T> TensorView::as() const {
        if (m_dtype != dtype_for<T>()) {
            throw std::logic_error("TensorView dtype does not match the requested type");
        }
        assert(num_elements() * sizeof(T) == m_size_bytes);
        return std::span<const T>(reinterpret_cast<const T*>(m_data), num_elements());
    }

    // as() is defined here and not in the header, so every supported type has to be stamped out here
    template std::span<float> TensorView::as<float>();
    template std::span<double> TensorView::as<double>();
    template std::span<std::int32_t> TensorView::as<std::int32_t>();
    template std::span<std::int64_t> TensorView::as<std::int64_t>();
    template std::span<bool> TensorView::as<bool>();

    template std::span<const float> TensorView::as<float>() const;
    template std::span<const double> TensorView::as<double>() const;
    template std::span<const std::int32_t> TensorView::as<std::int32_t>() const;
    template std::span<const std::int64_t> TensorView::as<std::int64_t>() const;
    template std::span<const bool> TensorView::as<bool>() const;
}
