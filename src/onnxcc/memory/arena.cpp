#include "onnxcc/memory/arena.h"

#include <cstdlib>
#include <limits>
#include <new>

namespace onnxcc {
    namespace {

        constexpr std::size_t kBufferAlign = 64;

        constexpr bool is_power_of_two(std::size_t value) noexcept {
            return value != 0 && (value & (value - 1)) == 0;
        }

        // next multiple of alignment at or above value, or 0 if that would wrap
        constexpr std::size_t align_up(std::size_t value, std::size_t alignment) noexcept {
            if (value > std::numeric_limits<std::size_t>::max() - (alignment - 1)) {
                return 0;
            }
            return (value + alignment - 1) & ~(alignment - 1);
        }

    }  // anonymous namespace

    MemoryArena::MemoryArena(std::size_t total_bytes) {
        // aligned_alloc is undefined unless size is a multiple of alignment
        const std::size_t total = align_up(total_bytes, kBufferAlign);
        if (total == 0 && total_bytes != 0) {
            throw std::bad_alloc{};
        }
        if (total == 0) {
            return;
        }

        void* buffer = std::aligned_alloc(kBufferAlign, total);
        if (buffer == nullptr) {
            throw std::bad_alloc{};
        }

        m_buffer = static_cast<std::byte*>(buffer);
        m_total = total;
    }

    MemoryArena::~MemoryArena() {
        std::free(m_buffer);
    }

    MemoryArena::MemoryArena(MemoryArena&& other) noexcept
        : m_buffer(other.m_buffer), m_offset(other.m_offset), m_total(other.m_total),
          m_peak(other.m_peak) {
        other.m_buffer = nullptr;
        other.m_offset = 0;
        other.m_total = 0;
        other.m_peak = 0;
    }

    MemoryArena& MemoryArena::operator=(MemoryArena&& other) noexcept {
        if (this == &other) {
            return *this;
        }

        std::free(m_buffer);

        m_buffer = other.m_buffer;
        m_offset = other.m_offset;
        m_total = other.m_total;
        m_peak = other.m_peak;

        other.m_buffer = nullptr;
        other.m_offset = 0;
        other.m_total = 0;
        other.m_peak = 0;
        return *this;
    }

    void* MemoryArena::allocate(std::size_t bytes, std::size_t alignment) {
        if (!is_power_of_two(alignment)) {
            return nullptr;
        }

        const std::size_t aligned = align_up(m_offset, alignment);
        if (aligned == 0 && m_offset != 0) {
            return nullptr;
        }
        if (aligned > m_total) {
            return nullptr;
        }

        // subtraction cannot wrap, unlike aligned + bytes
        if (bytes > m_total - aligned) {
            return nullptr;
        }

        std::byte* result = m_buffer + aligned;
        if (bytes == 0) {
            return result;
        }

        m_offset = aligned + bytes;
        if (m_offset > m_peak) {
            m_peak = m_offset;
        }
        return result;
    }

    void MemoryArena::reset() noexcept {
        m_offset = 0;
    }

    std::size_t MemoryArena::bytes_used() const noexcept {
        return m_offset;
    }

    std::size_t MemoryArena::bytes_peak() const noexcept {
        return m_peak;
    }

    std::size_t MemoryArena::bytes_total() const noexcept {
        return m_total;
    }
}
