#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "onnxcc/memory/arena.h"

namespace {

using onnxcc::MemoryArena;

std::size_t address_of(void* p) {
    return reinterpret_cast<std::uintptr_t>(p);
}

TEST(MemoryArena, TotalIsRoundedUpToSixtyFour) {
    EXPECT_EQ(128u, MemoryArena(100).bytes_total());
    EXPECT_EQ(64u, MemoryArena(1).bytes_total());
    EXPECT_EQ(64u, MemoryArena(64).bytes_total());
    EXPECT_EQ(0u, MemoryArena(0).bytes_total());
}

TEST(MemoryArena, StartsEmpty) {
    MemoryArena arena(1024);

    EXPECT_EQ(0u, arena.bytes_used());
    EXPECT_EQ(0u, arena.bytes_peak());
}

TEST(MemoryArena, EveryAllocationIsSixtyFourByteAligned) {
    MemoryArena arena(64 * 1024);

    for (std::size_t bytes : {1u, 7u, 63u, 64u, 65u, 127u, 4096u}) {
        void* p = arena.allocate(bytes);
        ASSERT_NE(nullptr, p) << "ran out of room at " << bytes << " bytes";
        EXPECT_EQ(0u, address_of(p) % 64) << "misaligned for " << bytes << " bytes";
    }
}

TEST(MemoryArena, AllocationsDoNotOverlap) {
    MemoryArena arena(1024);

    auto* first = static_cast<std::byte*>(arena.allocate(100));
    auto* second = static_cast<std::byte*>(arena.allocate(100));

    ASSERT_NE(nullptr, first);
    ASSERT_NE(nullptr, second);
    EXPECT_GE(second - first, 100);
}

TEST(MemoryArena, WrittenBytesSurviveLaterAllocations) {
    MemoryArena arena(1024);

    auto* first = static_cast<std::byte*>(arena.allocate(8));
    ASSERT_NE(nullptr, first);
    for (int i = 0; i < 8; ++i) {
        first[i] = static_cast<std::byte>(i);
    }

    ASSERT_NE(nullptr, arena.allocate(256));

    for (int i = 0; i < 8; ++i) {
        EXPECT_EQ(static_cast<std::byte>(i), first[i]);
    }
}

TEST(MemoryArena, UsedCountsPaddingNotJustPayload) {
    MemoryArena arena(1024);

    ASSERT_NE(nullptr, arena.allocate(1));
    EXPECT_EQ(1u, arena.bytes_used());

    // 1 rounds up to 64 before the second allocation starts
    ASSERT_NE(nullptr, arena.allocate(1));
    EXPECT_EQ(65u, arena.bytes_used());
}

TEST(MemoryArena, ExhaustionReturnsNullAndLeavesArenaUsable) {
    MemoryArena arena(128);

    ASSERT_NE(nullptr, arena.allocate(100));
    const std::size_t used = arena.bytes_used();

    EXPECT_EQ(nullptr, arena.allocate(100));
    EXPECT_EQ(used, arena.bytes_used()) << "a failed allocation must not move the offset";

    // the rest of the buffer is still handed out
    EXPECT_NE(nullptr, arena.allocate(8, 1));
}

TEST(MemoryArena, HugeRequestOverflowsInsteadOfSucceeding) {
    MemoryArena arena(1024);

    EXPECT_EQ(nullptr, arena.allocate(std::numeric_limits<std::size_t>::max()));
    EXPECT_EQ(nullptr, arena.allocate(std::numeric_limits<std::size_t>::max() - 32));
    EXPECT_EQ(0u, arena.bytes_used());
}

TEST(MemoryArena, AlignmentMustBeAPowerOfTwo) {
    MemoryArena arena(1024);

    EXPECT_EQ(nullptr, arena.allocate(8, 0));
    EXPECT_EQ(nullptr, arena.allocate(8, 3));
    EXPECT_EQ(nullptr, arena.allocate(8, 48));
    EXPECT_NE(nullptr, arena.allocate(8, 32));
}

TEST(MemoryArena, SmallerAlignmentWastesLessPadding) {
    MemoryArena arena(1024);

    ASSERT_NE(nullptr, arena.allocate(1, 1));
    void* p = arena.allocate(1, 1);

    ASSERT_NE(nullptr, p);
    EXPECT_EQ(2u, arena.bytes_used()) << "alignment 1 should not pad at all";
}

TEST(MemoryArena, ZeroBytesReturnsAPointerWithoutAdvancing) {
    MemoryArena arena(1024);

    ASSERT_NE(nullptr, arena.allocate(32));
    const std::size_t used = arena.bytes_used();

    EXPECT_NE(nullptr, arena.allocate(0));
    EXPECT_EQ(used, arena.bytes_used());
}

TEST(MemoryArena, ResetRewindsUsedButKeepsPeak) {
    MemoryArena arena(1024);

    ASSERT_NE(nullptr, arena.allocate(300));
    const std::size_t peak = arena.bytes_peak();
    ASSERT_EQ(300u, peak);

    arena.reset();

    EXPECT_EQ(0u, arena.bytes_used());
    EXPECT_EQ(peak, arena.bytes_peak());
    EXPECT_EQ(1024u, arena.bytes_total()) << "reset must not free the buffer";
}

TEST(MemoryArena, ResetReusesTheSameMemory) {
    MemoryArena arena(1024);

    void* first = arena.allocate(64);
    arena.reset();
    void* second = arena.allocate(64);

    EXPECT_EQ(first, second);
}

TEST(MemoryArena, PeakTracksTheLargestRunNotTheLast) {
    MemoryArena arena(1024);

    ASSERT_NE(nullptr, arena.allocate(500));
    arena.reset();
    ASSERT_NE(nullptr, arena.allocate(100));

    EXPECT_EQ(100u, arena.bytes_used());
    EXPECT_EQ(500u, arena.bytes_peak());
}

TEST(MemoryArena, MoveConstructionEmptiesTheSource) {
    MemoryArena source(1024);
    void* before = source.allocate(128);
    ASSERT_NE(nullptr, before);

    MemoryArena moved(std::move(source));

    EXPECT_EQ(1024u, moved.bytes_total());
    EXPECT_EQ(128u, moved.bytes_used());
    EXPECT_EQ(0u, source.bytes_total());  // NOLINT(bugprone-use-after-move)
    EXPECT_EQ(0u, source.bytes_used());
    EXPECT_EQ(nullptr, source.allocate(8)) << "a moved-from arena owns no buffer";
    EXPECT_NE(nullptr, moved.allocate(8));
}

TEST(MemoryArena, MoveAssignmentReplacesTheTarget) {
    MemoryArena target(64);
    MemoryArena source(1024);
    ASSERT_NE(nullptr, source.allocate(256));

    target = std::move(source);

    EXPECT_EQ(1024u, target.bytes_total());
    EXPECT_EQ(256u, target.bytes_used());
    EXPECT_EQ(0u, source.bytes_total());  // NOLINT(bugprone-use-after-move)
    EXPECT_NE(nullptr, target.allocate(8));
}

TEST(MemoryArena, SelfMoveAssignmentKeepsTheBuffer) {
    MemoryArena arena(1024);
    ASSERT_NE(nullptr, arena.allocate(128));

    MemoryArena& alias = arena;
    arena = std::move(alias);

    EXPECT_EQ(1024u, arena.bytes_total());
    EXPECT_EQ(128u, arena.bytes_used());
    EXPECT_NE(nullptr, arena.allocate(8));
}

TEST(MemoryArena, ZeroSizedArenaHandsOutNothing) {
    MemoryArena arena(0);

    EXPECT_EQ(0u, arena.bytes_total());
    EXPECT_EQ(nullptr, arena.allocate(1));
}

TEST(MemoryArena, FillingTheWholeBufferIsAllowed) {
    MemoryArena arena(256);
    std::vector<void*> blocks;

    for (int i = 0; i < 4; ++i) {
        void* p = arena.allocate(64);
        ASSERT_NE(nullptr, p) << "block " << i;
        blocks.push_back(p);
    }

    EXPECT_EQ(256u, arena.bytes_used());
    EXPECT_EQ(nullptr, arena.allocate(1));
}

}  // namespace
