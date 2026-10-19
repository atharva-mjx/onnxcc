#include "onnxcc/ir/attribute.h"
#include "onnxcc/ir/node.h"
#include <type_traits>    
#include <utility>
#include <stdexcept>

// Explicit includes so this file does not rely on transitive includes. std::visit and std::variant_size_v need <variant> directly.
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>


// Internal helpers for error messages. The anonymous namespace gives them internal linkage so they stay private to this file and can't collide with other translation units.
namespace{

// Helper to identify the type of the template T.
template <typename T>
std::string_view type_name()
{

    // std::is_same_v is a compile time boolean variable which tells if two types are same or not.
    if constexpr (std::is_same_v<T, std::int64_t>)
        return "int64";
    else if constexpr (std::is_same_v<T, float>)
        return "float";
    else if constexpr (std::is_same_v<T, std::string>)
        return "string";
    else if constexpr (std::is_same_v<T, std::vector<std::int64_t>>)
        return "vector<int64>";
    else if constexpr (std::is_same_v<T, std::vector<float>>)
        return "vector<float>";
    else if constexpr (std::is_same_v<T, onnxcc::Tensor>)
        return "Tensor";
    else
        // No "unknown" fallback: a type that is not an Attribute alternative (or a new alternative without a name here) fails to compile instead of printing a vague name.
        static_assert(!sizeof(T), "type_name<T>: not an Attribute alternative");
}

// Helper to identify the type name of the attribute.
std::string_view held_name(const onnxcc::Attribute &attr)
{
    // std::visit calls the lambda with whichever alternative the variant currently holds, so this follows the variant automatically. A new alternative without a type_name<T> entry fails to compile, so there is no "unknown" case.
    return std::visit(
        [](const auto &v) { return type_name<std::decay_t<decltype(v)>>(); },
        attr);
}
} //namespace

template <typename T>
std::optional<T> onnxcc::get_attr(const onnxcc::Node &node, std::string_view name)
{
    // Unordered_map .find wants std::string so we convert name into it.
    const auto attr_iter = node.attributes.find(std::string(name));

    // Returns nullopt when attribute not found as per requirement.
    if (attr_iter == node.attributes.end())
        return std::nullopt;

    const Attribute &attr_variant = attr_iter->second;

    // std::get_if<T> is a non-throwing check: returns a pointer to the T if the variant holds one, otherwise nullptr.
    // Must copy, not move: 'typed_value' points into the Node's attribute map, and node is const&.
    if (const T *typed_value = std::get_if<T>(&attr_variant))
        return *typed_value;

    // Type mismatch runtime error: clearly reports node name, op_type, attribute name and both types.
    throw std::runtime_error(
        "Node '" + node.name +
        "' (op_type '" + node.op_type +
        "'): attribute '" + std::string(name) +
        "' has type " + std::string(held_name(attr_variant)) +
        ", but was requested as " + std::string(type_name<T>()));
}

template <typename T>
T onnxcc::get_attr_or(const onnxcc::Node &node, std::string_view name, T fallback)
{
    auto value = onnxcc::get_attr<T>(node, name);

    // Returns fallback if the attribute is absent.
    // std::move is needed because the ternary yields an lvalue, which would not be implicitly moved on return and would be silently copied.
    return value ? std::move(*value) : std::move(fallback);
}

template <typename T>
T onnxcc::get_attr_required(const onnxcc::Node &node, std::string_view name)
{
    auto value = onnxcc::get_attr<T>(node, name);
    // Throws error if attribute is absent.
    if (value)
    {
        return std::move(*value);
    }
    throw std::runtime_error("Node '" + node.name + "' (op_type '" + node.op_type + "'): required attribute '" + std::string(name) + "' is missing");
}

// Explicitly instantiate all six supported attribute types.
#define INSTANTIATE_ATTR(T) \
    template std::optional<T> onnxcc::get_attr<T>( \
        const onnxcc::Node&, std::string_view); \
    template T onnxcc::get_attr_or<T>( \
        const onnxcc::Node&, std::string_view, T); \
    template T onnxcc::get_attr_required<T>( \
        const onnxcc::Node&, std::string_view);

INSTANTIATE_ATTR(std::int64_t)
INSTANTIATE_ATTR(float)
INSTANTIATE_ATTR(std::string)
INSTANTIATE_ATTR(std::vector<std::int64_t>)
INSTANTIATE_ATTR(std::vector<float>)
INSTANTIATE_ATTR(onnxcc::Tensor)

#undef INSTANTIATE_ATTR