#include "onnxcc/memory/tensor_view.h"
#include <stdexcept>
#include <cstddef>
#include <span>

#include "onnxcc/ir/tensor.h"
#include <cassert>
#include <cstdint>
#include "onnxcc/memory/arena.h"

namespace onnxcc{

    //over_tensor looks at tensor.data.data() and uses tensor.data.size() for byte count, tensor.data is std::vector<std::uint8_t> so it is contiguous and owns the memory. 
    // The   span is non-owning and can be used to view the tensor data in a type-safe way.
        TensorView::TensorView(std::byte* data, std::size_t size_bytes, TensorShape t_shape, DataType t_dtype){
            m_data = data;
            m_size_bytes = size_bytes;
            if(t_shape.is_dynamic()){
                throw std::logic_error("TensorView cannot be constructed with a dynamic shape");
            }
            assert(size_bytes == t_shape.num_bytes(t_dtype));
            m_shape = t_shape;
            m_dtype = t_dtype;

        }
        //Getters
        //Getter for shape
        const TensorShape& TensorView::shape() const noexcept{
            return m_shape;
        }
        //Getter for dtype
        DataType TensorView::dtype() const noexcept{
            return m_dtype;
        }
        //Getter for number of elements with 0 for default constructed view
        std::size_t TensorView::num_elements() const noexcept{
            if(m_size_bytes == 0){
                // default constructed view will have 0 bytes and hence 0 elements
                // assumes a default view cannot have non-zero bytes, which is true for the current implementation
            return 0;}
            return ( m_size_bytes / dtype_size(m_dtype)); //dtype_size from datatype.h, used as said
        }
        //Getter for non const data pointer    
        std::byte* TensorView::data() noexcept{
            return m_data;
        }
        //Getter for const data pointer
        const std::byte* TensorView::data() const noexcept{
            return m_data;
        }
        //static method, returns a TensorView over a MemoryArena, allocating the required bytes for the given shape and dtype. 
        // Throws std::logic_error if nullptr is recieved.
        TensorView TensorView::over_arena(MemoryArena& arena, const TensorShape& t_shape, DataType t_dtype){
            std::size_t bytes = t_shape.num_bytes(t_dtype);
            void* ptr = arena.allocate(bytes);
            if(ptr == nullptr){
                throw std::bad_alloc();

            }
            auto* byte_ptr = static_cast<std::byte*>(ptr);//converted to check for nullptr.

            return TensorView(byte_ptr, bytes, t_shape, t_dtype);
        }

        TensorView TensorView::over_tensor(Tensor& tensor){
            std::size_t bytes= tensor.data.size();
            auto* byte_ptr = reinterpret_cast<std::byte*>(tensor.data.data());
            return TensorView(byte_ptr, bytes, tensor.shape, tensor.dtype);
        }


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
        DataType dtype_for<std::int32_t>(){
            return DataType::INT32;
        }
        template <>
        DataType dtype_for<std::int64_t>(){
            return DataType::INT64;
        }
        template <>
        DataType dtype_for<bool>(){
            return DataType::BOOL;
        }

        template <typename T>
        std::span<T> TensorView::as() {

        if (m_dtype != dtype_for<T>()) {
            throw std::logic_error("dtype mismatch");
        }

        assert(m_size_bytes % sizeof(T) == 0);


        return std::span<T>(reinterpret_cast<T*>(m_data), num_elements());
        }
        
        template <typename T>
        std::span<const T> TensorView::as() const {

        if (m_dtype != dtype_for<T>()) {
            throw std::logic_error("dtype mismatch");
        }


        assert(m_size_bytes % sizeof(T) == 0);


        return std::span<const T>(reinterpret_cast<const T*>(m_data), num_elements());
        }

        template std::span<float> TensorView::as<float>();
        template std::span<double> TensorView::as<double>();
        
        template std::span<std::int32_t> TensorView::as<std::int32_t>();
        template std::span<std::int64_t> TensorView::as<std::int64_t>();
        
        template std::span<bool> TensorView::as<bool>();

        template std::span<const float> TensorView::as<float>() const ;
        template std::span<const double> TensorView::as<double>() const;
        
        template std::span<const std::int32_t> TensorView::as<std::int32_t>() const;
        template std::span<const std::int64_t> TensorView::as<std::int64_t>() const ;
        
        template std::span<const bool> TensorView::as<bool>() const ;

    }


