cmake_minimum_required(VERSION 3.22.1)
project(polyglot LANGUAGES C CXX)

# llama.cpp is not vendored: clone it next to this file (see mobile/android/README.md).
set(LLAMA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp)
if(NOT EXISTS ${LLAMA_DIR}/CMakeLists.txt)
    message(FATAL_ERROR "llama.cpp not found at ${LLAMA_DIR}; see mobile/android/README.md")
endif()

set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
add_subdirectory(${LLAMA_DIR} llama-build)

add_library(polyglot SHARED polyglot_jni.cpp)
target_link_libraries(polyglot PRIVATE llama android log)
