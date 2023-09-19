cmake_minimum_required(VERSION 3.26)
include(FetchContent)

FetchContent_Declare(
        gcem
        GIT_REPOSITORY https://github.com/kthohr/gcem.git
        GIT_TAG        8422f5307b0498d09cf626f38acb03fe9f1efd94  # release-v1.17.0
)

FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)

FetchContent_MakeAvailable(gcem googletest)

project(Aft)

set(HOMEBREW_PREFIX "/opt/homebrew"
    CACHE PATH "Path to Homebrew installation")

set(CMAKE_PREFIX_PATH
    "${HOMEBREW_PREFIX}")

list(TRANSFORM CMAKE_PREFIX_PATH APPEND "/include"
    OUTPUT_VARIABLE CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES)
set(CMAKE_C_STANDARD_INCLUDE_DIRECTORIES "${CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES}")

set(CMAKE_FIND_FRAMEWORK NEVER)
set(CMAKE_FIND_APPBUNDLE NEVER)

set(CMAKE_FIND_USE_CMAKE_SYSTEM_PATH FALSE)
set(CMAKE_FIND_USE_SYSTEM_ENVIRONMENT_PATH FALSE)

set(CMAKE_CXX_STANDARD 20)

enable_testing()

add_executable(Aft main.cpp)

target_include_directories(Aft PUBLIC include ${CMAKE_BINARY_DIR}/_deps/gcem-src/include)
target_link_libraries(Aft GTest::gtest)
target_compile_definitions(Aft PUBLIC $<$<CONFIG:DEBUG>:AFT_DEBUG>)

include(GoogleTest)
gtest_discover_tests(Aft)
