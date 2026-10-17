cmake_minimum_required(VERSION 3.20)

project(NumberGuesser C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

include(CTest)

# Headless targets are always available, so unit tests and parity checks do not
# need Raylib or a desktop display.
add_executable(test_nn
    tests/test_nn.c
    c/src/nn.c
)
target_include_directories(test_nn PRIVATE c/include)
if(UNIX AND NOT APPLE)
    target_link_libraries(test_nn PRIVATE m)
endif()
add_test(NAME c_unit_tests COMMAND test_nn)

add_executable(c_benchmark
    benchmark/c_benchmark.c
    c/src/nn.c
)
target_include_directories(c_benchmark PRIVATE c/include)
if(UNIX AND NOT APPLE)
    target_link_libraries(c_benchmark PRIVATE m)
endif()

option(BUILD_GUI "Build the Raylib desktop application" ON)
if(BUILD_GUI)
    find_package(raylib CONFIG REQUIRED)

    add_executable(number_guesser
        c/src/main.c
        c/src/nn.c
        c/src/ui.c
    )
    target_include_directories(number_guesser PRIVATE c/include)
    target_link_libraries(number_guesser PRIVATE raylib)
    if(UNIX AND NOT APPLE)
        target_link_libraries(number_guesser PRIVATE m)
    endif()
endif()
