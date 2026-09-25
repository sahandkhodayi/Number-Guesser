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
