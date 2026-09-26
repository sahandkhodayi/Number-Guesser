cmake_minimum_required(VERSION 3.20) # cmake version

project(NumberGuesser C) # project name and language

set(CMAKE_C_STANDARD 11) # c11


set(CMAKE_C_STANDARD_REQUIRED ON)

set(CMAKE_C_EXTENSIONS OFF)

if(MSVC)
    add_compile_options(/W4) # w4 compilter for MSVC

else()
    add_compile_options(-Wall -Wextra -Wpedantic) # gcc for this

endif()

find_package(raylib CONFIG REQUIRED) # framework or library

add_executable(number_guesser   # name of exe
    c/src/main.c
    c/src/nn.c               # exe file from these
    c/src/ui.c
)

target_include_directories(number_guesser PRIVATE c/include)
target_link_libraries(number_guesser PRIVATE raylib)

if(UNIX AND NOT APPLE)
    target_link_libraries(number_guesser PRIVATE m) # different math for other os 
endif()
