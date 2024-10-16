cmake_minimum_required(VERSION 3.10)
project(dat_to_text)

set(CMAKE_C_STANDARD 11)

# Include header files
include_directories(include)

# Add executable and source files
add_executable(dat_to_text
        src/dat_to_text.c
        src/feature.c
        src/io.c
        src/signanalysis.c
)

