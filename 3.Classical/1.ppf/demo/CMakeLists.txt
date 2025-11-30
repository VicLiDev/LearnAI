cmake_minimum_required(VERSION 3.5)
project(ppf_demo)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found")
endif()

message(STATUS "OpenCV include: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libs: ${OpenCV_LIBS}")

add_executable(ppf_demo_gpt src/ppf_demo_gpt.cpp)

target_include_directories(ppf_demo_gpt PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(ppf_demo_gpt ${OpenCV_LIBS})


add_executable(ppf_load_match src/ppf_load_match.cpp)

target_include_directories(ppf_load_match PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(ppf_load_match ${OpenCV_LIBS})


add_executable(ppf_normal_computation src/ppf_normal_computation.cpp)

target_include_directories(ppf_normal_computation PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(ppf_normal_computation ${OpenCV_LIBS})

