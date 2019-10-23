cmake_minimum_required(VERSION 3.4.1)

set(lib_DIR ${CMAKE_SOURCE_DIR}/libs)
set(inc_DIR ${CMAKE_SOURCE_DIR}/includes)

include_directories(${inc_DIR})

add_library(native-lib SHARED src/main/cpp/native-lib.cpp )

add_library(DetectMNN SHARED IMPORTED )
set_target_properties(DetectMNN PROPERTIES IMPORTED_LOCATION ${lib_DIR}/${ANDROID_ABI}/libDetectMNN.so)

add_library(MNN SHARED IMPORTED )
set_target_properties(MNN PROPERTIES IMPORTED_LOCATION ${lib_DIR}/${ANDROID_ABI}/libMNN.so)


find_library(log-lib log)
target_link_libraries(native-lib DetectMNN MNN ${log-lib})