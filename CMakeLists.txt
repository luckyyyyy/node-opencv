cmake_minimum_required(VERSION 3.15)

project(node-opencv
    VERSION 1.0.0
    DESCRIPTION "OpenCV Node.js addon"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(OPENCV_INCLUDE_DIR $ENV{OPENCV_INCLUDE_DIR})
set(OPENCV_LIB_DIR $ENV{OPENCV_LIB_DIR})
set(NODE_ADDON_API_DIR ${CMAKE_SOURCE_DIR}/node_modules/node-addon-api)

set(SOURCE_FILES
    src/main.cc
    src/image_io.cc
    src/mat.cc
    src/utility.cc
)

add_library(${PROJECT_NAME} SHARED ${SOURCE_FILES})

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_JS_INC}
    ${NODE_ADDON_API_DIR}
    ${OPENCV_INCLUDE_DIR}
)

target_compile_definitions(${PROJECT_NAME} PRIVATE
    NAPI_DISABLE_CPP_EXCEPTIONS
)

if(CMAKE_JS_VERSION)
    target_link_libraries(${PROJECT_NAME} PRIVATE ${CMAKE_JS_LIB})
endif()

if(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE
        /EHsc
        /GR
    )
    set(OPENCV_LIBS
        ${OPENCV_LIB_DIR}/opencv_world4100.lib
    )
else()
    target_compile_options(${PROJECT_NAME} PRIVATE
        -fexceptions
        -frtti
    )
endif()

if(APPLE)
    set_target_properties(${PROJECT_NAME} PROPERTIES
        XCODE_ATTRIBUTE_GCC_ENABLE_CPP_EXCEPTIONS YES
        XCODE_ATTRIBUTE_GCC_ENABLE_CPP_RTTI YES
        XCODE_ATTRIBUTE_CLANG_CXX_LIBRARY "libc++"
        XCODE_ATTRIBUTE_MACOSX_DEPLOYMENT_TARGET "10.7"
    )
    set(OPENCV_LIBS
        ${OPENCV_LIB_DIR}/libopencv_core.dylib
        ${OPENCV_LIB_DIR}/libopencv_imgcodecs.dylib
        ${OPENCV_LIB_DIR}/libopencv_imgproc.dylib
    )
elseif(UNIX AND NOT APPLE)
    set(OPENCV_LIBS
        ${OPENCV_LIB_DIR}/libopencv_core.so
        ${OPENCV_LIB_DIR}/libopencv_imgcodecs.so
        ${OPENCV_LIB_DIR}/libopencv_imgproc.so
    )
endif()

if(DEFINED OPENCV_LIBS)
    target_link_libraries(${PROJECT_NAME} PRIVATE ${OPENCV_LIBS})
endif()

set_target_properties(${PROJECT_NAME} PROPERTIES
    PREFIX ""
    SUFFIX ".node"
)

add_definitions(-DNODE_ADDON_API_CPP_EXCEPTIONS)
add_definitions(-DNODE_ADDON_API_CPP_EXCEPTIONS_ALL)