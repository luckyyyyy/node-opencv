{
  "targets": [{
    "target_name": "node-opencv",
    "cflags!": [ "-fno-exceptions", "-fno-rtti" ],
    "cflags_cc!": [ "-fno-exceptions", "-fno-rtti" ],
    "conditions": [
      ['OS=="mac"', {
        "xcode_settings": {
          "GCC_ENABLE_CPP_EXCEPTIONS": "YES",
          "GCC_ENABLE_CPP_RTTI": "YES",
          "CLANG_CXX_LIBRARY": "libc++",
          "MACOSX_DEPLOYMENT_TARGET": "10.7"
        },
        "include_dirs": [
          "<!@(node -p \"require('node-addon-api').include\")",
          "<!@(echo $OPENCV_INCLUDE_DIR)"
        ],
        "libraries": [
          "<!@(echo $OPENCV_LIB_DIR)/libopencv_core.dylib",
          "<!@(echo $OPENCV_LIB_DIR)/libopencv_imgcodecs.dylib",
          "<!@(echo $OPENCV_LIB_DIR)/libopencv_imgproc.dylib"
        ]
      }],
      ['OS=="linux"', {
        "include_dirs": [
          "<!@(node -p \"require('node-addon-api').include\")",
          "<!@(echo $OPENCV_INCLUDE_DIR)"
        ],
        "libraries": [
          "<!@(echo $OPENCV_LIB_DIR)/libopencv_core.so",
          "<!@(echo $OPENCV_LIB_DIR)/libopencv_imgcodecs.so",
          "<!@(echo $OPENCV_LIB_DIR)/libopencv_imgproc.so"
        ]
      }],
      ['OS=="win"', {
        "msvs_settings": {
          "VCCLCompilerTool": {
            "ExceptionHandling": 1,
            "RuntimeTypeInfo": "true"
          }
        },
        "include_dirs": [
          "<!@(node -p \"require('node-addon-api').include\")",
          "<!@(echo %OPENCV_INCLUDE_DIR%)"
        ],
        "libraries": [
          "<!@(echo %OPENCV_LIB_DIR%)/opencv_core480.lib",
          "<!@(echo %OPENCV_LIB_DIR%)/opencv_imgcodecs480.lib",
          "<!@(echo %OPENCV_LIB_DIR%)/opencv_imgproc480.lib"
        ]
      }]
    ],
    "sources": [
      "src/main.cpp",
      "src/matrix.cpp"
    ],
    'defines': [ 'NAPI_DISABLE_CPP_EXCEPTIONS' ]
  }]
}
