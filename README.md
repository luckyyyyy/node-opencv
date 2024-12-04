# Node.js OpenCV

This guide helps you properly configure and use the Node.js OpenCV native addon.

## Requirements

- Node.js (14.x or higher recommended)
- OpenCV (4.x recommended)
- Cmake
- C++ Compiler
  - Windows: Visual Studio 2019 or higher
  - macOS: Xcode Command Line Tools
  - Linux: GCC/G++

## Installing OpenCV

### Windows
1. Download and install from [OpenCV Official Website](https://opencv.org/releases/)
2. Default path: `C:/opencv/`

### macOS
```bash
brew install opencv
```
Default path: `/opt/homebrew/Cellar/opencv/`

### Linux
```bash
sudo apt-get install libopencv-dev
# or
sudo yum install opencv-devel
```
Default path: `/usr/local/`

## Environment Variables

### Windows
```powershell
$Env:OPENCV_INCLUDE_DIR = "D:\opencv\build\include"
$Env:OPENCV_LIB_DIR = "D:\opencv\build\x64\vc16\lib"
# and added D:\opencv\build\x64\vc16\bin to system path
```

### macOS
```bash
export OPENCV_INCLUDE_DIR=/opt/homebrew/Cellar/opencv/4.10.0_12/include/opencv4
export OPENCV_LIB_DIR=/opt/homebrew/Cellar/opencv/4.10.0_12/lib
```

### Linux
```bash
export OPENCV_INCLUDE_DIR=/usr/local/include/opencv4
export OPENCV_LIB_DIR=/usr/local/lib
```

## Building and Running

1. Build native module:
```bash
yarn install
```

2. Use in JavaScript:
```javascript
const addon = require('node-opencv');
const [image1, image2] = await Promise.all([
  cv.imdecodeAsync(full),
  cv.imdecodeAsync(image),
]);
const matched = await image1.matchTemplateAsync(image2, cv.TM_CCOEFF_NORMED);
const minMax = await matched.minMaxLocAsync();
console.log(minMax.maxVal * 100);
```

## Common Issues

### OpenCV Libraries Not Found
- Check environment variables
- Verify OpenCV installation path
- For Windows users, ensure correct library version (e.g., opencv_core480.lib)

### Build Errors
- Ensure all necessary build tools are installed
- Windows users: Verify Visual Studio and Windows SDK installation
- Check OpenCV version compatibility

### Runtime Errors
- Windows: Ensure OpenCV DLLs are in system path
- Linux/macOS: Verify dynamic library paths

## Version Compatibility

| OpenCV Version | Node.js Version | Support |
|---------------|----------------|----------|
| 4.x           | 14.x+          | ✅       |
| 3.x           | 12.x+          | ✅       |

## Additional Resources

- [OpenCV Documentation](https://docs.opencv.org/)
- [Node Addon API](https://github.com/nodejs/node-addon-api)

## License

This project is licensed under the MIT License - see the LICENSE file for details.