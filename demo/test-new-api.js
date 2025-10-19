const cv = require('../index.js');
const fs = require('fs/promises');

async function testNewFunctions() {
  console.log('Testing new OpenCV API bindings...\n');

  try {
    // Load a test image
    const imagePath = './demo/20250123-112635.png';
    const imageBuffer = await fs.readFile(imagePath);
    const image = await cv.imdecode(imageBuffer);
    
    console.log('✓ Image loaded:', image.size);

    // Test flip
    console.log('\nTesting flip...');
    const flippedH = await cv.flip(image, 1); // Flip horizontally
    console.log('✓ Flipped horizontally:', flippedH.size);
    
    const flippedV = await cv.flip(image, 0); // Flip vertically
    console.log('✓ Flipped vertically:', flippedV.size);
    
    const flippedBoth = await cv.flip(image, -1); // Flip both
    console.log('✓ Flipped both:', flippedBoth.size);

    // Test rotate
    console.log('\nTesting rotate...');
    const rotated90 = await cv.rotate(image, cv.ROTATE_90_CLOCKWISE);
    console.log('✓ Rotated 90° clockwise:', rotated90.size);
    
    const rotated180 = await cv.rotate(image, cv.ROTATE_180);
    console.log('✓ Rotated 180°:', rotated180.size);
    
    const rotated270 = await cv.rotate(image, cv.ROTATE_90_COUNTERCLOCKWISE);
    console.log('✓ Rotated 270° clockwise:', rotated270.size);

    // Test gaussian blur
    console.log('\nTesting Gaussian blur...');
    const blurred = await cv.gaussianBlur(image, 5, 5, 1.5);
    console.log('✓ Gaussian blur applied:', blurred.size);

    // Test Canny edge detection
    console.log('\nTesting Canny edge detection...');
    const edges = await cv.canny(image, 50, 150);
    console.log('✓ Canny edges detected:', edges.size);

    // Test threshold
    console.log('\nTesting threshold...');
    const thresholded = await image.threshold(127, 255, cv.THRESH_BINARY);
    console.log('✓ Threshold applied:', thresholded.size);

    // Test adaptive threshold (needs grayscale image)
    console.log('\nTesting adaptive threshold...');
    const grayImage = await cv.imread(imagePath, cv.IMREAD_GRAYSCALE);
    const adaptiveThresh = await cv.adaptiveThreshold(
      grayImage,
      255,
      cv.ADAPTIVE_THRESH_GAUSSIAN_C,
      cv.THRESH_BINARY,
      11,
      2
    );
    console.log('✓ Adaptive threshold applied:', adaptiveThresh.size);

    // Test in_range
    console.log('\nTesting in_range...');
    const mask = await cv.inRange(image, [0, 0, 0], [128, 128, 128]);
    console.log('✓ In range mask created:', mask.size);

    // Test find_contours
    console.log('\nTesting find_contours...');
    const contours = await cv.findContours(
      edges,
      cv.RETR_EXTERNAL,
      cv.CHAIN_APPROX_SIMPLE
    );
    console.log('✓ Contours found:', contours.length);

    // Test draw_contours
    if (contours.length > 0) {
      console.log('\nTesting draw_contours...');
      const imageWithContours = await cv.drawContours(
        image,
        contours,
        -1, // Draw all contours
        { val0: 0, val1: 255, val2: 0, val3: 255 }, // Green color
        2
      );
      console.log('✓ Contours drawn:', imageWithContours.size);
    }

    // Test split and merge
    console.log('\nTesting split...');
    const channels = await cv.split(image);
    console.log('✓ Image split into', channels.length, 'channels');

    console.log('\nTesting merge...');
    const merged = await cv.merge(channels);
    console.log('✓ Channels merged:', merged.size);

    console.log('\n✅ All tests passed!');
  } catch (error) {
    console.error('❌ Test failed:', error.message);
    console.error(error.stack);
    process.exit(1);
  }
}

testNewFunctions().catch(console.error);
