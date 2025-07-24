const cv = require('../main.js');
const fs = require('fs');

async function testFlip() {
  console.log('Testing cv.flip() function...');
  
  try {
    // Load a test image
    const image = cv.imread('./demo/1.jpg');
    console.log(`Original image size: ${image.cols}x${image.rows}`);
    
    // Test horizontal flip
    const flippedHorizontal = image.flip(cv.FLIP_HORIZONTAL);
    console.log(`Horizontal flip successful - size: ${flippedHorizontal.cols}x${flippedHorizontal.rows}`);
    
    // Test vertical flip
    const flippedVertical = image.flip(cv.FLIP_VERTICAL);
    console.log(`Vertical flip successful - size: ${flippedVertical.cols}x${flippedVertical.rows}`);
    
    // Test both axes flip
    const flippedBoth = image.flip(cv.FLIP_BOTH);
    console.log(`Both axes flip successful - size: ${flippedBoth.cols}x${flippedBoth.rows}`);
    
    // Verify flip constants
    console.log(`Flip constants - Horizontal: ${cv.FLIP_HORIZONTAL}, Vertical: ${cv.FLIP_VERTICAL}, Both: ${cv.FLIP_BOTH}`);
    
    // Verify that dimensions remain the same
    if (image.cols === flippedHorizontal.cols && 
        image.rows === flippedHorizontal.rows &&
        image.cols === flippedVertical.cols && 
        image.rows === flippedVertical.rows &&
        image.cols === flippedBoth.cols && 
        image.rows === flippedBoth.rows) {
      console.log('✅ All tests passed! Dimensions preserved correctly.');
    } else {
      console.log('❌ Test failed! Dimensions not preserved.');
    }
    
    // Save test results (optional - can verify visually)
    try {
      const encodedHorizontal = cv.imencode('.jpg', flippedHorizontal);
      fs.writeFileSync('./demo/test_flip_horizontal.jpg', encodedHorizontal);
      console.log('Horizontal flip image saved as test_flip_horizontal.jpg');
      
      const encodedVertical = cv.imencode('.jpg', flippedVertical);
      fs.writeFileSync('./demo/test_flip_vertical.jpg', encodedVertical);
      console.log('Vertical flip image saved as test_flip_vertical.jpg');
      
      const encodedBoth = cv.imencode('.jpg', flippedBoth);
      fs.writeFileSync('./demo/test_flip_both.jpg', encodedBoth);
      console.log('Both axes flip image saved as test_flip_both.jpg');
    } catch (saveError) {
      console.log('Note: Could not save test images (this is optional)');
    }
    
    // Clean up
    image.release();
    flippedHorizontal.release();
    flippedVertical.release();
    flippedBoth.release();
    
  } catch (error) {
    console.error('❌ Test failed with error:', error.message);
    process.exit(1);
  }
}

testFlip().catch(console.error);