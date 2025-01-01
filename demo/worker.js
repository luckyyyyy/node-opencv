const { parentPort, threadId } = require('worker_threads');
const cv = require('../index.js');
const fs = require('fs/promises');

async function workerTask() {
  const full = await fs.readFile('./demo/full.jpg');
  const image = await fs.readFile('./demo/1.jpg');

  try {
    const [image1, image2, image3, image4] = await Promise.all([
      cv.imdecodeAsync(full),
      cv.imdecodeAsync(image),
      cv.imreadAsync('./demo/full.jpg'),
      cv.imreadAsync('./demo/1.jpg'),
    ]);

    const matched = await image1.matchTemplateAsync(image2, cv.TM_CCOEFF_NORMED);
    const minMax = await matched.minMaxLocAsync();
    // console.log(minMax);
    console.log('worker: ' + threadId, image2.size, minMax.maxVal * 100);

    // parentPort.postMessage({
    //   size: image2.size,
    //   maxVal: minMax.maxVal * 100
    // });

    matched.release();
    image1.release();
    image2.release();
    image3.release();
    image4.release();
  } catch (error) {
    parentPort.postMessage({ error: error.message });
  }
}

workerTask().catch((error) => {
  parentPort.postMessage({ error: error.message });
});