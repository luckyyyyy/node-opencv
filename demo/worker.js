const { Worker, isMainThread, parentPort, threadId } = require('worker_threads');
const cv = require('../index.js');
const fs = require('fs/promises');

if (isMainThread) {
  // 主线程：创建5个worker
  const workers = [];

  for (let i = 0; i < 5; i++) {
    const worker = new Worker(__filename);
    workers.push(worker);

    worker.on('message', (data) => {
      console.log(`Worker ${i}: ${data.message || data.error || JSON.stringify(data)}`);
    });

    worker.on('error', (error) => {
      console.error(`Worker ${i} error:`, error);
    });
  }

  // 2秒后终止所有worker
  setTimeout(() => {
    workers.forEach((worker, index) => {
      worker.terminate();
      console.log(`Worker ${index} terminated`);
    });
  }, 2000);

  setTimeout(() => {
    console.log('Main thread exiting');
  }, 5000);

} else {
  // worker线程：持续做模板匹配
  async function workerTask() {
    const full = await fs.readFile('./demo/full.jpg');
    const image = await fs.readFile('./demo/1.jpg');

    while (true) {
      try {
        const [image1, image2] = await Promise.all([
          cv.imdecode(full),
          cv.imdecode(image),
        ]);

        const matched = await image1.matchTemplate(image2, cv.TM_CCOEFF_NORMED);
        const minMax = await matched.minMaxLoc();

        parentPort.postMessage({
          message: `Thread ${threadId}: Match ${(minMax.maxVal * 100).toFixed(2)}%`
        });

        matched.release();
        image1.release();
        image2.release();

        // await new Promise(resolve => setTimeout(resolve, 100)); // 100ms间隔
      } catch (error) {
        parentPort.postMessage({ error: error.message });
        break;
      }
    }
  }

  workerTask().catch((error) => {
    parentPort.postMessage({ error: error.message });
  });
}
