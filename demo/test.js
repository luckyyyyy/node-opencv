const cv = require('../index.js');
const fs = require('fs/promises');
const { Worker } = require('worker_threads');
const path = require('path');

console.log(cv.getBuildInformation())

function createWorker() {
  const worker =  new Worker(path.join(__dirname, 'worker.js'));
  setTimeout(() => {
    console.log(`worker: ${worker.threadId} terminate`);
    // worker.terminate();
  }, 200);
}

async function main() {
  const full = await fs.readFile('./demo/20250123-112635.png');
  const image = await fs.readFile('./demo/20250123-112630.png');
  try {
    const a = await cv.imdecode(Buffer.from('hello world'));
  } catch (error) {
    console.log('test error',error.message)
  }

  const [image1, image2, image3, image4] = await Promise.all([
    cv.imdecode(full),
    cv.imdecode(image),
    cv.imread('./demo/full.jpg'),
    cv.imread('./demo/1.jpg'),
  ]);


  const matched = await image1.matchTemplate(image2, cv.TM_CCOEFF_NORMED);
  console.time('matchTemplateAll')
  const result1 = await image1.matchTemplateAll(image2, cv.TM_CCOEFF_NORMED, 0.8, 0.1);
  console.log(result1)
  const result = await image3.matchTemplateAll(image4, cv.TM_CCOEFF_NORMED, 0.8, 0.1);
  console.log(result)
  console.timeEnd('matchTemplateAll')
  // const matched2 = await image3.matchTemplate(image4, cv.TM_CCOEFF_NORMED);
  // const minMax2 = await matched2.minMaxLoc();
  // console.log(minMax)
  const minMax2 = await matched.minMaxLoc();
  console.log(minMax2)


  // console.log(minMax.maxVal * 100);
  // console.log(image2.size)
  // matched.release();
  // // matched2.release();
  // image1.release();
  // image2.release();
  // image3.release();
  // image4.release();
  // console.log(1)
  // console.log(image4)
  // const b =  cv.imread('./full.jpg');
  // const c =  cv.imread('./1.jpg');
  // const a =  await b.matchTemplate(c, cv.TM_CCOEFF_NORMED);
  // b.release();
  // c.release();
  // a.release();
}

// setInterval(() => {
main().catch(console.error);
// }, 50);

// setInterval(() => {
  // for (let i = 0; i < 5; i++) {
  //   createWorker();
  // }
// }, 3000);


// setInterval(() => {
//   const used = process.memoryUsage();
//   console.log(`${Math.round(used.rss / 1024 / 1024 * 100) / 100} MB`)
// }, 1000);
