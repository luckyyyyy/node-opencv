const cv = require('../index.js');
const fs = require('fs/promises');
const { Worker } = require('worker_threads');
const path = require('path');

function createWorker() {
  return new Worker(path.join(__dirname, 'worker.js'));
}

async function main() {
  const full = await fs.readFile('./demo/full.jpg');
  const image = await fs.readFile('./demo/1.jpg');
  try {
    const a = cv.imdecode('hello world');
  } catch (error) {
    console.log(1,error.message)
  }
  setInterval(async () => {
        // // const a =  cv.imdecode(full)
    // a.release()
    const [image1, image2, image3, image4] = await Promise.all([
      cv.imdecodeAsync(full),
      cv.imdecodeAsync(image),
      cv.imreadAsync('./demo/full.jpg'),
      cv.imreadAsync('./demo/1.jpg'),
    ]);

    const matched = await image1.matchTemplateAsync(image2, cv.TM_CCOEFF_NORMED);
    const minMax = await matched.minMaxLocAsync();
    // const matched2 = await image3.matchTemplateAsync(image4, cv.TM_CCOEFF_NORMED);
    // const minMax2 = await matched2.minMaxLocAsync();
    console.log(image1.cols, image3.rows, image2.data, minMax)

    // console.log(minMax.maxVal * 100);
    console.log(image2.size)
    matched.release();
    // matched2.release();
    image1.release();
    image2.release();
    image3.release();
    image4.release();
    // console.log(1)
    // console.log(image4)
    // const b =  cv.imread('./full.jpg');
    // const c =  cv.imread('./1.jpg');
    // const a =  await b.matchTemplateAsync(c, cv.TM_CCOEFF_NORMED);
    // b.release();
    // c.release();
    // a.release();

  }, 1000);
}

main().catch(console.error);

for (let i = 0; i < 5; i++) {
  createWorker();
}

setInterval(() => {
  const used = process.memoryUsage();

  console.log(`${Math.round(used.rss / 1024 / 1024 * 100) / 100} MB`)
}, 100);
