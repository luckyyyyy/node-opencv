const cv = require('./index.js');
const fs = require('fs/promises');

async function main() {
  const full = await fs.readFile('./full.jpg');
  const image = await fs.readFile('./1.jpg');
  setInterval(async() => {
    const [image1, image2, image3, image4] = await Promise.all([
      cv.imdecodeAsync(full),
      cv.imdecodeAsync(image),
      cv.imreadAsync('./full.jpg'),
      cv.imreadAsync('./1.jpg'),
    ]);
    const matched = await image1.matchTemplateAsync(image2, cv.TM_CCOEFF_NORMED);
    const minMax = await matched.minMaxLocAsync();
    const matched2 = await image3.matchTemplateAsync(image4, cv.TM_CCOEFF_NORMED);
    const minMax2 = await matched2.minMaxLocAsync();

    console.log(image1.cols, image3.rows, image2.data)

    console.log(minMax.maxVal * 100, minMax2.maxVal * 100);
  }, 100)
}

main().catch(console.error);

setInterval(() => {
  const used = process.memoryUsage();
  console.log({
    rss: `${Math.round(used.rss / 1024 / 1024 * 100) / 100} MB`,
    heapTotal: `${Math.round(used.heapTotal / 1024 / 1024 * 100) / 100} MB`,
    heapUsed: `${Math.round(used.heapUsed / 1024 / 1024 * 100) / 100} MB`,
    external: `${Math.round(used.external / 1024 / 1024 * 100) / 100} MB`,
  });
}, 100);
