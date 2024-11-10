const cv = require('./index.js');
const fs = require('fs/promises');

async function main() {
  const full = await fs.readFile('./full.jpg');
  const image = await fs.readFile('./1.jpg');
  setInterval(async() => {
    const [image1, image2] = await Promise.all([
      cv.imdecodeAsync(full),
      cv.imdecodeAsync(image),
    ]);
    const matched = await image1.matchTemplateAsync(image2, cv.TM_CCOEFF_NORMED);
    const minMax = await matched.minMaxLocAsync();
    console.log(minMax.maxVal * 100);
  }, 100)
}

main().catch(console.error);

setInterval(() => {
  const used = process.memoryUsage();
  // console.log({
  //   rss: `${Math.round(used.rss / 1024 / 1024 * 100) / 100} MB`,
  //   heapTotal: `${Math.round(used.heapTotal / 1024 / 1024 * 100) / 100} MB`,
  //   heapUsed: `${Math.round(used.heapUsed / 1024 / 1024 * 100) / 100} MB`,
  //   external: `${Math.round(used.external / 1024 / 1024 * 100) / 100} MB`,
  // });
}, 100);
