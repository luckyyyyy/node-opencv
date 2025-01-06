const cv = require('./node-opencv-rs.node');
const { promisify } = require('util');

const imdecodeAsync = promisify(cv.imdecodeCallback);
const imreadAsync = promisify(cv.imreadCallback);

cv.Mat.prototype.matchTemplateAsync = promisify(cv.Mat.prototype.matchTemplateCallback);
cv.Mat.prototype.minMaxLocAsync = promisify(cv.Mat.prototype.minMaxLocCallback);

const cvProxy = new Proxy(cv, {
  get(target, prop) {
    if (prop === 'imdecodeAsync') {
      return imdecodeAsync;
    } else if (prop === 'imreadAsync') {
      return imreadAsync;
    } else {
      return target[prop];
    }
  }
});

module.exports = cvProxy;
