const addon = require('./build/Release/node-opencv.node');

module.exports = {
    imread: addon.imread,
    imdecode: addon.imdecode,
    imdecodeAsync: addon.imdecodeAsync,
    imreadAsync: addon.imreadAsync,
    TM_CCOEFF_NORMED: addon.TM_CCOEFF_NORMED,
    TM_CCORR_NORMED: addon.TM_CCORR_NORMED,
    TM_SQDIFF_NORMED: addon.TM_SQDIFF_NORMED,
    TM_CCOEFF: addon.TM_CCOEFF,
    TM_CCORR: addon.TM_CCORR,
    TM_SQDIFF: addon.TM_SQDIFF,
    IMREAD_COLOR: addon.IMREAD_COLOR,
    IMREAD_GRAYSCALE: addon.IMREAD_GRAYSCALE,
    IMREAD_UNCHANGED: addon.IMREAD_UNCHANGED,
    IMREAD_ANYDEPTH: addon.IMREAD_ANYDEPTH,
    IMREAD_ANYCOLOR: addon.IMREAD_ANYCOLOR,

};
