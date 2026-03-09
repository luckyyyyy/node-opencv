'use strict';

// On Windows, prepend the bundled lib\ directory to PATH so that
// opencv_world*.dll shipped inside this package is found by LoadLibrary
// before the .node binding is loaded.
if (process.platform === 'win32') {
  const path = require('node:path');
  const fs = require('node:fs');
  const libDir = path.join(__dirname, 'lib');
  if (fs.existsSync(libDir)) {
    process.env.PATH = libDir + ';' + (process.env.PATH || '');
  }
}

module.exports = require('./_binding.js');
