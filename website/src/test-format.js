// test-format.js
import { readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const keypointsData = JSON.parse(
  readFileSync(join(__dirname, 'assets', 'serve-2-avatar.json'), 'utf8')
);

console.log('Keypoints data type:', typeof keypointsData);
console.log('Is array?', Array.isArray(keypointsData));
console.log('Has frames property?', 'frames' in keypointsData);

if ('frames' in keypointsData) {
  console.log('Number of frames:', keypointsData.frames.length);
  if (keypointsData.frames.length > 0) {
    console.log('First frame structure:', JSON.stringify(keypointsData.frames[0], null, 2).substring(0, 500) + '...');
  }
} else if (Array.isArray(keypointsData)) {
  console.log('Number of frames:', keypointsData.length);
  if (keypointsData.length > 0) {
    console.log('First frame structure:', JSON.stringify(keypointsData[0], null, 2).substring(0, 500) + '...');
  }
} 