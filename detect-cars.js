const tf = require('@tensorflow/tfjs-node');
const cocoSsd = require('@tensorflow-models/coco-ssd');
const fs = require('fs').promises;  // Use fs.promises for async file operations
const axios = require('axios');

const imageName = 'car_image1.jpg';
const imgurl = 'https://storage.googleapis.com/smartdrive-storage/1696233015786.jpg';

async function downloadImage(url, filename) {
  const response = await axios({
    url,
    responseType: 'arraybuffer',
  });

  await fs.writeFile(filename, Buffer.from(response.data));  // Use await for async file write
}

async function run() {
  await downloadImage(imgurl, imageName);

  const model = await cocoSsd.load();
  const imgBuffer = await fs.readFile(imageName);  // Use fs.promises.readFile

  const inputImage = tf.node.decodeImage(new Uint8Array(imgBuffer));

  const predictions = await model.detect(inputImage);
  const actualCarPrediction = findLargestCar(predictions);

  const isCropped = checkIsCropped(actualCarPrediction, inputImage.shape);
  console.log(isCropped);

  tf.dispose([inputImage]);
}

function findLargestCar(predictions) {
  let largestArea = 0;
  let largestCarPrediction = null;

  for (const prediction of predictions) {
    if (prediction.class === 'car') {
      const [yMin, xMin, yMax, xMax] = prediction.bbox;
      const boundingBoxArea = (yMax - yMin) * (xMax - xMin);

      if (boundingBoxArea > largestArea) {
        largestArea = boundingBoxArea;
        largestCarPrediction = prediction;
      }
    }
  }

  return largestCarPrediction;
}

function checkIsCropped(prediction, imageShape) {
  // Adjust as needed based on your requirements
  const [yMin, xMin, yMax, xMax] = prediction.bbox;

  return {
    isCropped: xMin <= 0 || xMax >= imageShape[1] || yMin <= 0 || yMax >= imageShape[0],
    boundingBox: [yMin, xMin, yMax, xMax],
  };
}

run();
