const express = require('express');
const axios = require('axios');
const fs = require('fs-extra');
const tf = require('@tensorflow/tfjs-node');
const cocoSsd = require('@tensorflow-models/coco-ssd');
const deeplab = require('@tensorflow-models/deeplab');
const port = 3000;

const app = express();

app.get('/', (req, res) => {
 res.send('Hello, this is your Express API!');
});

app.get('/api/hello', async (req, res) => {
 try {
    const imagePath = req.query.imgPath;
    const result = await run(imagePath);
    res.json({ message: result });
 } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Something went wrong!' });
 }
});

async function downloadImage(url, filename) {
 const response = await axios({
    url,
    responseType: 'arraybuffer',
 });

 await fs.writeFile(filename, Buffer.from(response.data));
}

async function run(imgurl) {
 const imageName = 'car_image1.jpg';
 await downloadImage(imgurl, imageName);

 const model = await cocoSsd.load();
 const imgBuffer = await fs.readFile(imageName);

 const inputImage = tf.node.decodeImage(new Uint8Array(imgBuffer));

 const predictions = await model.detect(inputImage);

 const actualCarPrediction = findLargestCar(predictions);

 if(actualCarPrediction == null) {
    return 'No Cars Detected';
 }

 const isCropped = checkIsCropped(actualCarPrediction, inputImage.shape);

 const segmentationModel = await deeplab.load();
 const segmentationResult = await segmentationModel.segment(inputImage);

 tf.dispose([inputImage]);

 return isCropped;
}

function findLargestCar(predictions) {
 let largestArea = 0;
 let largestCarPrediction = null;

 if(predictions.length == 0) {
    return largestCarPrediction;
 }

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
 const [yMin, xMin, yMax, xMax] = prediction.bbox;

 return {
    isCropped: xMin <= 0 || xMax >= imageShape[1] || yMin <= 0 || yMax >= imageShape[0],
 };
}

app.listen(port, () => {
 console.log(`Server is running on http://localhost:${port}`);
});