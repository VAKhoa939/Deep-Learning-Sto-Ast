const ort = require('onnxruntime-node');
const sharp = require('sharp');

const classes = [
  'plane', 'car', 'bird', 'cat',
  'deer', 'dog', 'frog', 'horse',
  'ship', 'truck'
];

async function predict(imageTensor, threshold = 0.3) {
  const session = await ort.InferenceSession.create('image_classifier_model.onnx');
  const input = {
    input: new ort.Tensor('float32', imageTensor.data, [1, 3, 32, 32])
  };
  const output = await session.run(input);
  const scores = output.output.data;

  // Apply softmax to get probabilities
  const expScores = scores.map(Math.exp);
  const sumExpScores = expScores.reduce((a, b) => a + b, 0);
  const probs = expScores.map(e => e / sumExpScores);

  const maxProb = Math.max(...probs);
  const maxIndex = probs.indexOf(maxProb);

  if (maxProb < threshold) {
    return 'unknown';  // Confidence too low, treat as unknown
  } else {
    return classes[maxIndex];  // Return predicted class
  }
}

async function base64ToFloat32Tensor(content, mimeType) {
  const buffer = Buffer.from(content, 'base64');

  // Convert and resize image to 32x32, and extract raw RGB pixel data
  const { data, info } = await sharp(buffer)
    .resize(32, 32)
    .toFormat('raw')
    .toBuffer({ resolveWithObject: true });

  // data is a Uint8Array with [R, G, B, R, G, B, ...] interleaved
  const floatData = new Float32Array(3 * 32 * 32);
  for (let i = 0; i < 32 * 32; i++) {
    floatData[i] = data[i * 3] / 255.0;         // R
    floatData[32 * 32 + i] = data[i * 3 + 1] / 255.0; // G
    floatData[2 * 32 * 32 + i] = data[i * 3 + 2] / 255.0; // B
  }

  return new ort.Tensor('float32', floatData, [1, 3, 32, 32]);
}

module.exports = {
  predict,
  base64ToFloat32Tensor
};