const brain = require('brain.js');

// Helper function: Convert Hex color string (e.g. "#FF0000") to normalized RGB values (0 to 1)
// Neural networks perform best when inputs are normalized between 0 and 1.
function hexToRgbNormalized(hex) {
  // Remove leading '#' if present
  hex = hex.replace(/^#/, '');

  // Parse Red, Green, Blue
  const r = parseInt(hex.substring(0, 2), 16) / 255;
  const g = parseInt(hex.substring(2, 4), 16) / 255;
  const b = parseInt(hex.substring(4, 6), 16) / 255;

  return { r, g, b };
}

// 1. Initialize Neural Network
// We use NeuralNetwork GPU/CPU default implementation
const net = new brain.NeuralNetwork();

// 2. Prepare Training Data
// We train the network to decide text contrast (black vs white text) for a given background color
const trainingData = [
  // Dark Backgrounds -> Choose White Text
  { input: hexToRgbNormalized('#000000'), output: { whiteText: 1 } }, // Black background
  { input: hexToRgbNormalized('#0000FF'), output: { whiteText: 1 } }, // Pure Blue background
  { input: hexToRgbNormalized('#121212'), output: { whiteText: 1 } }, // Dark Gray background
  { input: hexToRgbNormalized('#800000'), output: { whiteText: 1 } }, // Maroon background
  {input: hexToRgbNormalized('#2f4048'), output:{whiteText: 1}},

  // Light Backgrounds -> Choose Black Text
  { input: hexToRgbNormalized('#FFFFFF'), output: { blackText: 1 } }, // White background
  { input: hexToRgbNormalized('#FFFF00'), output: { blackText: 1 } }, // Bright Yellow background
  { input: hexToRgbNormalized('#E0E0E0'), output: { blackText: 1 } }, // Light Gray background
  { input: hexToRgbNormalized('#00FFFF'), output: { blackText: 1 } }  // Cyan background
];

// 3. Train Network
console.log('Training network...');
net.train(trainingData, {
  iterations: 2000, // Number of training cycles
  errorThresh: 0.005, // Stop training if error drops below this threshold
  log: true,
  logPeriod: 20
});

const json = net.toJSON();
console.log(JSON.stringify(json, null, 2));

console.log('Training complete!\n');

// 4. Test with unseen Hex Colors
function predictTextContrast(hexColor) {
  const rgb = hexToRgbNormalized(hexColor);
  const result = net.run(rgb);

  console.log(`Background Color: ${hexColor}`);
  console.log(`- White Text Confidence: ${(result.whiteText || 0).toFixed(4)}`);
  console.log(`- Black Text Confidence: ${(result.blackText || 0).toFixed(4)}`);

  const choice = (result.whiteText || 0) > (result.blackText || 0) ? 'WHITE' : 'BLACK';
  console.log(`-> Recommendation: Use ${choice} text.\n`);
}

// Test predictions
predictTextContrast('#110022'); // Very dark purple
predictTextContrast('#FFF8DC'); // Cornsilk / Light off-white
predictTextContrast('#008000'); // Dark green
