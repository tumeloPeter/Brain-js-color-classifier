const brain = require('brain.js');
const fs = require('fs');
const path = require('path');

const MODEL_PATH = path.join(__dirname, 'model.json');

// Helper function: Convert Hex color string to normalized RGB values (0 to 1)
function hexToRgbNormalized(hex) {
  hex = hex.replace(/^#/, '');
  const r = parseInt(hex.substring(0, 2), 16) / 255;
  const g = parseInt(hex.substring(2, 4), 16) / 255;
  const b = parseInt(hex.substring(4, 6), 16) / 255;
  return { r, g, b };
}

// Function to make predictions
function predictTextContrast(net, hexColor) {
  const rgb = hexToRgbNormalized(hexColor);
  const result = net.run(rgb);

  console.log(`Background Color: ${hexColor}`);
  console.log(`- White Text Confidence: ${(result.whiteText || 0).toFixed(4)}`);
  console.log(`- Black Text Confidence: ${(result.blackText || 0).toFixed(4)}`);

  const choice = (result.whiteText || 0) > (result.blackText || 0) ? 'WHITE' : 'BLACK';
  console.log(`-> Recommendation: Use ${choice} text.\n`);
}

function run() {
  const net = new brain.NeuralNetwork();

  // Check if a saved model already exists
  if (fs.existsSync(MODEL_PATH)) {
    console.log('Found saved model! Loading from disk (skipping training)...');
    const modelData = JSON.parse(fs.readFileSync(MODEL_PATH, 'utf8'));
    net.fromJSON(modelData);
  } else {
    console.log('No saved model found. Preparing training data...');
    const trainingData = [
      // Dark Backgrounds -> Choose White Text
      { input: hexToRgbNormalized('#000000'), output: { whiteText: 1 } },
      { input: hexToRgbNormalized('#0000FF'), output: { whiteText: 1 } },
      { input: hexToRgbNormalized('#121212'), output: { whiteText: 1 } },
      { input: hexToRgbNormalized('#800000'), output: { whiteText: 1 } },

      // Light Backgrounds -> Choose Black Text
      { input: hexToRgbNormalized('#FFFFFF'), output: { blackText: 1 } },
      { input: hexToRgbNormalized('#FFFF00'), output: { blackText: 1 } },
      { input: hexToRgbNormalized('#E0E0E0'), output: { blackText: 1 } },
      { input: hexToRgbNormalized('#00FFFF'), output: { blackText: 1 } }
    ];

    console.log('Training network...');
    net.train(trainingData, {
      iterations: 2000,
      errorThresh: 0.005
    });
    console.log('Training complete!');

    // Save the trained model to disk
    console.log('Saving model to model.json...');
    fs.writeFileSync(MODEL_PATH, JSON.stringify(net.toJSON(), null, 2), 'utf8');
    console.log('Model saved successfully!\n');
  }

  // Test predictions
  console.log('--- Testing Predictions ---');
  predictTextContrast(net, '#110022'); // Very dark purple
  predictTextContrast(net, '#FFF8DC'); // Cornsilk / Light off-white
  predictTextContrast(net, '#008000'); // Dark green
}

run();
