const brain = require('brain.js');

// 1. Prepare Training Data (Time Series of daily max temperatures in Celsius)
// We provide a sequence of temperatures representing weekly trends (e.g., warming up, cooling down)
const trainingData = [
  [20, 21, 22, 23, 24, 25, 26], // Steady warming trend
  [15, 17, 20, 22, 21, 19, 16], // Warming then cooling (spring/fall pattern)
  [30, 32, 33, 35, 34, 32, 30], // Hot summer week
  [5,  6,  8,  7,  9,  10, 11]   // Cold winter week
];

// 2. Initialize LSTM Time Step Neural Network
// LSTM (Long Short-Term Memory) is designed specifically for sequential data over time.
const net = new brain.recurrent.LSTMTimeStep();

// 3. Train the Network
console.log('Training weather time-series model...');
net.train(trainingData, {
  iterations: 1000,
  errorThresh: 0.005,
  log: true,
  logPeriod: 200
});

console.log('Training complete!\n--- Weather Forecasts ---\n');

// 4. Forecast future temperatures based on a recent sequence
// Let's test a sequence that looks like a warming trend: [22, 23, 24, 25]
const recentTemperatures = [22, 23, 24, 25];

// Predict the next temperature
const nextTemp = net.run(recentTemperatures);

console.log(`Recent temperatures: [${recentTemperatures.join(', ')}] °C`);
console.log(`-> Predicted temperature for the next day: ${nextTemp.toFixed(1)} °C\n`);

// 5. Multi-step forecasting (predicting multiple days ahead)
// We can append our prediction to the array and forecast further into the future!
const forecastWindow = [...recentTemperatures];
const daysToForecast = 3;

console.log(`Forecasting next ${daysToForecast} days:`);
for (let i = 0; i < daysToForecast; i++) {
  const prediction = net.run(forecastWindow);
  console.log(`  Day +${i + 1}: ${prediction.toFixed(1)} °C`);
  
  // Slide the window forward by pushing the prediction and shifting/dropping the oldest
  forecastWindow.shift();
  forecastWindow.push(prediction);
}
