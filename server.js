require('@tensorflow/tfjs-node');

const http = require('http');
const socketio = require('socket.io');
const sign_type = require('./sign_type');

const TIMEOUT_BETWEEN_EPOCHS_MS = 500;
const PORT = 8001;

// util function to sleep for a given ms
function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// Main function to start server, perform model training, and emit stats via the socket connection
async function run() {
  const port = process.env.PORT || PORT;
  const server = http.createServer();
  const io = socketio(server);

  server.listen(port, () => {
    console.log(`  > Running socket on port: ${port}`);
  });

  io.on('connection', (socket) => {
    socket.on('predictSample', async (sample) => {
      io.emit('predictResult', await sign_type.predictSample(sample));
    });
  });

  let numTrainingIterations = 10;
  for (var i = 0; i < numTrainingIterations; i++) {
    console.log(`Training iteration : ${i+1} / ${numTrainingIterations}`);
    await sign_type.model.fitDataset(sign_type.trainingDataASL, {epochs: 1});
    console.log('accuracyPerClass', await sign_type.evaluateASL(true));
    await sleep(TIMEOUT_BETWEEN_EPOCHS_MS);
  }

  io.emit('trainingComplete', true);
}

run();