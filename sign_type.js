const tf = require('@tensorflow/tfjs');
const fs = require('fs')
var parse = require('csv-parse');

// const dev = process.env.NODE_ENV !== 'production';
// const server = dev ? 'http://localhost:3000' : 'www.dupa.pl';

var parser = parse({columns: true}, function (err, records) {
	console.log(records);
});

// util function to normalize a value between a given range.
function normalize(value, min, max) {
    if (min === undefined || max === undefined) {
        return value;
    }
    return (value - min) / (max - min);
}

function normalizeASL(value) {
    return value / 255;
}

// data can be loaded from URLs or local file paths when running in Node.js. 
const TRAIN_DATA_PATH =
    'https://storage.googleapis.com/mlb-pitch-data/pitch_type_training_data.csv';
const TEST_DATA_PATH = 'https://storage.googleapis.com/mlb-pitch-data/pitch_type_test_data.csv';

// console.log(fs.createReadStream(__dirname + "/sign_mnist_train.csv").pipe(parser))

// const TRAIN_DATA_ASL_PATH = fs.createReadStream(__dirname + "/sign_mnist_train.csv").pipe(parser);

// const TEST_DATA_ASL_PATH = fs.createReadStream(__dirname + "/sign_mnist_test.csv").pipe(parser);

// const TRAIN_DATA_ASL_PATH = "C:\Users\michal\Documents\tensorflow\sign_mnist_train.csv";
// const TEST_DATA_ASL_PATH = "C:\Users\michal\Documents\tensorflow\sign_mnist_test.csv";

// const TRAIN_DATA_ASL_PATH = "./sign_mnist_train.csv";
// const TEST_DATA_ASL_PATH = "./sign_mnist_test.csv";

// const TRAIN_DATA_ASL_PATH = "file://./sign_mnist_train.csv";
// const TEST_DATA_ASL_PATH = "file://./sign_mnist_test.csv";

// const TRAIN_DATA_ASL_PATH = "file://C:\Users\michal\Documents\tensorflow\sign_mnist_train.csv";
// const TEST_DATA_ASL_PATH = "file://C:\Users\michal\Documents\tensorflow\sign_mnist_test.csv";

const TRAIN_DATA_ASL_PATH = "file://sign_mnist_train.csv";
const TEST_DATA_ASL_PATH = "file://sign_mnist_test.csv";

// Constants from training data
const VX0_MIN = -18.885;
const VX0_MAX = 18.065;
const VY0_MIN = -152.463;
const VY0_MAX = -86.374;
const VZ0_MIN = -15.5146078412997;
const VZ0_MAX = 9.974;
const AX_MIN = -48.0287647107959;
const AX_MAX = 30.592;
const AY_MIN = 9.397;
const AY_MAX = 49.18;
const AZ_MIN = -49.339;
const AZ_MAX = 2.95522851438373;
const START_SPEED_MIN = 59;
const START_SPEED_MAX = 104.4;

const NUM_PITCH_CLASSES = 7;
const TRAINING_DATA_LENGTH = 7000;
const TEST_DATA_LENGTH = 700;

const NUM_S_L_LETTERS = 26;
const TRAINING_DATA_LENGTH_ASL = 25455;
const TEST_DATA_LENGTH_ASL = 7172;

// Converts a row from the CSV into features and labels.
// Each feature field is normalized within training data constants
const csvTransform =
    ({ xs, ys }) => {
        const values = [
            normalize(xs.vx0, VX0_MIN, VX0_MAX),
            normalize(xs.vy0, VY0_MIN, VY0_MAX),
            normalize(xs.vz0, VZ0_MIN, VZ0_MAX), normalize(xs.ax, AX_MIN, AX_MAX),
            normalize(xs.ay, AY_MIN, AY_MAX), normalize(xs.az, AZ_MIN, AZ_MAX),
            normalize(xs.start_speed, START_SPEED_MIN, START_SPEED_MAX),
            xs.left_handed_pitcher
        ];
        return { xs: values, ys: ys.pitch_code };
    }

const csvTransformASL = ({ xs, ys }) => {
    const values = [];
    for (let i = 1; i <= 784; i++){
        values.push(normalizeASL(xs["pixel" + i]));
    }

    return { xs: values, ys: ys.label };

}

const trainingDataASL =
    tf.data.csv(TRAIN_DATA_ASL_PATH, { columnConfigs: { label: { isLabel: true } } })
        .map(csvTransformASL)
        .shuffle(TRAINING_DATA_LENGTH_ASL)
        .batch(100);

const trainingData =
    tf.data.csv(TRAIN_DATA_PATH, { columnConfigs: { pitch_code: { isLabel: true } } })
        .map(csvTransform)
        .shuffle(TRAINING_DATA_LENGTH)
        .batch(100);

// Load all training data in one batch to use for evaluation
const trainingValidationDataASL =
    tf.data.csv(TRAIN_DATA_ASL_PATH, { columnConfigs: { label: { isLabel: true } } })
        .map(csvTransformASL)
        .batch(TRAINING_DATA_LENGTH_ASL);

// Load all test data in one batch to use for evaluation
const testValidationDataASL =
    tf.data.csv(TEST_DATA_ASL_PATH, { columnConfigs: { label: { isLabel: true } } })
        .map(csvTransformASL)
        .batch(TEST_DATA_LENGTH_ASL);

// Load all training data in one batch to use for evaluation
const trainingValidationData =
    tf.data.csv(TRAIN_DATA_PATH, { columnConfigs: { pitch_code: { isLabel: true } } })
        .map(csvTransform)
        .batch(TRAINING_DATA_LENGTH);

// Load all test data in one batch to use for evaluation
const testValidationData =
    tf.data.csv(TEST_DATA_PATH, { columnConfigs: { pitch_code: { isLabel: true } } })
        .map(csvTransform)
        .batch(TEST_DATA_LENGTH);

let model = null



async function setModel(){
    return await tf.loadLayersModel("file://my-model-1")
}

try {
    if (fs.existsSync("file://my-model-1")) {
        model = tf.loadLayersModel("file://my-model-1")
    }
    else
        model = tf.sequential();
  } catch(err) {
    
    console.error(err)
}

model.add(tf.layers.dense({ units: 392, activation: 'relu', inputShape: [784] }));
model.add(tf.layers.dense({ units: 196, activation: 'relu' }));
model.add(tf.layers.dense({ units: 98, activation: 'relu' }));
model.add(tf.layers.dense({ units: NUM_S_L_LETTERS, activation: 'softmax' }));

// model.add(tf.layers.dense({ units: 250, activation: 'relu', inputShape: [8] }));
// model.add(tf.layers.dense({ units: 175, activation: 'relu' }));
// model.add(tf.layers.dense({ units: 150, activation: 'relu' }));
// model.add(tf.layers.dense({ units: NUM_PITCH_CLASSES, activation: 'softmax' }));

model.compile({
    optimizer: tf.train.adam(),
    loss: 'sparseCategoricalCrossentropy',
    metrics: ['accuracy']
});

async function evaluateASL(useTestData) {
    let results = {};
    await trainingValidationDataASL.forEachAsync(SLTypeBatch => {
        const values = model.predict(SLTypeBatch.xs).dataSync();
        const classSize = TRAINING_DATA_LENGTH_ASL / NUM_S_L_LETTERS;
        for (let i = 0; i < NUM_S_L_LETTERS; i++) {
            results[signFromClassNum(i)] = {
                training: calcSignClassEvalAsl(i, classSize, values)
            };
        }
    });

    if (useTestData) {
        await testValidationDataASL.forEachAsync(SLTypeBatch => {
            const values = model.predict(SLTypeBatch.xs).dataSync();
            const classSize = TEST_DATA_LENGTH_ASL / NUM_S_L_LETTERS;
            for (let i = 0; i < NUM_S_L_LETTERS; i++) {
                results[signFromClassNum(i)].validation =
                    calcSignClassEvalAsl(i, classSize, values);
            }
        });
    }
    return results;
}

function calcSignClassEvalAsl(signIndex, classSize, values) {
    // Output has 7 different class values for each pitch, offset based on
    // which pitch class (ordered by i)
    let index = (signIndex * classSize * NUM_S_L_LETTERS) + signIndex;
    let total = 0;
    for (let i = 0; i < classSize; i++) {
        total += values[index];
        index += NUM_S_L_LETTERS;
    }
    return total / classSize;
}

async function predictSampleASL(sample) {
    let result = model.predict(tf.tensor(sample, [1, sample.length])).arraySync();
    var maxValue = 0;
    var predictedSign = 25;
    for (var i = 0; i < NUM_S_L_LETTERS; i++) {
        if (result[0][i] > maxValue) {
            predictedSign = i;
            maxValue = result[0][i];
        }
    }
    return signFromClassNum(predictedSign);
}

// Returns pitch class evaluation percentages for training data 
// with an option to include test data
// async function evaluate(useTestData) {
//     let results = {};
//     await trainingValidationData.forEachAsync(pitchTypeBatch => {
//         const values = model.predict(pitchTypeBatch.xs).dataSync();
//         const classSize = TRAINING_DATA_LENGTH / NUM_PITCH_CLASSES;
//         for (let i = 0; i < NUM_PITCH_CLASSES; i++) {
//             results[pitchFromClassNum(i)] = {
//                 training: calcPitchClassEval(i, classSize, values)
//             };
//         }
//     });

//     if (useTestData) {
//         await testValidationData.forEachAsync(pitchTypeBatch => {
//             const values = model.predict(pitchTypeBatch.xs).dataSync();
//             const classSize = TEST_DATA_LENGTH / NUM_PITCH_CLASSES;
//             for (let i = 0; i < NUM_PITCH_CLASSES; i++) {
//                 results[pitchFromClassNum(i)].validation =
//                     calcPitchClassEval(i, classSize, values);
//             }
//         });
//     }
//     return results;
// }

// async function predictSample(sample) {
//     let result = model.predict(tf.tensor(sample, [1, sample.length])).arraySync();
//     var maxValue = 0;
//     var predictedPitch = 7;
//     for (var i = 0; i < NUM_PITCH_CLASSES; i++) {
//         if (result[0][i] > maxValue) {
//             predictedPitch = i;
//             maxValue = result[0][i];
//         }
//     }
//     return pitchFromClassNum(predictedPitch);
// }

// // Determines accuracy evaluation for a given pitch class by index
// function calcPitchClassEval(pitchIndex, classSize, values) {
//     // Output has 7 different class values for each pitch, offset based on
//     // which pitch class (ordered by i)
//     let index = (pitchIndex * classSize * NUM_PITCH_CLASSES) + pitchIndex;
//     let total = 0;
//     for (let i = 0; i < classSize; i++) {
//         total += values[index];
//         index += NUM_PITCH_CLASSES;
//     }
//     return total / classSize;
// }

// // Returns the string value for Baseball pitch labels
// function pitchFromClassNum(classNum) {
//     switch (classNum) {
//         case 0:
//             return 'Fastball (2-seam)';
//         case 1:
//             return 'Fastball (4-seam)';
//         case 2:
//             return 'Fastball (sinker)';
//         case 3:
//             return 'Fastball (cutter)';
//         case 4:
//             return 'Slider';
//         case 5:
//             return 'Changeup';
//         case 6:
//             return 'Curveball';
//         default:
//             return 'Unknown';
//     }
// }

function signFromClassNum(classNum) {
    switch (classNum) {
        case 0:
            return 'A';
        case 1:
            return 'B';
        case 2:
            return 'C';
        case 3:
            return 'D';
        case 4:
            return 'E';
        case 5:
            return 'F';
        case 6:
            return 'G';
        case 7:
            return 'H';
        case 8:
            return 'I';
        case 9:
            return 'J';
        case 10:
            return 'K';
        case 11:
            return 'L';
        case 12:
            return 'M';
        case 13:
            return 'N';
        case 14:
            return 'O';
        case 15:
            return 'P';
        case 16:
            return 'Q';
        case 17:
            return 'R';
        case 18:
            return 'S';
        case 19:
            return 'R';
        case 20:
            return 'U';
        case 21:
            return 'V';
        case 22:
            return 'W';
        case 23:
            return 'X';
        case 24:
            return 'Y';
        case 25:
            return 'Z';
    }
}

module.exports = {
    evaluateASL,
    model,
    signFromClassNum,
    predictSampleASL,
    testValidationDataASL,
    trainingDataASL,
    TEST_DATA_LENGTH_ASL
}