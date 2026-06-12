import * as ort from "onnxruntime-web";

// Force ONNX Runtime Web to load WASM runtime files from CDN
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";
ort.env.wasm.numThreads = 1;

// const MODEL_PATH = "/shuttlemodel.onnx";
// const IMG_SIZE = 640;
const MODEL_PATH = `${import.meta.env.BASE_URL}shuttlemodel-416.onnx`;
const IMG_SIZE = 416;
// const MODEL_PATH = "/shuttlemodel-.onnx";
// const IMG_SIZE = 320;
const NUM_CLASSES = 3;
const CONF_THRESHOLD = 0.35;
const IOU_THRESHOLD = 0.45;

const CLASS_NAMES = ["feather_new", "feather_old", "nylon"];

const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const statusText = document.getElementById("status");
const startBtn = document.getElementById("startBtn");

let session = null;
let running = false;

async function loadModel() {
  try {
    statusText.innerText = "Loading ONNX model...";

    // Helps avoid mobile/browser threading issues
    ort.env.wasm.numThreads = 1;

    session = await ort.InferenceSession.create(MODEL_PATH, {
      executionProviders: ["wasm"]
    });

    console.log("Model loaded successfully");
    console.log("Input names:", session.inputNames);
    console.log("Output names:", session.outputNames);

    statusText.innerText = "Model loaded. Tap Start Camera.";
  } catch (err) {
    console.error("Model failed to load:", err);
    statusText.innerText = "Model failed to load. Check browser console.";
  }
}

async function startCamera() {
  try {
    statusText.innerText = "Starting camera...";

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      statusText.innerText = "Camera API not available. Use HTTPS or localhost.";
      return;
    }

    let stream;

    // First try rear camera for phones/tablets
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: { ideal: "environment" },
          width: { ideal: 640 },
          height: { ideal: 640 }
        },
        audio: false
      });
    } catch (mobileCameraError) {
      console.warn("Rear camera failed, trying default camera:", mobileCameraError);

      // Fallback for Windows/laptop webcams
      stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: false
      });
    }

    video.srcObject = stream;

    video.onloadedmetadata = async () => {
      await video.play();

      running = true;

      if (session) {
        detectLoop();
        statusText.innerText = "Camera running. Detection started.";
      } else {
        statusText.innerText = "Camera running. Waiting for model to load.";
      }
    };
  } catch (err) {
    console.error("Camera failed:", err);
    statusText.innerText = `Camera failed: ${err.name} - ${err.message}`;
  }
}

function preprocess() {
  const tempCanvas = document.createElement("canvas");
  tempCanvas.width = IMG_SIZE;
  tempCanvas.height = IMG_SIZE;

  const tempCtx = tempCanvas.getContext("2d");
  tempCtx.drawImage(video, 0, 0, IMG_SIZE, IMG_SIZE);

  const imageData = tempCtx.getImageData(0, 0, IMG_SIZE, IMG_SIZE);
  const data = imageData.data;

  const floatData = new Float32Array(1 * 3 * IMG_SIZE * IMG_SIZE);

  for (let i = 0; i < IMG_SIZE * IMG_SIZE; i++) {
    const r = data[i * 4] / 255.0;
    const g = data[i * 4 + 1] / 255.0;
    const b = data[i * 4 + 2] / 255.0;

    floatData[i] = r;
    floatData[IMG_SIZE * IMG_SIZE + i] = g;
    floatData[2 * IMG_SIZE * IMG_SIZE + i] = b;
  }

  return new ort.Tensor("float32", floatData, [1, 3, IMG_SIZE, IMG_SIZE]);
}

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

function parseYOLOOutput(outputTensor) {
  const output = outputTensor.data;
  const dims = outputTensor.dims;

  let boxes = [];

  /*
    Common Ultralytics detection ONNX output:
    [1, 4 + num_classes, num_predictions]
    Example with 3 classes:
    [1, 7, 8400]
  */

  let channels = dims[1];
  let numPredictions = dims[2];

  const isTransposed = channels < numPredictions;

  for (let i = 0; i < numPredictions; i++) {
    let x, y, w, h;
    let classScores = [];

    if (isTransposed) {
      x = output[0 * numPredictions + i];
      y = output[1 * numPredictions + i];
      w = output[2 * numPredictions + i];
      h = output[3 * numPredictions + i];

      for (let c = 0; c < NUM_CLASSES; c++) {
        classScores.push(output[(4 + c) * numPredictions + i]);
      }
    } else {
      const offset = i * channels;
      x = output[offset + 0];
      y = output[offset + 1];
      w = output[offset + 2];
      h = output[offset + 3];

      for (let c = 0; c < NUM_CLASSES; c++) {
        classScores.push(output[offset + 4 + c]);
      }
    }

    let bestClass = 0;
    let bestScore = classScores[0];

    for (let c = 1; c < classScores.length; c++) {
      if (classScores[c] > bestScore) {
        bestScore = classScores[c];
        bestClass = c;
      }
    }

    const confidence = bestScore;

    if (confidence >= CONF_THRESHOLD) {
      const x1 = x - w / 2;
      const y1 = y - h / 2;
      const x2 = x + w / 2;
      const y2 = y + h / 2;

      boxes.push({
        x1,
        y1,
        x2,
        y2,
        confidence,
        classId: bestClass
      });
    }
  }

  return nonMaxSuppression(boxes, IOU_THRESHOLD);
}

function intersectionOverUnion(a, b) {
  const x1 = Math.max(a.x1, b.x1);
  const y1 = Math.max(a.y1, b.y1);
  const x2 = Math.min(a.x2, b.x2);
  const y2 = Math.min(a.y2, b.y2);

  const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);

  const areaA = Math.max(0, a.x2 - a.x1) * Math.max(0, a.y2 - a.y1);
  const areaB = Math.max(0, b.x2 - b.x1) * Math.max(0, b.y2 - b.y1);

  const union = areaA + areaB - intersection;

  return union === 0 ? 0 : intersection / union;
}

function nonMaxSuppression(boxes, iouThreshold) {
  boxes.sort((a, b) => b.confidence - a.confidence);

  const selected = [];

  while (boxes.length > 0) {
    const current = boxes.shift();
    selected.push(current);

    boxes = boxes.filter(box => {
      if (box.classId !== current.classId) return true;
      return intersectionOverUnion(current, box) < iouThreshold;
    });
  }

  return selected;
}

function drawDetections(detections) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const scaleX = canvas.width / IMG_SIZE;
  const scaleY = canvas.height / IMG_SIZE;

  ctx.lineWidth = 3;
  ctx.font = "18px Arial";

  for (const det of detections) {
    const x = det.x1 * scaleX;
    const y = det.y1 * scaleY;
    const w = (det.x2 - det.x1) * scaleX;
    const h = (det.y2 - det.y1) * scaleY;

    const label = `${CLASS_NAMES[det.classId]} ${(det.confidence * 100).toFixed(1)}%`;

    ctx.strokeStyle = "lime";
    ctx.fillStyle = "lime";

    ctx.strokeRect(x, y, w, h);
    ctx.fillText(label, x, Math.max(20, y - 6));
  }
}

async function detectLoop() {
  if (!running || !session) return;

  try {
    const inputTensor = preprocess();

    const feeds = {};
    feeds[session.inputNames[0]] = inputTensor;

    const results = await session.run(feeds);
    const outputName = session.outputNames[0];
    const outputTensor = results[outputName];

    const detections = parseYOLOOutput(outputTensor);
    drawDetections(detections);

    statusText.innerText = `Running. Detections: ${detections.length}`;
  } catch (err) {
    console.error(err);
    statusText.innerText = "Error during inference. Check console.";
  }

  requestAnimationFrame(detectLoop);
}

startBtn.addEventListener("click", startCamera);

loadModel();