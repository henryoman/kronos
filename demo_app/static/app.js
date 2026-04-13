const state = {
  demos: [],
  candles: [],
  prediction: [],
};

const elements = {
  select: document.querySelector("#demoSelect"),
  loadCandles: document.querySelector("#loadCandles"),
  runForecast: document.querySelector("#runForecast"),
  title: document.querySelector("#title"),
  side: document.querySelector("#side"),
  pUp: document.querySelector("#pUp"),
  forecastReturn: document.querySelector("#forecastReturn"),
  horizon: document.querySelector("#horizon"),
  summary: document.querySelector("#summary"),
  payload: document.querySelector("#payload"),
  chart: document.querySelector("#chart"),
};

function money(value) {
  if (!Number.isFinite(value)) return "-";
  return value.toLocaleString(undefined, { maximumFractionDigits: 4 });
}

function percent(value) {
  if (!Number.isFinite(value)) return "-";
  return `${(value * 100).toFixed(2)}%`;
}

function selectedDemo() {
  return state.demos.find((demo) => demo.id === elements.select.value);
}

function setBusy(isBusy) {
  elements.loadCandles.disabled = isBusy;
  elements.runForecast.disabled = isBusy;
  document.body.classList.toggle("busy", isBusy);
}

async function getJson(url, options = {}) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || response.statusText);
  }
  return data;
}

async function loadConfig() {
  const config = await getJson("/api/config");
  state.demos = config.demos;
  elements.select.innerHTML = state.demos
    .map((demo) => `<option value="${demo.id}">${demo.label}</option>`)
    .join("");
  updateTitle();
  await loadCandles();
}

function updateTitle() {
  const demo = selectedDemo();
  if (!demo) return;
  elements.title.textContent = `${demo.label}`;
  elements.summary.textContent = `${demo.provider}/${demo.market} ${demo.symbol} ${demo.interval}`;
}

async function loadCandles() {
  const demo = selectedDemo();
  if (!demo) return;
  setBusy(true);
  try {
    const data = await getJson(`/api/candles?id=${encodeURIComponent(demo.id)}`);
    state.candles = data.candles;
    state.prediction = [];
    drawChart();
    elements.payload.textContent = JSON.stringify(data.demo, null, 2);
    elements.summary.textContent = `${data.candles.length} closed candles loaded for ${demo.symbol}.`;
  } catch (error) {
    elements.summary.textContent = error.message;
  } finally {
    setBusy(false);
  }
}

async function runForecast() {
  const demo = selectedDemo();
  if (!demo) return;
  setBusy(true);
  try {
    elements.summary.textContent = "Loading Kronos and sampling forecast paths...";
    const data = await getJson("/api/forecast", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id: demo.id }),
    });
    state.candles = data.candles;
    state.prediction = data.prediction;
    elements.side.textContent = data.strategy.side.toUpperCase();
    elements.pUp.textContent = percent(data.forecast.p_up);
    elements.forecastReturn.textContent = percent(data.forecast.forecast_return);
    elements.horizon.textContent = data.forecast.horizon_label;
    elements.payload.textContent = JSON.stringify(data, null, 2);
    elements.summary.textContent = `${demo.symbol} close ${money(data.forecast.current_close)} -> mean final ${money(data.forecast.mean_final_close)} until ${data.forecast.forecast_until}.`;
    drawChart();
  } catch (error) {
    elements.summary.textContent = error.message;
  } finally {
    setBusy(false);
  }
}

function resizeCanvas(canvas) {
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.max(320, Math.floor(rect.width * ratio));
  canvas.height = Math.max(260, Math.floor(rect.height * ratio));
  return ratio;
}

function drawChart() {
  const canvas = elements.chart;
  const ratio = resizeCanvas(canvas);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);

  const width = canvas.width / ratio;
  const height = canvas.height / ratio;
  ctx.clearRect(0, 0, width, height);

  const candles = state.candles;
  const prediction = state.prediction;
  if (!candles.length) {
    drawEmpty(ctx, width, height);
    return;
  }

  const visibleCandles = candles.slice(-160);
  const all = [...visibleCandles, ...prediction];
  const highs = all.map((bar) => bar.high);
  const lows = all.map((bar) => bar.low);
  const min = Math.min(...lows);
  const max = Math.max(...highs);
  const pad = (max - min) * 0.08 || max * 0.01 || 1;
  const yMin = min - pad;
  const yMax = max + pad;

  const left = 62;
  const right = 24;
  const top = 24;
  const bottom = 44;
  const plotW = width - left - right;
  const plotH = height - top - bottom;
  const count = all.length;
  const slot = plotW / Math.max(count, 1);
  const bodyW = Math.max(3, Math.min(11, slot * 0.58));

  const y = (price) => top + ((yMax - price) / (yMax - yMin)) * plotH;
  const x = (index) => left + index * slot + slot / 2;

  drawGrid(ctx, width, height, left, right, top, bottom, yMin, yMax, y);

  visibleCandles.forEach((bar, index) => {
    drawCandle(ctx, x(index), y, bodyW, bar, false);
  });

  if (prediction.length) {
    const splitX = x(visibleCandles.length - 0.5);
    ctx.strokeStyle = "#ffcf5a";
    ctx.setLineDash([5, 7]);
    ctx.beginPath();
    ctx.moveTo(splitX, top);
    ctx.lineTo(splitX, top + plotH);
    ctx.stroke();
    ctx.setLineDash([]);

    prediction.forEach((bar, index) => {
      drawCandle(ctx, x(visibleCandles.length + index), y, bodyW, bar, true);
    });
  }
}

function drawGrid(ctx, width, height, left, right, top, bottom, yMin, yMax, y) {
  ctx.fillStyle = "#f5f2e8";
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = "rgba(31, 35, 29, 0.13)";
  ctx.lineWidth = 1;
  ctx.font = "12px Georgia, serif";
  ctx.fillStyle = "#4f5747";

  for (let i = 0; i <= 5; i += 1) {
    const price = yMin + ((yMax - yMin) * i) / 5;
    const py = y(price);
    ctx.beginPath();
    ctx.moveTo(left, py);
    ctx.lineTo(width - right, py);
    ctx.stroke();
    ctx.fillText(price.toFixed(2), 10, py + 4);
  }

  ctx.strokeStyle = "#1f231d";
  ctx.strokeRect(left, top, width - left - right, height - top - bottom);
}

function drawCandle(ctx, cx, y, bodyW, bar, isPrediction) {
  const up = bar.close >= bar.open;
  const color = isPrediction ? "#806a00" : up ? "#177a59" : "#b23b3b";
  const wickTop = y(bar.high);
  const wickBottom = y(bar.low);
  const bodyTop = y(Math.max(bar.open, bar.close));
  const bodyBottom = y(Math.min(bar.open, bar.close));
  const bodyH = Math.max(2, bodyBottom - bodyTop);

  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.globalAlpha = isPrediction ? 0.76 : 1;
  ctx.beginPath();
  ctx.moveTo(cx, wickTop);
  ctx.lineTo(cx, wickBottom);
  ctx.stroke();
  ctx.fillRect(cx - bodyW / 2, bodyTop, bodyW, bodyH);
  ctx.globalAlpha = 1;
}

function drawEmpty(ctx, width, height) {
  ctx.fillStyle = "#f5f2e8";
  ctx.fillRect(0, 0, width, height);
  ctx.fillStyle = "#1f231d";
  ctx.font = "20px Georgia, serif";
  ctx.fillText("No candles loaded", 32, 54);
}

elements.select.addEventListener("change", () => {
  updateTitle();
  loadCandles();
});
elements.loadCandles.addEventListener("click", loadCandles);
elements.runForecast.addEventListener("click", runForecast);
window.addEventListener("resize", drawChart);

loadConfig();
