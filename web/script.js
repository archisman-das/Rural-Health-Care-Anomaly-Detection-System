(function () {
  const demoData = {
    dashboardState: "Demo data loaded",
    lastUpdated: new Date().toLocaleString(),
    summary: [
      { label: "Patients scored", value: "2,184", detail: "across the latest monitoring window" },
      { label: "High-risk alerts", value: "74", detail: "alerts above the ensemble threshold" },
      { label: "Feedback reviews", value: "131", detail: "clinician true/false-positive tags" },
      { label: "Agreement rate", value: "82%", detail: "three-model consensus on the same patient" },
    ],
    riskTiles: [
      { label: "Low risk", value: "1,428", band: "risk-low" },
      { label: "Medium risk", value: "682", band: "risk-medium" },
      { label: "High risk", value: "74", band: "risk-high" },
      { label: "Trend shift", value: "+11%", band: "risk-medium" },
      { label: "False positives", value: "19", band: "risk-low" },
      { label: "Readmissions", value: "43", band: "risk-high" },
    ],
    riskLegend: [
      { title: "Low", color: "#8ff1cf", detail: "Routine follow-up. Scores remain below 0.4 and no urgent alert is fired." },
      { title: "Medium", color: "#ffcb6d", detail: "Closer review recommended. Scores are between 0.4 and 0.7." },
      { title: "High", color: "#ff7f96", detail: "Immediate attention. Final ensemble score is at or above 0.7." },
    ],
    modelRows: [
      {
        model: "Isolation Forest",
        color: "#62d4ff",
        precision: 0.81,
        recall: 0.76,
        f1: 0.78,
        auc: 0.86,
        auprc: 0.79,
        latency: "4.2 ms",
        ram: "38 MB",
      },
      {
        model: "Autoencoder",
        color: "#8ff1cf",
        precision: 0.84,
        recall: 0.80,
        f1: 0.82,
        auc: 0.89,
        auprc: 0.84,
        latency: "6.8 ms",
        ram: "92 MB",
      },
      {
        model: "Deep SVDD",
        color: "#ffcb6d",
        precision: 0.82,
        recall: 0.78,
        f1: 0.80,
        auc: 0.88,
        auprc: 0.82,
        latency: "7.1 ms",
        ram: "105 MB",
      },
      {
        model: "Ensemble",
        color: "#ff7f96",
        precision: 0.87,
        recall: 0.83,
        f1: 0.85,
        auc: 0.92,
        auprc: 0.89,
        latency: "8.9 ms",
        ram: "118 MB",
      },
    ],
    trend: [
      { label: "Week 1", value: 0.28 },
      { label: "Week 2", value: 0.31 },
      { label: "Week 3", value: 0.37 },
      { label: "Week 4", value: 0.33 },
      { label: "Week 5", value: 0.44 },
      { label: "Week 6", value: 0.48 },
      { label: "Week 7", value: 0.51 },
      { label: "Week 8", value: 0.57 },
    ],
    distribution: [4, 7, 11, 15, 19, 16, 12, 9, 5, 2],
    features: [
      { name: "age × glucose", value: 0.92, source: "age, glucose_fasting_mg_dl", type: "interaction" },
      { name: "7-day mean heart rate", value: 0.82, source: "heart_rate_bpm", type: "rolling mean" },
      { name: "30-day std blood pressure", value: 0.74, source: "systolic_bp_mmhg, diastolic_bp_mmhg", type: "rolling std" },
      { name: "electrolyte imbalance", value: 0.68, source: "na_mmol_l, k_mmol_l, ca_mg_dl", type: "clinical" },
    ],
    agreement: [
      { title: "All three models flagged", value: "58 alerts", detail: "Highest-confidence group for clinician prioritization.", score: 0.91 },
      { title: "Two-model agreement", value: "119 alerts", detail: "Useful for secondary review and active learning.", score: 0.74 },
      { title: "Single-model dissent", value: "34 alerts", detail: "Typically monitored for model drift or noise.", score: 0.41 },
    ],
    feedback: [
      { title: "True positives", value: "96", detail: "Clinicians confirmed the alert was valid.", score: 0.78, positive: true },
      { title: "False positives", value: "35", detail: "Alerts that should be down-weighted in retraining.", score: 0.39, positive: false },
    ],
    runtime: [
      { title: "Inference latency", metric: "8.9 ms", detail: "Ensemble scoring per patient on the reference device." },
      { title: "Training time", metric: "21 min", detail: "Full pipeline fit on the current tabular dataset." },
      { title: "Model size", metric: "24 MB", detail: "Serialized production bundle size." },
      { title: "RAM usage", metric: "118 MB", detail: "Peak memory footprint while scoring." },
    ],
  };

  const state = { data: demoData };

  const els = {};

  function $(id) {
    return document.getElementById(id);
  }

  function clamp(value, min, max) {
    return Math.min(max, Math.max(min, value));
  }

  function formatPercent(value) {
    return `${Math.round(value * 100)}%`;
  }

  function renderSummary() {
    const grid = $("summary-grid");
    grid.innerHTML = "";
    state.data.summary.forEach((item) => {
      const card = document.createElement("article");
      card.className = "metric-card";
      card.innerHTML = `
        <div class="metric-card__label">${item.label}</div>
        <div class="metric-card__value">${item.value}</div>
        <div class="metric-card__detail">${item.detail}</div>
      `;
      grid.appendChild(card);
    });
  }

  function renderRiskMap() {
    const map = $("patient-risk-map");
    const legend = $("risk-legend");
    map.innerHTML = "";
    legend.innerHTML = "";
    state.data.riskTiles.forEach((tile) => {
      const el = document.createElement("div");
      el.className = `risk-tile ${tile.band}`;
      el.innerHTML = `
        <div class="risk-tile__label">${tile.label}</div>
        <div class="risk-tile__value">${tile.value}</div>
      `;
      map.appendChild(el);
    });

    state.data.riskLegend.forEach((item) => {
      const el = document.createElement("article");
      el.className = "legend-item";
      el.innerHTML = `
        <div class="legend-item__title">
          <span class="legend-dot" style="background:${item.color}"></span>
          ${item.title}
        </div>
        <div class="legend-item__copy">${item.detail}</div>
      `;
      legend.appendChild(el);
    });
  }

  function drawAxes(ctx, width, height, padding) {
    ctx.strokeStyle = "rgba(166, 189, 220, 0.16)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padding, height - padding);
    ctx.lineTo(width - padding, height - padding);
    ctx.moveTo(padding, padding);
    ctx.lineTo(padding, height - padding);
    ctx.stroke();
  }

  function drawTrendChart() {
    const canvas = $("trend-chart");
    const ctx = canvas.getContext("2d");
    const { width, height } = canvas;
    const data = state.data.trend;
    const padding = 42;

    ctx.clearRect(0, 0, width, height);
    const bg = ctx.createLinearGradient(0, 0, 0, height);
    bg.addColorStop(0, "rgba(98, 212, 255, 0.14)");
    bg.addColorStop(1, "rgba(8, 15, 28, 0.02)");
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, width, height);
    drawAxes(ctx, width, height, padding);

    const values = data.map((d) => d.value);
    const min = 0;
    const max = 1;
    const innerW = width - padding * 2;
    const innerH = height - padding * 2;
    const stepX = innerW / Math.max(1, data.length - 1);

    ctx.beginPath();
    data.forEach((point, index) => {
      const x = padding + index * stepX;
      const y = height - padding - ((point.value - min) / (max - min)) * innerH;
      if (index === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    ctx.strokeStyle = "#62d4ff";
    ctx.lineWidth = 3;
    ctx.stroke();

    ctx.fillStyle = "#8ff1cf";
    data.forEach((point, index) => {
      const x = padding + index * stepX;
      const y = height - padding - ((point.value - min) / (max - min)) * innerH;
      ctx.beginPath();
      ctx.arc(x, y, 4.5, 0, Math.PI * 2);
      ctx.fill();
    });

    ctx.fillStyle = "#9fb2ca";
    ctx.font = "12px Aptos, Segoe UI, sans-serif";
    data.forEach((point, index) => {
      if (index % 2 === 0) {
        const x = padding + index * stepX;
        ctx.fillText(point.label, x - 18, height - 16);
      }
    });
    ctx.fillText("0.0", 8, height - padding + 4);
    ctx.fillText("1.0", 8, padding + 4);
  }

  function drawDistributionChart() {
    const canvas = $("distribution-chart");
    const ctx = canvas.getContext("2d");
    const { width, height } = canvas;
    const values = state.data.distribution;
    const padding = 42;
    const innerW = width - padding * 2;
    const innerH = height - padding * 2;
    const barW = innerW / values.length - 6;
    const max = Math.max(...values);

    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "rgba(8, 15, 28, 0.02)";
    ctx.fillRect(0, 0, width, height);
    drawAxes(ctx, width, height, padding);

    values.forEach((value, index) => {
      const x = padding + index * (innerW / values.length) + 3;
      const h = (value / max) * innerH;
      const y = height - padding - h;
      const gradient = ctx.createLinearGradient(0, y, 0, height - padding);
      gradient.addColorStop(0, "rgba(143, 241, 207, 0.95)");
      gradient.addColorStop(1, "rgba(98, 212, 255, 0.65)");
      ctx.fillStyle = gradient;
      ctx.fillRect(x, y, barW, h);
    });
  }

  function renderModelTable() {
    const body = $("model-table-body");
    body.innerHTML = "";
    state.data.modelRows.forEach((row) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td><span class="model-badge"><span class="model-dot" style="background:${row.color}"></span>${row.model}</span></td>
        <td>${formatPercent(row.precision)}</td>
        <td>${formatPercent(row.recall)}</td>
        <td>${formatPercent(row.f1)}</td>
        <td>${formatPercent(row.auc)}</td>
        <td>${formatPercent(row.auprc)}</td>
        <td>${row.latency}</td>
        <td>${row.ram}</td>
      `;
      body.appendChild(tr);
    });
  }

  function renderFeatures() {
    const list = $("feature-list");
    list.innerHTML = "";
    state.data.features.forEach((feature) => {
      const el = document.createElement("article");
      el.className = "feature-card";
      el.innerHTML = `
        <div class="feature-card__top">
          <div>
            <div class="feature-card__name">${feature.name}</div>
            <div class="feature-card__source">${feature.type}</div>
          </div>
          <div class="pill">${formatPercent(feature.value)}</div>
        </div>
        <div class="bar" aria-hidden="true"><span style="width:${clamp(feature.value, 0.05, 1) * 100}%"></span></div>
        <div class="feature-card__source">${feature.source}</div>
      `;
      list.appendChild(el);
    });
  }

  function renderAgreement() {
    const stack = $("agreement-stack");
    const feedback = $("feedback-panel");
    stack.innerHTML = "";
    feedback.innerHTML = "";

    state.data.agreement.forEach((item) => {
      const el = document.createElement("article");
      el.className = "agreement-card";
      el.innerHTML = `
        <div class="agreement-card__top">
          <div>
            <div class="feature-card__name">${item.title}</div>
            <div class="agreement-card__detail">${item.detail}</div>
          </div>
          <div class="pill">${item.value}</div>
        </div>
        <div class="bar" aria-hidden="true"><span style="width:${clamp(item.score, 0.05, 1) * 100}%"></span></div>
      `;
      stack.appendChild(el);
    });

    const feedbackGrid = document.createElement("div");
    feedbackGrid.className = "feedback-grid";
    state.data.feedback.forEach((item) => {
      const el = document.createElement("article");
      el.className = `feedback-card ${item.positive ? "feedback-card--positive" : "feedback-card--negative"}`;
      el.innerHTML = `
        <div class="agreement-card__top">
          <div>
            <div class="feature-card__name">${item.title}</div>
            <div class="agreement-card__detail">${item.detail}</div>
          </div>
          <div class="pill">${item.value}</div>
        </div>
        <div class="bar" aria-hidden="true"><span style="width:${clamp(item.score, 0.05, 1) * 100}%"></span></div>
      `;
      feedbackGrid.appendChild(el);
    });
    feedback.appendChild(feedbackGrid);
  }

  function renderRuntime() {
    const grid = $("runtime-grid");
    grid.innerHTML = "";
    state.data.runtime.forEach((item) => {
      const el = document.createElement("article");
      el.className = "runtime-item";
      el.innerHTML = `
        <div class="runtime-item__top">
          <div class="feature-card__name">${item.title}</div>
        </div>
        <div class="runtime-item__metric">${item.metric}</div>
        <div class="runtime-item__detail">${item.detail}</div>
      `;
      grid.appendChild(el);
    });
  }

  function renderMeta() {
    $("dashboard-status").textContent = state.data.dashboardState;
    $("last-updated").textContent = `Last updated: ${state.data.lastUpdated}`;
  }

  function renderAll() {
    renderMeta();
    renderSummary();
    renderRiskMap();
    renderModelTable();
    renderFeatures();
    renderAgreement();
    renderRuntime();
    drawTrendChart();
    drawDistributionChart();
  }

  async function loadDashboardData() {
    try {
      const response = await fetch("./dashboard-data.json", { cache: "no-store" });
      if (!response.ok) {
        throw new Error(`dashboard-data.json returned ${response.status}`);
      }
      const data = await response.json();
      state.data = {
        ...demoData,
        ...data,
      };
      state.data.dashboardState = data.dashboardState || "Live dashboard data loaded";
      state.data.lastUpdated = data.lastUpdated || new Date().toLocaleString();
    } catch (error) {
      state.data = demoData;
    }
    renderAll();
  }

  document.addEventListener("DOMContentLoaded", () => {
    els.reloadButton = $("reload-button");
    if (els.reloadButton) {
      els.reloadButton.addEventListener("click", () => loadDashboardData());
    }
  });
})();
