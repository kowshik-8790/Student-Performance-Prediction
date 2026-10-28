// Dynamic UI Controllers & Chart.js Visualizations

function updateVal(elementId, text) {
  const el = document.getElementById(elementId);
  if (el) {
    el.textContent = text;
  }
}

// File picker listener for batch upload
document.addEventListener("DOMContentLoaded", () => {
  const fileInput = document.getElementById("csv_file");
  const fileNameDisplay = document.getElementById("fileName");

  if (fileInput && fileNameDisplay) {
    fileInput.addEventListener("change", (e) => {
      if (e.target.files.length > 0) {
        fileNameDisplay.textContent = e.target.files[0].name;
      } else {
        fileNameDisplay.textContent = "No file selected";
      }
    });
  }

  // Dashboard Chart.js rendering
  if (typeof window.gradeData !== "undefined" && document.getElementById("gradeDistributionChart")) {
    renderGradeChart(window.gradeData);
  }

  if (typeof window.featureImportanceData !== "undefined" && document.getElementById("featureImportanceChart")) {
    renderFeatureImportanceChart(window.featureImportanceData);
  }
});

function renderGradeChart(gradeData) {
  const ctx = document.getElementById("gradeDistributionChart").getContext("2d");
  const labels = Object.keys(gradeData);
  const data = Object.values(gradeData);

  const colors = {
    "A": "#10b981",
    "B": "#0284c7",
    "C": "#f59e0b",
    "D": "#ea580c",
    "F": "#ef4444"
  };

  const bgColors = labels.map(g => colors[g] || "#6366f1");

  new Chart(ctx, {
    type: "doughnut",
    data: {
      labels: labels.map(g => `Grade ${g}`),
      datasets: [{
        data: data,
        backgroundColor: bgColors,
        borderWidth: 2,
        borderColor: "#ffffff"
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: "bottom", labels: { boxWidth: 12 } },
        tooltip: {
          callbacks: {
            label: (ctx) => ` ${ctx.label}: ${ctx.raw} students`
          }
        }
      },
      cutout: "60%"
    }
  });
}

function renderFeatureImportanceChart(featureData) {
  const ctx = document.getElementById("featureImportanceChart").getContext("2d");

  // Take top 6 drivers
  const entries = Object.entries(featureData).slice(0, 6);
  const labels = entries.map(e => e[0].replace(/_/g, " ").replace(/\b\w/g, l => l.toUpperCase()));
  const values = entries.map(e => e[1]);

  new Chart(ctx, {
    type: "bar",
    data: {
      labels: labels,
      datasets: [{
        label: "Relative Predictive Importance (%)",
        data: values,
        backgroundColor: "#4f46e5",
        borderRadius: 6
      }]
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => ` Importance: ${ctx.raw}%`
          }
        }
      },
      scales: {
        x: {
          grid: { color: "#e2e8f0" },
          ticks: {
            callback: (val) => val + "%"
          }
        },
        y: {
          grid: { display: false }
        }
      }
    }
  });
}
