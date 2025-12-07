let emissionChart;

// Load ONNX model
let session;
ort.InferenceSession.create("emission_model.onnx").then(s => {
    session = s;
    console.log("ONNX model loaded successfully");
});

// Predict function
async function predict() {
    if (!session) {
        alert("Model is still loading... Please wait.");
        return;
    }

    let inputs = [
        Number(document.getElementById("pm25").value),
        Number(document.getElementById("pm10").value),
        Number(document.getElementById("no2").value),
        Number(document.getElementById("so2").value),
        Number(document.getElementById("waste").value),
        Number(document.getElementById("vehicles").value),
        Number(document.getElementById("ind").value),
        Number(document.getElementById("dom").value),
        Number(document.getElementById("pop").value),
    ];

    // Convert to ONNX tensor
    const tensor = new ort.Tensor("float32", Float32Array.from(inputs), [1, 9]);
    const feeds = { float_input: tensor };

    // Run model
    const results = await session.run(feeds);
    const prediction = results.variable.data[0];

    // Show prediction
    document.getElementById("result").innerText =
        `Predicted Emission Index: ${prediction.toFixed(2)}`;

    // Update chart
    updateChart(prediction);
}

// Chart function with premium white text
function updateChart(predValue) {
    const ctx = document.getElementById("emissionChart").getContext("2d");

    if (emissionChart) {
        emissionChart.data.labels.push(`Prediction ${emissionChart.data.labels.length + 1}`);
        emissionChart.data.datasets[0].data.push(predValue);
        emissionChart.update();
        return;
    }

    // FIRST TIME CHART CREATION
    emissionChart = new Chart(ctx, {
        type: "line",
        data: {
            labels: ["Prediction 1"],
            datasets: [{
                label: "Emission Index",
                data: [predValue],
                borderColor: "#00ffcc",
                backgroundColor: "rgba(0,255,204,0.25)",
                borderWidth: 3,
                tension: 0.35,
                pointRadius: 6,
                pointBackgroundColor: "#00ffcc",
                pointBorderColor: "#00332e",
                pointHoverRadius: 8
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    labels: {
                        color: "white",
                        font: { size: 14, weight: "500" }
                    }
                }
            },
            scales: {
                y: {
                    ticks: {
                        color: "white",
                        font: { size: 13 }
                    },
                    grid: {
                        color: "rgba(255,255,255,0.15)"
                    }
                },
                x: {
                    ticks: {
                        color: "white",
                        font: { size: 13 }
                    },
                    grid: {
                        color: "rgba(255,255,255,0.15)"
                    }
                }
            }
        }
    });
}
