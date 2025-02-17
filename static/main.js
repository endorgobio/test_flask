// Initialize the Plotly graph
let graphDiv = document.getElementById('plotly-graph');

// Function to update the graph
function updateGraph(datasetId) {
    // Fetch data from the Flask backend
    fetch(`/get_data/${datasetId}`)
        .then(response => response.json())
        .then(data => {
            // Create the 3D scatter plot
            let trace = {
                x: data.x,
                y: data.y,
                z: data.z,
                mode: 'markers',
                marker: {
                    size: 10,
                    color: 'blue'
                },
                type: 'scatter3d'
            };

            // Update the graph
            Plotly.newPlot(graphDiv, [trace], {
                margin: { l: 0, r: 0, b: 0, t: 0 },
                scene: {
                    xaxis: { title: "X-Axis" },
                    yaxis: { title: "Y-Axis" },
                    zaxis: { title: "Z-Axis" }
                }
            });
        });
}

// Event listener for the dropdown menu
document.getElementById('dataset-select').addEventListener('change', function() {
    let datasetId = this.value;
    updateGraph(datasetId);
});

// Initialize the graph with the first dataset
updateGraph("dataset1");