function updateGraph() {
    let dataset = $("#dataset-select").val();
    $.ajax({
        url: "/update_graph",
        type: "POST",
        contentType: "application/json",
        data: JSON.stringify({ dataset: dataset }),
        success: function(response) {
            let graph_json = JSON.parse(response);
            Plotly.newPlot('graph', graph_json.data, graph_json.layout);
        }
    });
}

// function updateGraph() {
//     let dataset = document.getElementById("dataset-select").value;

//     fetch("/update_graph", {
//         method: "POST",
//         headers: {
//             "Content-Type": "application/json"
//         },
//         body: JSON.stringify({ dataset: dataset })
//     })
//     .then(response => response.json()) // Parse the JSON response
//     .then(graph_json => {
        
//         console.log("Graph JSON Response:", graph_json);
//         // Check if the response contains valid data
//         if (graph_json && graph_json.data && Array.isArray(graph_json.data) && graph_json.data.length > 0) {
//             console.log("Graph Data:", graph_json.data);
//             console.log("Graph Layout:", graph_json.layout);

//             // Ensure the container is cleared before creating the plot
//             const graphContainer = document.getElementById('graph');
//             graphContainer.innerHTML = ''; // Clear the container before plotting

//             // Create the plot with the correct data and layout
//             Plotly.newPlot('graph', graph_json.data, graph_json.layout);
//         } else {
//             console.error("Invalid graph data or layout:", graph_json);
//         }
//     })
//     .catch(error => {
//         console.error("Error updating graph:", error);
//     });
// }


function uploadFile() {
    let fileInput = document.getElementById('file-input');
    if (fileInput.files.length === 0) {
        alert("Please select a file first!");
        return;
    }

    let formData = new FormData();
    formData.append("file", fileInput.files[0]);

    $.ajax({
        url: "/upload",
        type: "POST",
        data: formData,
        processData: false,
        contentType: false,
        success: function(response) {
            let graph_json = JSON.parse(response);
            Plotly.newPlot('graph', graph_json.data, graph_json.layout);
        },
        error: function() {
            alert("Error uploading file. Please try again.");
        }
    });
}

$(document).ready(function() {
    updateGraph();
});

document.addEventListener('DOMContentLoaded', function() {
    const datasetSelect = document.getElementById('dataset-select');
    datasetSelect.addEventListener('change', updateGraph);

    // Load the initial graph
    updateGraph();
});


// // Initialize the Plotly graph
// let graphDiv = document.getElementById('plotly-graph');

// // Function to update the graph
// function updateGraph(datasetId) {
//     // Fetch data from the Flask backend
//     fetch(`/get_data/${datasetId}`)
//         .then(response => response.json())
//         .then(data => {
//             // Create the 3D scatter plot
//             let trace = {
//                 x: data.x,
//                 y: data.y,
//                 z: data.z,
//                 mode: 'markers',
//                 marker: {
//                     size: 10,
//                     color: 'blue'
//                 },
//                 type: 'scatter3d'
//             };

//             // Update the graph
//             Plotly.newPlot(graphDiv, [trace], {
//                 margin: { l: 0, r: 0, b: 0, t: 0 },
//                 scene: {
//                     xaxis: { title: "X-Axis" },
//                     yaxis: { title: "Y-Axis" },
//                     zaxis: { title: "Z-Axis" }
//                 }
//             });
//         });
// }

// // Event listener for the dropdown menu
// document.getElementById('dataset-select').addEventListener('change', function() {
//     let datasetId = this.value;
//     updateGraph(datasetId);
// });

// // Initialize the graph with the first dataset
// updateGraph("dataset1");