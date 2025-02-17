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