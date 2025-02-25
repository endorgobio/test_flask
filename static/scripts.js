function updateGraph(dataset) {
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

// TODO: Solve the response data 
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

// TODO: Solve the response data 
// async function uploadFile() {
//     let fileInput = document.getElementById('file-input');
//     if (fileInput.files.length === 0) {
//         alert("Please select a file first!");
//         return;
//     }

//     let formData = new FormData();
//     formData.append("file", fileInput.files[0]);

//     try {
//         let response = await fetch("/upload", {
//             method: "POST",
//             body: formData
//         });

//         if (!response.ok) {
//             throw new Error("Error uploading file. Please try again.");
//         }

//         // let graph_json = await response.json();
//         let graph_json = await  JSON.parse(response)

//         Plotly.newPlot('graph', graph_json.data, graph_json.layout);
//     } catch (error) {
//         alert(error.message);
//     }
// }


// async function runModelAndDisplay() {
//     try {
//         let response = await fetch('/run_model', {  // Flask route
//             method: 'GET',
//             headers: { 'Content-Type': 'application/json' }
//         });

//         if (!response.ok) {
//             throw new Error('Network response was not ok');
//         }

//         let result = await response.json();  // Expecting JSON response
//         document.getElementById("model-output").value = result.output; // Access the 'output' key
//     } catch (error) {
//         console.error("Error running model:", error);
//         alert("Error running model. Please try again.");
//     }
// }

async function runModelWithFile() {
    let fileInput = document.getElementById('file-data');
    
    if (fileInput.files.length === 0) {
        alert("Please select a file first!");
        return;
    }

    let file = fileInput.files[0];
    let reader = new FileReader();

    reader.onload = async function(event) {
        let fileContent = event.target.result;  // Read file content

        try {
            let response = await fetch('/run_model', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ file_data: fileContent })  // Send file data
            });

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            let result = await response.json();
            document.getElementById("model-output").value = result.output;
        } catch (error) {
            console.error("Error running model:", error);
            alert("Error running model. Please try again.");
        }
    };

    reader.readAsText(file);  // Read file as text (works for JSON and CSV)
}


async function runSampleModel() {
    // Show loading indicator (optional)
    const button = document.querySelector('.btn-success');
    const originalText = button.innerText;
    button.innerText = 'Running...';
    button.disabled = true;

    try {
        // Gather input values
        const parameters = {
            container_value: parseFloat(document.getElementById('container_value').value),
            deposit: parseFloat(document.getElementById('deposit').value),
            clasification: parseFloat(document.getElementById('clasification').value),
            washing: parseFloat(document.getElementById('washing').value),
            transportation: parseFloat(document.getElementById('transportation').value)
        };

        // Send POST request to Flask backend
        const response = await fetch('/run_sample_model', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(parameters)
        });

        if (!response.ok) throw new Error('Network response was not ok');

        const data = await response.json();

        // Display result
        alert("Result: " + data.result);
    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred while running the model.');
    } finally {
        // Reset button state
        button.innerText = originalText;
        button.disabled = false;
    }
}

// document.addEventListener('DOMContentLoaded', function() {
//     const datasetSelect = document.getElementById('dataset-select');
//     datasetSelect.addEventListener('change', updateGraph);

//     // Load the initial graph
//     updateGraph();
// });


// document.addEventListener('DOMContentLoaded', function() {
//     const datasetSelect = document.getElementById('dataset-select');
//     datasetSelect.addEventListener('change', function() {
//         const selectedDataset = datasetSelect.value;
//         updateGraph(selectedDataset);
//     });

//     // Load the initial graph with the given initial dataset value
//     updateGraph(initialDataset);
// });
document.addEventListener('DOMContentLoaded', function () {
    // Render the Plotly graph
    Plotly.newPlot('graph', graph_json.data, graph_json.layout);
});


