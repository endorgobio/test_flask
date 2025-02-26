document.addEventListener('DOMContentLoaded', function () {
    // Render the Plotly graph
    Plotly.newPlot('graph', graph_json.data, graph_json.layout);
});

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

        // // Display result
        // alert("Result: " + data.layout);
        // Plotly.newPlot('graph', data.data, data.layout);
        updateGraph();
    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred while running the model.');
    } finally {
        // Reset button state
        button.innerText = originalText;
        button.disabled = false;
    }
}


// function updateGraph() {
//     $.ajax({
//         url: "/update_graph",
//         type: "POST",
//         contentType: "application/json",
//         success: function(response) {
//             let graph_json = JSON.parse(response);
//             Plotly.newPlot('graph', graph_json.data, graph_json.layout);
            
//         }
//     });
// }

async function updateGraph() {
    try {
        const response = await fetch('/update_graph', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });

        if (!response.ok) {
            throw new Error('Network response was not ok');
        }

        const graph_json = await response.json();
        Plotly.newPlot('graph', graph_json.data, graph_json.layout);
    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred while updating the graph.');
    }
}

async function uploadFile() {
    // Show loading indicator (optional)
    const button = document.getElementById('btn_load');
    const originalText = button.innerText;
    button.innerText = 'Loading...';
    button.disabled = true;

    try {
        let fileInput = document.getElementById('file-input');
        if (fileInput.files.length === 0) {
            alert("Please select a file first!");
            return;
        }

        let formData = new FormData();
        formData.append("file", fileInput.files[0]);

        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error('Network response was not ok');
        }

        const graph_json = await response.json();
        Plotly.newPlot('graph', graph_json.data, graph_json.layout);
    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred while uploading the file.');
    } finally {
        // Reset button state
        button.innerText = originalText;
        button.disabled = false;
    }
}




