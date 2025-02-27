// Event listener for DOMContentLoaded to render the initial Plotly graph
document.addEventListener('DOMContentLoaded', function () {
    // Render the Plotly graph with initial data
    Plotly.newPlot('graph', graph_json.data, graph_json.layout);
});

/**
 * Run the sample model and update the graph.
 */
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

        // Update the Plotly graph
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

/**
 * Update the Plotly graph with new data from the server.
 */
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

/**
 * Upload a file and update the graph and control values based on the uploaded data.
 */
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

        const data = await response.json();      
        const graph_json = JSON.parse(data.graph_json);        
        const controls_default = data.controls_default;

        Plotly.newPlot('graph', graph_json.data, graph_json.layout);

        // Update control column default values
        document.getElementById('container_value').value = controls_default['container_value'];
        document.getElementById('container_value').min = controls_default['container_value_min'];
        document.getElementById('container_value').max = controls_default['container_value_max'];
        document.getElementById('deposit').value = controls_default.deposit_value;
        document.getElementById('deposit').min = controls_default.deposit_min;  
        document.getElementById('deposit').max = controls_default.deposit_max;
        document.getElementById('clasification').value = controls_default.clasification_value;
        document.getElementById('clasification').min = controls_default.clasification_min;
        document.getElementById('clasification').max = controls_default.clasification_max;
        document.getElementById('washing').value = controls_default.washing_value;
        document.getElementById('washing').min = controls_default.washing_min;
        document.getElementById('washing').max = controls_default.washing_max;
        document.getElementById('transportation').value = controls_default.transportation_value;
        document.getElementById('transportation').min = controls_default.transportation_min;
        document.getElementById('transportation').max = controls_default.transportation_max;
    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred while uploading the file.');
    } finally {
        // Reset button state
        button.innerText = originalText;
        button.disabled = false;
    }
}



