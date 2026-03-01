// Global state
let regions = [];
let currentModel = null;
let currentAutomaton = null;
let currentTrajectory = null;
let currentSymbolic = null;
let currentController = null;  // Added missing variable

// Canvas for workspace drawing
const canvas = document.getElementById('workspace_canvas');
const ctx = canvas ? canvas.getContext('2d') : null;
let drawing = false;
let startX, startY;
let currentRect = null;

// Initialize on load
document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM loaded, initializing...');
    loadConfig();
    setupCanvas();
    updateButtonStates(); // Initial button states
});

// Update button states based on current data
function updateButtonStates() {
    const buildBtn = document.getElementById('buildBtn');
    const synthesizeBtn = document.getElementById('synthesizeBtn');
    const simulateBtn = document.getElementById('simulateBtn');
    const exportBtn = document.getElementById('exportBtn');
    const generateBtn = document.getElementById('generateBtn');

    // Build abstraction: needs model AND regions
    if (buildBtn) {
        buildBtn.disabled = !(currentModel && regions.length > 0);
    }

    // Synthesize: needs abstraction AND automaton
    if (synthesizeBtn) {
        synthesizeBtn.disabled = !(currentSymbolic && currentAutomaton);
    }

    // Simulate: needs controller
    if (simulateBtn) {
        simulateBtn.disabled = !currentController;
    }

    // Export: needs controller
    if (exportBtn) {
        exportBtn.disabled = !currentController;
    }

    // Generate automaton: needs regions
    if (generateBtn) {
        generateBtn.disabled = regions.length === 0;
    }

    console.log('Button states updated:', {
        model: !!currentModel,
        regions: regions.length,
        symbolic: !!currentSymbolic,
        automaton: !!currentAutomaton,
        controller: !!currentController
    });
}

// Load configuration
async function loadConfig() {
    try {
        console.log('Loading config...');
        const response = await fetch('/api/config');
        const data = await response.json();
        console.log('Config loaded:', data);

        const cpuInput = document.getElementById('cpu_cores');
        const gpuCheck = document.getElementById('use_gpu');
        const modelSelect = document.getElementById('model_select');

        if (cpuInput) cpuInput.value = data.hardware.num_cpu_cores;
        if (gpuCheck) gpuCheck.checked = data.hardware.use_gpu;

        // Populate model dropdown
        if (modelSelect) {
            modelSelect.innerHTML = '<option value="">Select a model...</option>';
            data.models.forEach(model => {
                const option = document.createElement('option');
                option.value = model;
                option.textContent = model;
                modelSelect.appendChild(option);
            });
        }
    } catch (error) {
        console.error('Error loading config:', error);
    }
}

// Update hardware config
async function updateConfig() {
    try {
        const config = {
            num_cpu_cores: parseInt(document.getElementById('cpu_cores').value || '4'),
            use_gpu: document.getElementById('use_gpu').checked
        };

        const response = await fetch('/api/config', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(config)
        });

        if (response.ok) {
            alert('✅ Configuration updated');
        }
    } catch (error) {
        console.error('Error updating config:', error);
        alert('❌ Error updating configuration');
    }
}

// Canvas setup
function setupCanvas() {
    if (!canvas) {
        console.error('Canvas not found');
        return;
    }
    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', endDrawing);
    canvas.addEventListener('mouseout', cancelDrawing);
    console.log('Canvas setup complete');
}

function startDrawing(e) {
    drawing = true;
    const rect = canvas.getBoundingClientRect();
    startX = e.clientX - rect.left;
    startY = e.clientY - rect.top;
}

function draw(e) {
    if (!drawing) return;

    const rect = canvas.getBoundingClientRect();
    const currentX = e.clientX - rect.left;
    const currentY = e.clientY - rect.top;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    redrawRegions();

    ctx.strokeStyle = 'blue';
    ctx.lineWidth = 2;
    ctx.strokeRect(startX, startY, currentX - startX, currentY - startY);

    currentRect = {
        x: Math.min(startX, currentX),
        y: Math.min(startY, currentY),
        width: Math.abs(currentX - startX),
        height: Math.abs(currentY - startY)
    };
}

function endDrawing() {
    if (!drawing || !currentRect) return;
    drawing = false;

    const regionName = prompt('Enter region name:', `region_${regions.length + 1}`);
    if (regionName) {
        const x1 = (currentRect.x / canvas.width) * 10;
        const y1 = (currentRect.y / canvas.height) * 10;
        const x2 = ((currentRect.x + currentRect.width) / canvas.width) * 10;
        const y2 = ((currentRect.y + currentRect.height) / canvas.height) * 10;

        regions.push({
            name: regionName,
            bounds: [[x1, x2], [y1, y2]],
            color: `hsl(${regions.length * 30}, 70%, 50%)`
        });

        redrawRegions();
        updateRegionList();
        updateButtonStates(); // Update buttons after adding region
    }

    currentRect = null;
}

function cancelDrawing() {
    drawing = false;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    redrawRegions();
}

function redrawRegions() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    regions.forEach(region => {
        const x1 = (region.bounds[0][0] / 10) * canvas.width;
        const x2 = (region.bounds[0][1] / 10) * canvas.width;
        const y1 = (region.bounds[1][0] / 10) * canvas.height;
        const y2 = (region.bounds[1][1] / 10) * canvas.height;

        ctx.fillStyle = region.color;
        ctx.globalAlpha = 0.3;
        ctx.fillRect(x1, y1, x2 - x1, y2 - y1);

        ctx.strokeStyle = region.color;
        ctx.lineWidth = 2;
        ctx.globalAlpha = 1;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

        ctx.fillStyle = 'black';
        ctx.font = '12px Arial';
        ctx.fillText(region.name, x1 + 5, y1 + 20);
    });
}

function clearCanvas() {
    regions = [];
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    updateRegionList();
    updateButtonStates(); // Update buttons after clearing
}

function updateRegionList() {
    const list = document.getElementById('region_list');
    if (!list) return;

    list.innerHTML = '<h4>Regions:</h4>';
    if (regions.length === 0) {
        list.innerHTML += '<p style="color: #999; font-style: italic;">No regions drawn yet</p>';
        return;
    }

    regions.forEach(region => {
        const div = document.createElement('div');
        div.textContent = `${region.name}: (${region.bounds[0][0].toFixed(1)}-${region.bounds[0][1].toFixed(1)}, ${region.bounds[1][0].toFixed(1)}-${region.bounds[1][1].toFixed(1)})`;
        div.style.color = region.color;
        div.style.fontWeight = 'bold';
        list.appendChild(div);
    });
}

// Select robot model
async function selectModel(event) {
    const modelSelect = document.getElementById('model_select');
    const modelName = modelSelect.value;

    if (!modelName) {
        alert('Please select a model');
        return;
    }

    const button = event.target;
    const originalText = button.textContent;
    button.textContent = '⏳ Loading...';
    button.disabled = true;

    const statusDiv = document.getElementById('model_status');
    if (statusDiv) {
        statusDiv.innerHTML = 'Loading model...';
        statusDiv.className = 'status-box loading';
    }

    try {
        console.log('Loading model:', modelName);
        const response = await fetch('/api/select_model', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                model_name: modelName,
                resolutions: [10, 10, 10]
            })
        });

        const data = await response.json();
        console.log('Model response:', data);

        if (data.status === 'success') {
            currentModel = {
                name: modelName,
                state_dim: data.state_dim,
                input_dim: data.input_dim,
                n_cells: data.n_cells,
                state_names: data.state_names,
                input_names: data.input_names
            };

            if (statusDiv) {
                statusDiv.innerHTML = `✅ Model loaded: ${data.state_dim}D, ${data.n_cells} cells`;
                statusDiv.className = 'status-box success';
            }

            updateInitialStateInputs(data.state_dim);
            updateButtonStates(); // This will enable Build button if regions exist

            alert(`✅ Model loaded successfully!`);
        } else {
            throw new Error(data.error || 'Failed to load model');
        }
    } catch (error) {
        console.error('Error:', error);
        if (statusDiv) {
            statusDiv.innerHTML = `❌ Error: ${error.message}`;
            statusDiv.className = 'status-box error';
        }
        alert('❌ Error loading model: ' + error.message);
    } finally {
        button.textContent = originalText;
        button.disabled = false;
    }
}

function updateInitialStateInputs(dim) {
    const container = document.getElementById('initial_state_inputs');
    if (!container) return;

    container.innerHTML = '<h4>Initial State:</h4>';

    if (!dim) return;

    for (let i = 0; i < dim; i++) {
        const div = document.createElement('div');

        const label = document.createElement('label');
        label.textContent = `x${i+1}:`;
        label.style.width = '30px';
        label.style.display = 'inline-block';

        const input = document.createElement('input');
        input.type = 'number';
        input.id = `init_x${i}`;
        input.value = '0';
        input.step = '0.1';
        input.style.width = '80px';
        input.style.marginLeft = '5px';

        div.appendChild(label);
        div.appendChild(input);
        container.appendChild(div);
    }
}

// Generate automaton from prompt
// Generate automaton from prompt
async function generateAutomaton() {
    const prompt = document.getElementById('prompt_input').value;
    if (!prompt) {
        alert('Please enter a prompt');
        return;
    }

    if (regions.length === 0) {
        alert('Please draw at least one region first');
        return;
    }

    const generateBtn = document.getElementById('generateBtn');
    const originalText = generateBtn.textContent;
    generateBtn.textContent = '⏳ Generating...';
    generateBtn.disabled = true;

    try {
        const regionNames = regions.map(r => r.name);
        console.log('Generating automaton for:', {prompt, regionNames});

        const response = await fetch('/api/generate_automaton', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                prompt: prompt,
                region_names: regionNames
            })
        });

        const data = await response.json();
        console.log('Generation response:', data);

        if (data.status === 'success') {
            // Store the automaton
            currentAutomaton = data.automaton;

            // Display it
            document.getElementById('automaton_display').style.display = 'block';
            document.getElementById('automaton_json').textContent =
                JSON.stringify(data.automaton, null, 2);

            // CRITICAL: Update button states
            updateButtonStates();

            console.log('✅ Automaton stored:', currentAutomaton);
            alert('✅ Automaton generated successfully!');
        } else {
            throw new Error(data.error || 'Generation failed');
        }
    } catch (error) {
        console.error('Error:', error);
        alert('❌ Error: ' + error.message);
    } finally {
        generateBtn.textContent = originalText;
        generateBtn.disabled = false;
    }
}
// Build abstraction
async function buildAbstraction(event) {
    if (!currentModel) {
        alert('❌ Please load a model first');
        return;
    }

    if (regions.length === 0) {
        alert('❌ Please draw at least one region');
        return;
    }

    const button = event.target;
    const originalText = button.textContent;
    button.textContent = '⏳ Building...';
    button.disabled = true;

    const statsDiv = document.getElementById('abstraction_stats');
    statsDiv.innerHTML = '<p>Building abstraction... (this may take a moment)</p>';

    try {
        const regionData = {};
        regions.forEach(region => {
            regionData[region.name] = region.bounds;
        });

        console.log('Building abstraction with regions:', regionData);

        const response = await fetch('/api/build_abstraction', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({regions: regionData})
        });

        const data = await response.json();
        console.log('Abstraction response:', data);

        if (data.status === 'success') {
            currentSymbolic = {
                n_cells: data.n_cells,
                n_inputs: data.n_inputs
            };

            statsDiv.innerHTML = `
                <div style="background: #d4edda; color: #155724; padding: 10px; border-radius: 4px;">
                    <p><strong>✅ Abstraction built successfully!</strong></p>
                    <p>📊 Cells: ${data.n_cells.toLocaleString()}</p>
                    <p>🎮 Inputs: ${data.n_inputs}</p>
                </div>
            `;

            updateButtonStates(); // This will enable Synthesize if automaton exists
        } else {
            throw new Error(data.error || 'Build failed');
        }
    } catch (error) {
        console.error('Error:', error);
        statsDiv.innerHTML = `
            <div style="background: #f8d7da; color: #721c24; padding: 10px; border-radius: 4px;">
                <p><strong>❌ Error:</strong> ${error.message}</p>
            </div>
        `;
    } finally {
        button.textContent = originalText;
        button.disabled = false;
    }
}

// Synthesize controller
async function synthesize(event) {
    if (!currentSymbolic) {
        alert('❌ Please build abstraction first');
        return;
    }

    if (!currentAutomaton) {
        alert('❌ Please generate automaton first');
        return;
    }

    const button = event.target;
    const originalText = button.textContent;
    button.textContent = '⏳ Synthesizing...';
    button.disabled = true;

    const statsDiv = document.getElementById('synthesis_stats');
    statsDiv.innerHTML = '<p>Synthesizing controller... (this may take a moment)</p>';

    try {
        console.log('Synthesizing controller...');
        const response = await fetch('/api/synthesize', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({type: 'safety'})
        });

        const data = await response.json();
        console.log('Synthesis response:', data);

        if (data.status === 'success') {
            currentController = true; // Mark that we have a controller

            statsDiv.innerHTML = `
                <div style="background: #d4edda; color: #155724; padding: 10px; border-radius: 4px;">
                    <p><strong>✅ Controller synthesized successfully!</strong></p>
                    <p>📊 Product States: ${data.n_product_states.toLocaleString()}</p>
                    <p>🎯 Winning States: ${data.n_winning_states.toLocaleString()}</p>
                </div>
            `;

            updateButtonStates(); // This will enable Simulate and Export
        } else {
            throw new Error(data.error || 'Synthesis failed');
        }
    } catch (error) {
        console.error('Error:', error);
        statsDiv.innerHTML = `
            <div style="background: #f8d7da; color: #721c24; padding: 10px; border-radius: 4px;">
                <p><strong>❌ Error:</strong> ${error.message}</p>
            </div>
        `;
    } finally {
        button.textContent = originalText;
        button.disabled = false;
    }
}

// Run simulation
// Run simulation
// Run simulation
async function runSimulation(event) {
    if (!currentModel) {
        alert('❌ Please load a model first');
        return;
    }

    if (!currentController) {
        alert('❌ Please synthesize a controller first');
        return;
    }

    const button = event.target;
    const originalText = button.textContent;
    button.textContent = '⏳ Simulating...';
    button.disabled = true;

    // Clear previous plot
    document.getElementById('plot_container').innerHTML = '<p>Running simulation...</p>';

    try {
        const dim = currentModel.state_dim;
        const initialState = [];
        for (let i = 0; i < dim; i++) {
            const input = document.getElementById(`init_x${i}`);
            initialState.push(input ? parseFloat(input.value) || 0 : 0);
        }

        console.log('Running simulation with initial state:', initialState);

        // Run simulation - MAKE SURE num_steps is 200
        const response = await fetch('/api/simulate', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                initial_state: initialState,
                num_steps: 200  // This MUST be 200
            })
        });

        const data = await response.json();
        console.log('Simulation response:', data);

        if (data.status === 'success') {
            // Store trajectory
            currentTrajectory = data.trajectory;
            console.log(`✅ Trajectory stored: ${currentTrajectory.length} points`);

            // Check if we have enough points
               if (currentTrajectory.length === 1) {
                console.log('⚠️ Only 1 point, duplicating for visualization');
                currentTrajectory.push(currentTrajectory[0]);
            }


            // Prepare region data
            const regionData = {};
            regions.forEach(region => {
                regionData[region.name] = region.bounds;
            });

            // Request visualization
            console.log('Requesting visualization...');

            const vizResponse = await fetch('/api/visualize', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    trajectory: currentTrajectory,
                    regions: regionData
                })
            });

            const vizData = await vizResponse.json();
            console.log('Visualization response:', vizData);

            if (vizData.status === 'success') {
                document.getElementById('plot_container').innerHTML =
                    `<img src="${vizData.image}" style="max-width:100%; border-radius: 4px; border: 1px solid #ddd;">`;
                console.log('✅ Plot displayed successfully');
            } else {
                throw new Error(vizData.error || 'Visualization failed');
            }
        } else {
            throw new Error(data.error || 'Simulation failed');
        }
    } catch (error) {
        console.error('❌ Error:', error);
        document.getElementById('plot_container').innerHTML =
            `<p style="color: red;">❌ Error: ${error.message}</p>`;
        alert('❌ Error: ' + error.message);
    } finally {
        button.textContent = originalText;
        button.disabled = false;
    }
}
// Export controller
async function exportController() {
    if (!currentController) {
        alert('❌ No controller to export');
        return;
    }

    try {
        console.log('Exporting controller...');
        const response = await fetch('/api/export_controller');
        const data = await response.json();

        const blob = new Blob([JSON.stringify(data, null, 2)], {type: 'application/json'});
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'controller.json';
        a.click();
        URL.revokeObjectURL(url);

        console.log('Controller exported successfully');
    } catch (error) {
        console.error('Error:', error);
        alert('❌ Error exporting controller: ' + error.message);
    }
}

// Helper function for drawing rectangle
function drawRectangle() {
    alert('Click and drag on the canvas to draw a region');
}

// Make functions globally available
window.selectModel = selectModel;
window.updateConfig = updateConfig;
window.generateAutomaton = generateAutomaton;
window.buildAbstraction = buildAbstraction;
window.synthesize = synthesize;
window.runSimulation = runSimulation;
window.exportController = exportController;
window.drawRectangle = drawRectangle;
window.clearCanvas = clearCanvas;