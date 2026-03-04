// Global state
let regions = [];
let currentModel = null;
let currentAutomaton = null;
let currentTrajectory = null;
let currentSymbolic = null;
let currentController = null;
let regionsByCoords = []; // For high-dimensional models

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
    updateButtonStates();

    // Initialize with 2D default for custom model
    if (document.getElementById('state_dim')) {
        updateCustomInputs();
    }
});

// ==================== UTILITY FUNCTIONS ====================

// Update button states based on current data
// Update button states based on current data
function updateButtonStates() {
    console.log('🔄 updateButtonStates called');

    const buildBtn = document.getElementById('buildBtn');
    const synthesizeBtn = document.getElementById('synthesizeBtn');
    const simulateBtn = document.getElementById('simulateBtn');
    const exportBtn = document.getElementById('exportBtn');
    const generateBtn = document.getElementById('generateBtn');

    // FORCE ENABLE ALL BUTTONS FOR TESTING
    if (buildBtn) {
        buildBtn.disabled = false;
        console.log('buildBtn ENABLED');
    }
    if (synthesizeBtn) {
        synthesizeBtn.disabled = false;
        console.log('synthesizeBtn ENABLED');
    }
    if (simulateBtn) {
        simulateBtn.disabled = false;
        console.log('simulateBtn ENABLED');
    }
    if (exportBtn) {
        exportBtn.disabled = false;
        console.log('exportBtn ENABLED');
    }
    if (generateBtn) {
        generateBtn.disabled = false;
        console.log('generateBtn ENABLED');
    }
}
// ==================== CONFIGURATION ====================

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
                if (model !== 'custom') { // Don't show 'custom' in dropdown
                    const option = document.createElement('option');
                    option.value = model;
                    option.textContent = model;
                    modelSelect.appendChild(option);
                }
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

// ==================== CANVAS FUNCTIONS (2D) ====================

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
        updateButtonStates();
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
    updateButtonStates();
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

// ==================== MODEL SELECTION ====================

// Toggle between predefined and custom model
function toggleModelType() {
    const modelType = document.querySelector('input[name="model_type"]:checked').value;
    const predefinedDiv = document.getElementById('predefined_model_div');
    const customDiv = document.getElementById('custom_model_div');

    if (modelType === 'predefined') {
        predefinedDiv.style.display = 'block';
        customDiv.style.display = 'none';
        // Show workspace for predefined (assuming 2D)
        document.getElementById('workspace_section').style.display = 'block';
        document.getElementById('high_dim_regions').style.display = 'none';
    } else {
        predefinedDiv.style.display = 'none';
        customDiv.style.display = 'block';
        updateCustomInputs();
    }
}

// Update custom model inputs based on dimensions
function updateCustomInputs() {
    const stateDim = parseInt(document.getElementById('state_dim').value) || 2;
    const inputDim = parseInt(document.getElementById('input_dim').value) || 1;

    // Update equations
    const equationsDiv = document.getElementById('equations_list');
    if (equationsDiv) {
        equationsDiv.innerHTML = '';
        for (let i = 0; i < stateDim; i++) {
            const div = document.createElement('div');
            div.style.margin = '8px 0';
            div.innerHTML = `
                <label>x${i}' = </label>
                <input type="text" id="eq_${i}" placeholder="e.g., x0 + u0*cos(x2)" style="width: 80%;">
            `;
            equationsDiv.appendChild(div);
        }
    }

    // Update bounds
    const boundsDiv = document.getElementById('bounds_list');
    if (boundsDiv) {
        boundsDiv.innerHTML = '';
        for (let i = 0; i < stateDim; i++) {
            const div = document.createElement('div');
            div.style.margin = '8px 0';
            div.style.display = 'flex';
            div.style.gap = '10px';
            div.innerHTML = `
                <label style="width: 60px;">x${i}:</label>
                <input type="number" id="bound_low_${i}" placeholder="min" value="-10" style="width: 100px;">
                <span>to</span>
                <input type="number" id="bound_high_${i}" placeholder="max" value="10" style="width: 100px;">
            `;
            boundsDiv.appendChild(div);
        }
    }

    // Update inputs
    const inputsDiv = document.getElementById('inputs_list');
    if (inputsDiv) {
        inputsDiv.innerHTML = '';
        for (let i = 0; i < inputDim; i++) {
            const div = document.createElement('div');
            div.style.margin = '8px 0';
            div.innerHTML = `
                <label>u${i} values (comma-separated):</label>
                <input type="text" id="input_vals_${i}" placeholder="e.g., -1,0,1" value="0,1" style="width: 100%;">
            `;
            inputsDiv.appendChild(div);
        }
    }

    // Update disturbance bounds
    const distDiv = document.getElementById('disturbance_list');
    if (distDiv) {
        distDiv.innerHTML = '';
        for (let i = 0; i < stateDim; i++) {
            const div = document.createElement('div');
            div.style.margin = '8px 0';
            div.innerHTML = `
                <label>Disturbance bound for x${i}:</label>
                <input type="number" id="dist_${i}" value="0.01" step="0.001" min="0" style="width: 100px;">
            `;
            distDiv.appendChild(div);
        }
    }

    // Create resolution inputs
    createResolutionInputs(stateDim);

    // Show/hide region definition based on dimension
    const highDimRegions = document.getElementById('high_dim_regions');
    const workspaceSection = document.getElementById('workspace_section');

    if (stateDim > 2) {
        if (highDimRegions) highDimRegions.style.display = 'block';
        if (workspaceSection) workspaceSection.style.display = 'none';
    } else {
        if (highDimRegions) highDimRegions.style.display = 'none';
        if (workspaceSection) workspaceSection.style.display = 'block';
    }
}

// Create resolution inputs
function createResolutionInputs(dim) {
    const container = document.getElementById('resolution_inputs');
    if (!container) return;

    container.innerHTML = '';

    for (let i = 0; i < dim; i++) {
        const div = document.createElement('div');
        div.style.margin = '8px 0';
        div.style.display = 'flex';
        div.style.alignItems = 'center';

        const label = document.createElement('label');
        label.textContent = `Dimension ${i+1}:`;
        label.style.width = '100px';

        const input = document.createElement('input');
        input.type = 'number';
        input.id = `res_${i}`;
        input.value = '10';
        input.min = '2';
        input.max = '200';
        input.style.width = '80px';
        input.oninput = updateCellEstimate;

        div.appendChild(label);
        div.appendChild(input);
        container.appendChild(div);
    }

    document.getElementById('resolution_container').style.display = 'block';
    updateCellEstimate();
}

// Update cell estimate
function updateCellEstimate() {
    const inputs = document.querySelectorAll('[id^="res_"]');
    let total = 1;
    inputs.forEach(input => {
        total *= parseInt(input.value) || 10;
    });
    const cellCount = document.getElementById('cell_count');
    if (cellCount) {
        cellCount.textContent = total.toLocaleString();
    }
}

// Set resolution presets
function setResolution(res) {
    const inputs = document.querySelectorAll('[id^="res_"]');
    inputs.forEach(input => {
        input.value = res;
    });
    updateCellEstimate();
}

// Load model (predefined or custom)
async function loadModel() {
    const modelType = document.querySelector('input[name="model_type"]:checked').value;

    if (modelType === 'predefined') {
        await selectModel(event);
    } else {
        await loadCustomModel();
    }
}

// Select predefined model
// Select predefined model
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

    // Get resolutions
    const resolutionInputs = document.querySelectorAll('[id^="res_"]');
    const resolutions = [];
    resolutionInputs.forEach(input => {
        resolutions.push(parseInt(input.value) || 10);
    });

    try {
        console.log('Loading model:', modelName, 'resolutions:', resolutions);

        const response = await fetch('/api/select_model', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                model_name: modelName,
                resolutions: resolutions
            })
        });

        const data = await response.json();
        console.log('Model response:', data);

        if (data.status === 'success') {
            // Store model info - MAKE SURE THIS IS SET CORRECTLY
            currentModel = {
                name: modelName,
                state_dim: data.state_dim,
                input_dim: data.input_dim,
                n_cells: data.n_cells,
                state_names: data.state_names,
                input_names: data.input_names
            };

            console.log('✅ currentModel set:', currentModel);

            if (statusDiv) {
                statusDiv.innerHTML = `✅ Model loaded: ${data.state_dim}D, ${data.n_cells.toLocaleString()} cells`;
                statusDiv.className = 'status-box success';
            }

            updateInitialStateInputs(data.state_dim);

            // CRITICAL: Update button states after model is loaded
            updateButtonStates();

            // Force check regions
            console.log('Regions after model load:', regions);
            console.log('Regions length:', regions.length);

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
// Load custom model
async function loadCustomModel() {
    const name = document.getElementById('custom_model_name').value || 'custom_robot';
    const stateDim = parseInt(document.getElementById('state_dim').value) || 2;
    const inputDim = parseInt(document.getElementById('input_dim').value) || 1;

    // Get equations
    const equations = [];
    for (let i = 0; i < stateDim; i++) {
        const eq = document.getElementById(`eq_${i}`)?.value;
        if (!eq) {
            alert(`Please enter equation for x${i}'`);
            return;
        }
        equations.push(eq);
    }

    // Get state bounds
    const stateBounds = [];
    for (let i = 0; i < stateDim; i++) {
        const low = parseFloat(document.getElementById(`bound_low_${i}`).value);
        const high = parseFloat(document.getElementById(`bound_high_${i}`).value);
        if (isNaN(low) || isNaN(high)) {
            alert(`Please enter valid bounds for x${i}`);
            return;
        }
        stateBounds.push([low, high]);
    }

    // Get input values
    const inputValues = [];
    for (let i = 0; i < inputDim; i++) {
        const valsStr = document.getElementById(`input_vals_${i}`).value;
        const vals = valsStr.split(',').map(v => parseFloat(v.trim())).filter(v => !isNaN(v));
        if (vals.length === 0) {
            alert(`Please enter valid input values for u${i}`);
            return;
        }
        inputValues.push(vals);
    }

    // Generate all combinations of inputs (Cartesian product)
    // In loadCustomModel function, replace the inputs creation part:

    // Generate all combinations of inputs (Cartesian product)
    const allInputs = [];
    function generateCombinations(arrays, current = [], index = 0) {
        if (index === arrays.length) {
            // Make sure to create a copy of current
            allInputs.push([...current]);
            return;
        }
        for (const val of arrays[index]) {
            generateCombinations(arrays, [...current, val], index + 1);
        }
    }
    generateCombinations(inputValues);

    console.log(`Generated ${allInputs.length} input combinations`);

    // Get disturbance bounds
    const distBounds = [];
    for (let i = 0; i < stateDim; i++) {
        const val = parseFloat(document.getElementById(`dist_${i}`).value);
        distBounds.push(isNaN(val) ? 0.01 : val);
    }

    // Get resolutions
    const resolutionInputs = document.querySelectorAll('[id^="res_"]');
    const resolutions = [];
    resolutionInputs.forEach(input => {
        resolutions.push(parseInt(input.value) || 10);
    });

    // Determine which regions to use
    let regionData = {};
    if (stateDim > 2) {
        regionsByCoords.forEach(region => {
            regionData[region.name] = region.bounds;
        });
    } else {
        regions.forEach(region => {
            regionData[region.name] = region.bounds;
        });
    }

    // Prepare custom model data
    const customModelData = {
        name: name,
        state_dim: stateDim,
        input_dim: inputDim,
        equations: equations,
        state_bounds: stateBounds,
        inputs: allInputs,
        disturbance_bounds: distBounds,
        resolutions: resolutions,
        regions: regionData
    };

    console.log('Loading custom model:', customModelData);

    const button = document.getElementById('loadModelBtn');
    const originalText = button.textContent;
    button.textContent = '⏳ Loading...';
    button.disabled = true;

    const statusDiv = document.getElementById('model_status');
    if (statusDiv) {
        statusDiv.innerHTML = 'Loading custom model...';
        statusDiv.className = 'status-box loading';
    }

    try {
        const response = await fetch('/api/load_custom_model', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(customModelData)
        });

        const data = await response.json();
        console.log('Custom model response:', data);

        if (data.status === 'success') {
            currentModel = {
                name: name,
                state_dim: stateDim,
                input_dim: inputDim,
                n_cells: data.n_cells,
                state_names: data.state_names || Array(stateDim).fill().map((_, i) => `x${i}`),
                input_names: data.input_names || Array(inputDim).fill().map((_, i) => `u${i}`)
            };

            if (statusDiv) {
                statusDiv.innerHTML = `✅ Custom model loaded: ${stateDim}D, ${data.n_cells.toLocaleString()} cells`;
                statusDiv.className = 'status-box success';
            }

            updateInitialStateInputs(stateDim);
            updateButtonStates();

            alert('✅ Custom model loaded successfully!');
        } else {
            throw new Error(data.error || 'Failed to load custom model');
        }
    } catch (error) {
        console.error('Error:', error);
        if (statusDiv) {
            statusDiv.innerHTML = `❌ Error: ${error.message}`;
            statusDiv.className = 'status-box error';
        }
        alert('❌ Error: ' + error.message);
    } finally {
        button.textContent = originalText;
        button.disabled = false;
    }
}

// ==================== REGION DEFINITION FOR HIGH-D MODELS ====================

// Add region by coordinates for high-dimensional models
// Add region by coordinates for high-dimensional models
function addRegionByCoords() {
    const name = document.getElementById('region_name_input').value;
    if (!name) {
        alert('Please enter a region name');
        return;
    }

    const stateDim = parseInt(document.getElementById('state_dim').value) || 2;
    const bounds = [];

    for (let i = 0; i < stateDim; i++) {
        const low = prompt(`Enter lower bound for dimension ${i}:`, '0');
        const high = prompt(`Enter upper bound for dimension ${i}:`, '10');
        if (low === null || high === null) return;
        bounds.push([parseFloat(low), parseFloat(high)]);
    }

    regionsByCoords.push({
        name: name,
        bounds: bounds,
        color: `hsl(${regionsByCoords.length * 30}, 70%, 50%)`
    });

    updateRegionCoordsList();
    document.getElementById('region_name_input').value = '';

    // IMPORTANT: Update button states after adding region
    updateButtonStates();
}

function updateRegionCoordsList() {
    const list = document.getElementById('region_coords_list');
    if (!list) return;

    list.innerHTML = '<h4>Regions:</h4>';

    if (regionsByCoords.length === 0) {
        list.innerHTML += '<p style="color: #999; font-style: italic;">No regions defined</p>';
        return;
    }

    regionsByCoords.forEach((region, idx) => {
        const div = document.createElement('div');
        div.style.color = region.color;
        div.style.padding = '5px';
        div.style.borderBottom = '1px solid #ddd';
        div.style.display = 'flex';
        div.style.justifyContent = 'space-between';
        div.style.alignItems = 'center';

        let boundsText = region.name + ': ';
        region.bounds.forEach((b, i) => {
            boundsText += `x${i}[${b[0].toFixed(1)}-${b[1].toFixed(1)}] `;
        });

        const textSpan = document.createElement('span');
        textSpan.textContent = boundsText;

        const deleteBtn = document.createElement('button');
        deleteBtn.textContent = '×';
        deleteBtn.style.padding = '0 8px';
        deleteBtn.style.marginLeft = '10px';
        deleteBtn.onclick = () => {
            regionsByCoords.splice(idx, 1);
            updateRegionCoordsList();
        };

        div.appendChild(textSpan);
        div.appendChild(deleteBtn);
        list.appendChild(div);
    });
}

// ==================== INITIAL STATE INPUTS ====================

function updateInitialStateInputs(dim) {
    const container = document.getElementById('initial_state_inputs');
    if (!container) return;

    container.innerHTML = '<h4>Initial State:</h4>';

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

// ==================== AUTOMATON GENERATION ====================

// ==================== AUTOMATON GENERATION ====================

async function generateAutomaton() {
    const prompt = document.getElementById('prompt_input').value;
    if (!prompt) {
        alert('Please enter a prompt');
        return;
    }

    // Determine which regions to use
    let regionNames = [];

    if (currentModel) {
        // Model is loaded
        console.log('Model loaded:', currentModel.name, 'state_dim:', currentModel.state_dim);

        // Check if it's a 2D model or differential drive (which is 3D but uses 2D canvas)
        if (currentModel.name === "differential_drive" || currentModel.state_dim <= 2) {
            // Use canvas regions for 2D and differential drive models
            regionNames = regions.map(r => r.name);
            console.log('Using canvas regions for 2D/predefined model:', regionNames);
        } else {
            // High-dimensional custom model - use coordinate regions
            regionNames = regionsByCoords.map(r => r.name);
            console.log('Using coordinate regions for high-dim model:', regionNames);
        }
    } else {
        // No model loaded yet, check the selected model type
        const modelType = document.querySelector('input[name="model_type"]:checked')?.value;
        console.log('No model loaded, model type:', modelType);

        if (modelType === 'custom') {
            const stateDim = parseInt(document.getElementById('state_dim')?.value) || 2;
            if (stateDim > 2) {
                regionNames = regionsByCoords.map(r => r.name);
            } else {
                regionNames = regions.map(r => r.name);
            }
        } else {
            // Predefined model selected but not loaded - assume 2D canvas regions
            regionNames = regions.map(r => r.name);
        }
    }

    if (regionNames.length === 0) {
        alert('Please define at least one region first');
        return;
    }

    const generateBtn = document.getElementById('generateBtn');
    const originalText = generateBtn.textContent;
    generateBtn.textContent = '⏳ Generating...';
    generateBtn.disabled = true;

    try {
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
            currentAutomaton = data.automaton;
            document.getElementById('automaton_display').style.display = 'block';
            document.getElementById('automaton_json').textContent =
                JSON.stringify(data.automaton, null, 2);

            updateButtonStates();
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
// ==================== ABSTRACTION BUILDING ====================

// ==================== ABSTRACTION BUILDING ====================

async function buildAbstraction(event) {
    if (!currentModel) {
        alert('❌ Please load a model first');
        return;
    }

    // Determine which regions to use
    let regionData = {};
    const stateDim = currentModel.state_dim;

    console.log('Building abstraction for model:', currentModel.name, 'state_dim:', stateDim);

    // Check if it's a 2D model or differential drive (which is 3D but uses 2D canvas)
    if (currentModel.name === "differential_drive" || stateDim <= 2) {
        // Use canvas regions for 2D and differential drive models
        regions.forEach(region => {
            regionData[region.name] = region.bounds;
        });
        console.log('Using canvas regions for abstraction:', Object.keys(regionData));
    } else {
        // High-dimensional custom model - use coordinate regions
        regionsByCoords.forEach(region => {
            regionData[region.name] = region.bounds;
        });
        console.log('Using coordinate regions for abstraction:', Object.keys(regionData));
    }

    if (Object.keys(regionData).length === 0) {
        alert('❌ Please define at least one region');
        return;
    }

    const button = event.target;
    const originalText = button.textContent;
    button.textContent = '⏳ Building...';
    button.disabled = true;

    const statsDiv = document.getElementById('abstraction_stats');
    statsDiv.innerHTML = '<p>Building abstraction... (this may take a moment)</p>';

    try {
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

            updateButtonStates();
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
// ==================== CONTROLLER SYNTHESIS ====================

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
            body: JSON.stringify({type: 'automaton'})
        });

        const data = await response.json();
        console.log('Synthesis response:', data);

        if (data.status === 'success') {
            currentController = true;

            statsDiv.innerHTML = `
                <div style="background: #d4edda; color: #155724; padding: 10px; border-radius: 4px;">
                    <p><strong>✅ Controller synthesized successfully!</strong></p>
                    <p>📊 Product States: ${data.n_product_states.toLocaleString()}</p>
                    <p>🎯 Winning States: ${data.n_winning_states.toLocaleString()}</p>
                </div>
            `;

            updateButtonStates();
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

// ==================== SIMULATION ====================

// ==================== SIMULATION ====================

// ==================== SIMULATION ====================

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

    document.getElementById('plot_container').innerHTML = '<p>Running simulation...</p>';

    try {
        const dim = currentModel.state_dim;
        const initialState = [];
        for (let i = 0; i < dim; i++) {
            const input = document.getElementById(`init_x${i}`);
            initialState.push(input ? parseFloat(input.value) || 0 : 0);
        }

        console.log('Running simulation with initial state:', initialState);

        const response = await fetch('/api/simulate', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                initial_state: initialState,
                num_steps: 200
            })
        });

        const data = await response.json();
        console.log('Simulation response:', data);

        if (data.status === 'success') {
            currentTrajectory = data.trajectory;

            // Prepare region data
            let regionData = {};
            if (currentModel.name === "differential_drive" || currentModel.state_dim <= 2) {
                regions.forEach(region => {
                    regionData[region.name] = region.bounds;
                });
            } else {
                regionsByCoords.forEach(region => {
                    if (region.bounds.length >= 2) {
                        regionData[region.name] = [region.bounds[0], region.bounds[1]];
                    }
                });
            }

            // Get visualization
            const vizResponse = await fetch('/api/visualize', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    trajectory: currentTrajectory,
                    regions: regionData
                })
            });

            const vizData = await vizResponse.json();

            if (vizData.status === 'success') {
                document.getElementById('plot_container').innerHTML =
                    `<img src="${vizData.image}" style="max-width:100%; border-radius: 4px; border: 1px solid #ddd;">`;
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

// Add this NEW function to find a valid start
async function findValidStart() {
    if (!currentController) {
        alert('Please synthesize a controller first');
        return;
    }

    const button = document.getElementById('findStartBtn');
    if (button) {
        button.textContent = '🔍 Finding...';
        button.disabled = true;
    }

    try {
        const response = await fetch('/api/find_valid_start', {
            method: 'POST'
        });

        const data = await response.json();

        if (data.status === 'success') {
            // Update input fields with the center of the valid cell
            for (let i = 0; i < data.center.length; i++) {
                const input = document.getElementById(`init_x${i}`);
                if (input) {
                    input.value = data.center[i].toFixed(2);
                }
            }
            alert(`✅ Found valid start: cell ${data.cell_idx}, state ${data.auto_state}`);
        } else {
            alert('❌ No valid start found: ' + (data.error || 'Unknown error'));
        }
    } catch (error) {
        console.error('Error:', error);
        alert('❌ Error finding valid start');
    } finally {
        if (button) {
            button.textContent = '🎯 Find Valid Start';
            button.disabled = false;
        }
    }
}
// ==================== EXPORT ====================

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

// ==================== HELPER FUNCTIONS ====================

function drawRectangle() {
    alert('Click and drag on the canvas to draw a region');
}

// ==================== MAKE FUNCTIONS GLOBALLY AVAILABLE ====================

window.selectModel = selectModel;
window.updateConfig = updateConfig;
window.generateAutomaton = generateAutomaton;
window.buildAbstraction = buildAbstraction;
window.synthesize = synthesize;
window.runSimulation = runSimulation;
window.exportController = exportController;
window.drawRectangle = drawRectangle;
window.clearCanvas = clearCanvas;
window.toggleModelType = toggleModelType;
window.updateCustomInputs = updateCustomInputs;
window.addRegionByCoords = addRegionByCoords;
window.setResolution = setResolution;
window.loadModel = loadModel;