"""Main application with web interface."""
from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
import json
import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for Flask
import matplotlib.pyplot as plt
import os
from pathlib import Path

from config import Config
from models import MODEL_REGISTRY
from partition import Partition
from abstraction import AbstractionBuilder
from automaton import Automaton, ProductSystem
from synthesis import SynthesisEngine, SymbolicController
from llm_integration import LLMInterface, prompt_to_automaton
from simulation import Simulator
from visualization import Visualizer
from parallel import HybridBackend

# Setup paths for templates
BASE_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = BASE_DIR / 'templates'
STATIC_DIR = BASE_DIR / 'static'

app = Flask(__name__,
            template_folder=str(TEMPLATE_DIR),
            static_folder=str(STATIC_DIR))
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')

# Global state
current_config = Config()
current_model = None
current_partition = None
current_symbolic = None
current_automaton = None
current_product = None
current_controller = None
current_llm = None


@app.route('/')
def index():
    """Render main page."""
    return render_template('index.html')


@app.route('/api/config', methods=['GET', 'POST'])
def handle_config():
    """Get or update configuration."""
    global current_config, current_llm

    if request.method == 'GET':
        return jsonify({
            'hardware': {
                'num_cpu_cores': current_config.hardware.num_cpu_cores,
                'use_gpu': current_config.hardware.use_gpu
            },
            'models': list(MODEL_REGISTRY.keys()),
            'llm_providers': ['openrouter', 'groq', 'openai', 'anthropic']
        })
    else:
        data = request.json
        if data:
            # Update hardware config
            if 'hardware' in data:
                current_config.hardware.num_cpu_cores = data['hardware'].get('num_cpu_cores', 4)
                current_config.hardware.use_gpu = data['hardware'].get('use_gpu', False)

            # Update LLM config
            if 'llm' in data:
                current_config.llm.provider = data['llm'].get('provider', 'openrouter')
                current_config.llm.model = data['llm'].get('model', 'deepseek/deepseek-chat-v3-0324:free')
                current_config.llm.api_key = data['llm'].get('api_key', os.environ.get('OPENROUTER_API_KEY'))
                current_config.llm.temperature = data['llm'].get('temperature', 0.1)

                # Reinitialize LLM with new config
                current_llm = LLMInterface({
                    'provider': current_config.llm.provider,
                    'model': current_config.llm.model,
                    'api_key': current_config.llm.api_key,
                    'temperature': current_config.llm.temperature
                })

        return jsonify({'status': 'success'})


@app.route('/api/workspace', methods=['POST'])
def upload_workspace():
    """Upload workspace drawing."""
    data = request.json
    regions = data.get('regions', {})
    bounds = data.get('bounds', {})

    # Store region definitions
    current_config.workspace_bounds = bounds

    # Store regions for later use
    if not hasattr(current_config, 'regions'):
        current_config.regions = {}
    current_config.regions = regions

    return jsonify({
        'status': 'success',
        'region_names': list(regions.keys())
    })


@app.route('/api/select_model', methods=['POST'])
@app.route('/api/select_model', methods=['POST'])
def select_model():
    """Select robot model."""
    global current_model, current_partition
    import time
    start_total = time.time()

    data = request.json
    model_name = data.get('model_name')
    resolutions = data.get('resolutions', [100, 100, 100])

    if model_name not in MODEL_REGISTRY:
        return jsonify({'error': f'Model {model_name} not found'}), 400

    try:
        # Time model creation
        t0 = time.time()
        model_factory = MODEL_REGISTRY[model_name]
        current_model = model_factory()
        t1 = time.time()
        print(f"⏱️ Model creation: {t1 - t0:.3f}s")

        # Get state bounds
        bounds = current_model.get_state_bounds()

        # Time partition creation
        t2 = time.time()
        current_partition = Partition(bounds, resolutions=resolutions)
        t3 = time.time()
        print(f"⏱️ Partition creation: {t3 - t2:.3f}s")

        # Calculate total cells (now fast because we don't generate them)
        total_cells = len(current_partition)

        total_time = time.time() - start_total
        print(f"⏱️ TOTAL LOAD TIME: {total_time:.3f}s")

        return jsonify({
            'status': 'success',
            'state_dim': current_model.state_dim,
            'input_dim': current_model.input_dim,
            'n_cells': total_cells,
            'state_names': current_model.state_names,
            'input_names': current_model.input_names,
            'load_time': total_time
        })

    except Exception as e:
        print(f"❌ Error in select_model: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate_automaton', methods=['POST'])
def generate_automaton():
    """Generate automaton from prompt."""
    global current_automaton, current_llm

    data = request.json
    prompt = data.get('prompt')
    region_names = data.get('region_names', [])

    if not prompt:
        return jsonify({'error': 'No prompt provided'}), 400

    try:
        # Initialize LLM if not already done
        if current_llm is None:
            llm_config = {
                'provider': current_config.llm.provider,
                'model': current_config.llm.model,
                'api_key': current_config.llm.api_key or os.environ.get('OPENROUTER_API_KEY'),
                'temperature': current_config.llm.temperature
            }
            current_llm = LLMInterface(llm_config)

        # Test connection first
        if not current_llm.test_connection():
            return jsonify({'error': 'LLM connection failed. Check your API key.'}), 500

        # Generate automaton
        automaton_data = current_llm.generate_automaton(prompt, region_names)

        if automaton_data:
            # Convert to Automaton object
            current_automaton = Automaton.from_json(automaton_data)
            return jsonify({
                'status': 'success',
                'automaton': current_automaton.to_json()
            })
        else:
            return jsonify({'error': 'Failed to generate automaton'}), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/build_abstraction', methods=['POST'])
@app.route('/api/build_abstraction', methods=['POST'])
def build_abstraction():
    """Build symbolic abstraction."""
    global current_symbolic

    if current_model is None:
        return jsonify({'error': 'No model selected'}), 400

    if current_partition is None:
        return jsonify({'error': 'No partition created'}), 400

    try:
        # Get regions from request
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        region_defs = data.get('regions', {})
        print(f"Received regions: {region_defs}")  # Debug log

        # Create parallel backend
        parallel = HybridBackend(current_config.hardware)

        # Build abstraction
        builder = AbstractionBuilder(current_model, current_partition, parallel)
        current_symbolic = builder.build_successors(progress_bar=False)

        # Add labelling from regions if available
        if region_defs:
            # Convert region format if needed
            formatted_regions = {}
            for name, bounds in region_defs.items():
                # Ensure bounds are in the right format
                if isinstance(bounds, list) and len(bounds) >= 2:
                    formatted_regions[name] = bounds
                else:
                    print(f"⚠️ Invalid bounds for region {name}: {bounds}")

            if formatted_regions:
                builder.add_labelling(current_symbolic, formatted_regions)

        return jsonify({
            'status': 'success',
            'n_cells': current_symbolic.n_cells,
            'n_inputs': current_symbolic.n_inputs
        })

    except Exception as e:
        print(f"❌ Error building abstraction: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/synthesize', methods=['POST'])
@app.route('/api/synthesize', methods=['POST'])
def synthesize():
    """Synthesize controller."""
    global current_product, current_controller

    if current_symbolic is None:
        return jsonify({'error': 'No symbolic model built'}), 400

    if current_automaton is None:
        return jsonify({'error': 'No automaton generated'}), 400

    try:
        print(f"🤖 Synthesizing controller with automaton: {current_automaton}")

        # Create product system
        current_product = ProductSystem(current_automaton, current_symbolic)

        # Create synthesis engine
        parallel = HybridBackend(current_config.hardware)
        engine = SynthesisEngine(current_product, current_symbolic, parallel)

        # For automaton specifications, we want to reach accepting states
        # Find accepting states in the product
        accepting_states = set()
        for state in current_product.states:
            if state.auto_state in current_automaton.accepting:
                accepting_states.add(state)

        print(f"🎯 Found {len(accepting_states)} accepting states")

        if accepting_states:
            # Synthesize reachability controller to reach accepting states
            current_controller = engine.synthesize_reachability(accepting_states)
        else:
            # Fallback to safety on all states
            all_states = set(current_product.states)
            current_controller = engine.synthesize_safety(all_states)

        return jsonify({
            'status': 'success',
            'n_product_states': current_product.n_states,
            'n_winning_states': len(current_controller.winning_states)
        })

    except Exception as e:
        print(f"❌ Synthesis error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/simulate', methods=['POST'])
def simulate():
    """Run simulation."""
    data = request.json
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    initial_state = np.array(data.get('initial_state', [0, 0, 0]))
    num_steps = data.get('num_steps', 100)
    noise_scale = data.get('noise_scale', 0.0)

    if current_controller is None:
        return jsonify({'error': 'No controller synthesized'}), 400

    if current_model is None:
        return jsonify({'error': 'No model selected'}), 400

    if current_symbolic is None:
        return jsonify({'error': 'No symbolic model built'}), 400

    try:
        # Create simulator
        simulator = Simulator(current_model, current_symbolic,
                            current_controller, current_automaton)

        # Run simulation
        trajectory = simulator.simulate(num_steps, initial_state, noise_scale)

        # Convert to list for JSON
        traj_list = [state.tolist() for state in trajectory]

        return jsonify({
            'status': 'success',
            'trajectory': traj_list,
            'auto_trajectory': simulator.auto_trajectory if hasattr(simulator, 'auto_trajectory') else []
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/visualize', methods=['POST'])
def visualize():
    """Generate visualization."""
    data = request.json
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    trajectory_data = data.get('trajectory', [])
    regions = data.get('regions', {})

    if not trajectory_data:
        return jsonify({'error': 'No trajectory to visualize'}), 400

    try:
        # Convert trajectory data back to numpy arrays
        trajectory = [np.array(s) for s in trajectory_data]
        print(f"📊 Visualizing trajectory with {len(trajectory)} points")

        # Create visualizer
        from visualization import Visualizer
        vis = Visualizer(projection_dims=(0, 1))

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        vis.ax = ax
        vis.fig = fig

        # Set title
        ax.set_title(f'Robot Trajectory ({len(trajectory)} steps)', fontsize=14, fontweight='bold')

        # Plot regions if available
        if regions:
            # Generate colors for regions
            colors = ['#ff9999', '#99ff99', '#9999ff', '#ffff99', '#ff99ff']
            for i, (region_name, bounds) in enumerate(regions.items()):
                color = colors[i % len(colors)]

                # Plot region as rectangle
                if len(bounds) >= 2:
                    x1, x2 = bounds[0]
                    y1, y2 = bounds[1]
                    rect = plt.Rectangle(
                        (x1, y1), x2 - x1, y2 - y1,
                        facecolor=color, alpha=0.3, edgecolor=color, linewidth=2,
                        label=region_name
                    )
                    ax.add_patch(rect)

                    # Add region name
                    ax.text(x1 + (x2 - x1) / 2, y1 + (y2 - y1) / 2, region_name,
                            ha='center', va='center', fontsize=10, fontweight='bold',
                            color='black', bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))

        # Plot workspace bounds if available
        if current_model:
            bounds = current_model.get_state_bounds()
            ax.set_xlim(bounds[0])
            ax.set_ylim(bounds[1])
        else:
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 10)

        # Plot trajectory
        if len(trajectory) > 0:
            # Extract x,y coordinates (project onto first two dimensions)
            traj_x = [s[0] for s in trajectory]
            traj_y = [s[1] for s in trajectory]

            # Plot trajectory line
            ax.plot(traj_x, traj_y, 'b-', linewidth=2, alpha=0.7, label='Trajectory')

            # Plot points
            ax.scatter(traj_x, traj_y, c='blue', s=20, alpha=0.5, zorder=3)

            # Mark start and end
            ax.scatter(traj_x[0], traj_y[0], c='green', s=100, marker='o',
                       label='Start', zorder=4, edgecolor='black', linewidth=2)
            ax.scatter(traj_x[-1], traj_y[-1], c='red', s=100, marker='s',
                       label='End', zorder=4, edgecolor='black', linewidth=2)

            # Add text for start/end
            ax.annotate('Start', (traj_x[0], traj_y[0]),
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
            ax.annotate('End', (traj_x[-1], traj_y[-1]),
                        xytext=(5, 5), textcoords='offset points', fontsize=9)

        # Add grid and labels
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X Position', fontsize=12)
        ax.set_ylabel('Y Position', fontsize=12)
        ax.legend(loc='upper right')
        ax.set_aspect('equal')

        # Save to buffer
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        buf.seek(0)

        # Convert to base64 for embedding
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

        print(f"✅ Visualization generated successfully")

        return jsonify({
            'status': 'success',
            'image': f'data:image/png;base64,{image_base64}'
        })

    except Exception as e:
        print(f"❌ Visualization error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
@app.route('/api/export_controller', methods=['GET'])
def export_controller():
    """Export controller as JSON."""
    if current_controller is None:
        return jsonify({'error': 'No controller to export'}), 400

    try:
        # Convert controller to serializable format
        controller_data = {
            'metadata': {
                'model': current_model.name if current_model else 'unknown',
                'n_cells': current_symbolic.n_cells if current_symbolic else 0,
                'n_inputs': current_symbolic.n_inputs if current_symbolic else 0,
                'automaton': current_automaton.to_json() if current_automaton else None
            },
            'winning_states': [
                {'auto_state': s.auto_state, 'cell_idx': int(s.cell_idx)}
                for s in current_controller.winning_states
            ],
            'control_map': {}
        }

        # Add control map
        for state, inputs in current_controller.controller.items():
            key = f"{state.auto_state},{state.cell_idx}"
            controller_data['control_map'][key] = [inp.tolist() for inp in inputs]

        return jsonify(controller_data)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': current_model is not None,
        'automaton_generated': current_automaton is not None,
        'controller_synthesized': current_controller is not None
    })


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    # Load environment variables from .env file
    from dotenv import load_dotenv
    env_path = BASE_DIR / '.env'
    if env_path.exists():
        load_dotenv(env_path)

    # Get port from environment or use default
    port = int(os.environ.get('PORT', 5000))

    # Run app
    app.run(debug=True, host='0.0.0.0', port=port)