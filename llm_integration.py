import os
import requests
import json
import re
import random
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file with explicit path
env_path = Path(__file__).resolve().parent / '.env'
load_dotenv(env_path)


class LLMInterface:
    def __init__(self, config=None):
        # Print debug info (remove after fixing)
        print("🔍 Debug: Loading API key...")

        # Try multiple ways to get the API key
        self.api_key = None

        # Method 1: From config parameter
        if config and isinstance(config, dict):
            self.api_key = config.get('api_key')
            print(f"Method 1 - From config: {'Found' if self.api_key else 'Not found'}")

        # Method 2: From environment variable
        if not self.api_key:
            self.api_key = os.getenv('OPENROUTER_API_KEY')
            print(f"Method 2 - From env var: {'Found' if self.api_key else 'Not found'}")

        # Method 3: Hardcoded fallback (remove in production)
        if not self.api_key:
            self.api_key = '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'
            print("Method 3 - Using hardcoded key")

        # Check if we have a key
        if not self.api_key:
            print("⚠️ No API key found! Will use default automata only.")
            self.api_key = None
        else:
            print(f"✅ API Key loaded (starts with: {self.api_key[:15]}...)")

        self.base_url = "https://openrouter.ai/api/v1"

        # Headers required by OpenRouter
        self.headers = {
            "Authorization": f"Bearer {self.api_key}" if self.api_key else "",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:5000",
            "X-Title": "Symbolic Control Framework"
        }

        # Available free models
        self.models = {
            "default": "arcee-ai/trinity-large-preview:free",  # ✅ This works
            "step": "stepfun/step-3.5-flash:free",
            "glm": "z-ai/glm-4.5-air:free",
            "nvidia": "nvidia/nemotron-3-nano-30b-a3b:free",
            "gpt-oss": "openai/gpt-oss-120b:free",
            "qwen": "qwen/qwen3-32b:free",
            "llama": "meta-llama/llama-3.3-70b-instruct:free",
            "gemma": "google/gemma-3-27b-it:free"
        }

    def test_connection(self):
        """Test the API connection"""
        if not self.api_key:
            print("⚠️ No API key available - skipping connection test")
            return False

        try:
            print("🔄 Testing connection to OpenRouter...")

            payload = {
                "model": self.models["default"],
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 10
            }

            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=10
            )

            if response.status_code == 200:
                print("✅ OpenRouter connection successful!")
                return True
            else:
                print(f"❌ Error {response.status_code}: {response.text}")
                return False

        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False

    def generate_automaton(self, prompt, region_names, model=None):
        """
        Generate automaton from natural language prompt.
        Falls back to default automaton if generation fails.
        """
        # First, try to generate with LLM if API key exists
        if self.api_key:
            automaton = self._try_llm_generation(prompt, region_names, model)
            if automaton:
                return automaton

        # If LLM generation fails or no API key, use default
        print("⚠️ Using default automaton (LLM generation failed or no API key)")
        return self._get_default_automaton(prompt, region_names)

    def _try_llm_generation(self, prompt, region_names, model=None):
        """Attempt to generate automaton using LLM"""
        if not model:
            model = self.models["default"]

        system_prompt = """You are a formal methods expert specializing in robot control specifications.
Generate a deterministic finite automaton in JSON format that captures the desired behavior.

The automaton must have:
- states: list of state names (e.g., ["q0", "q1", "q2"])
- initial: the initial state name
- accepting: list of accepting/final state names
- transitions: dict mapping from state to dict of observation -> next_state

Observations are exactly the region names provided. Use "" for transitions that don't depend on observations.

Example format:
{
  "states": ["q0", "q1", "q2"],
  "initial": "q0",
  "accepting": ["q2"],
  "transitions": {
    "q0": {"red": "q1"},
    "q1": {"blue": "q2"},
    "q2": {}
  }
}"""

        user_prompt = f"""Region names in workspace: {region_names}

Task description: {prompt}

Generate an automaton that encodes this specification. Return ONLY the JSON object, no additional text."""

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 2000,
            "top_p": 0.95
        }

        try:
            print(f"🤖 Sending request to {model}...")
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                automaton_text = result['choices'][0]['message']['content']
                automaton_json = self._extract_json(automaton_text)

                if automaton_json:
                    print("✅ Automaton generated successfully!")
                    return automaton_json
                else:
                    print("❌ Could not extract JSON from response")
                    return None
            else:
                print(f"❌ API Error {response.status_code}: {response.text}")
                return None

        except Exception as e:
            print(f"❌ Error during LLM generation: {e}")
            return None

    def _get_default_automaton(self, prompt, region_names):
        """
        Generate a default automaton based on the prompt keywords.
        Falls back to simple patterns if prompt analysis fails.
        """
        prompt_lower = prompt.lower()

        # Pattern 1: Sequential visit (go to A then B)
        if len(region_names) >= 2 and any(word in prompt_lower for word in ["then", "after", "sequentially"]):
            return {
                "states": ["q0", "q1", "q2"],
                "initial": "q0",
                "accepting": ["q2"],
                "transitions": {
                    "q0": {region_names[0]: "q1"},
                    "q1": {region_names[1]: "q2"},
                    "q2": {}
                }
            }

        # Pattern 2: Avoidance (stay out of X)
        elif any(word in prompt_lower for word in ["avoid", "stay out", "never enter", "don't go"]):
            # Find which region to avoid
            avoid_region = None
            for region in region_names:
                if region.lower() in prompt_lower:
                    avoid_region = region
                    break

            if avoid_region:
                return {
                    "states": ["q0", "q1"],
                    "initial": "q0",
                    "accepting": ["q0"],
                    "transitions": {
                        "q0": {avoid_region: "q1"},
                        "q1": {}
                    }
                }

        # Pattern 3: Reachability (eventually reach X)
        elif any(word in prompt_lower for word in ["reach", "go to", "visit", "get to"]):
            # Find target region
            target_region = None
            for region in region_names:
                if region.lower() in prompt_lower:
                    target_region = region
                    break

            if target_region:
                return {
                    "states": ["q0", "q1"],
                    "initial": "q0",
                    "accepting": ["q1"],
                    "transitions": {
                        "q0": {target_region: "q1"},
                        "q1": {}
                    }
                }

        # Pattern 4: Stay in region (remain in X)
        elif any(word in prompt_lower for word in ["stay", "remain", "keep"]):
            stay_region = None
            for region in region_names:
                if region.lower() in prompt_lower:
                    stay_region = region
                    break

            if stay_region:
                return {
                    "states": ["q0", "q1"],
                    "initial": "q0",
                    "accepting": ["q0", "q1"],
                    "transitions": {
                        "q0": {stay_region: "q1"},
                        "q1": {stay_region: "q1"}
                    }
                }

        # Pattern 5: Patrol between regions
        elif len(region_names) >= 2 and any(word in prompt_lower for word in ["patrol", "alternate", "between"]):
            return {
                "states": ["q0", "q1", "q2"],
                "initial": "q0",
                "accepting": ["q0", "q1", "q2"],
                "transitions": {
                    "q0": {region_names[0]: "q1"},
                    "q1": {region_names[1]: "q2"},
                    "q2": {region_names[0]: "q1"}
                }
            }

        # Default: Simple automaton that just tracks being in any region
        if len(region_names) == 1:
            return {
                "states": ["q0", "q1"],
                "initial": "q0",
                "accepting": ["q1"],
                "transitions": {
                    "q0": {region_names[0]: "q1"},
                    "q1": {}
                }
            }
        elif len(region_names) >= 2:
            return {
                "states": ["q0", "q1", "q2"],
                "initial": "q0",
                "accepting": ["q2"],
                "transitions": {
                    "q0": {region_names[0]: "q1"},
                    "q1": {region_names[1]: "q2"},
                    "q2": {}
                }
            }
        else:
            # No regions
            return {
                "states": ["q0"],
                "initial": "q0",
                "accepting": ["q0"],
                "transitions": {
                    "q0": {}
                }
            }

    def _extract_json(self, text):
        """Extract JSON object from text"""
        # Try to find JSON between triple backticks
        code_block = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
        if code_block:
            text = code_block.group(1)

        # Find the first { and last }
        start = text.find('{')
        end = text.rfind('}') + 1

        if start >= 0 and end > start:
            json_str = text[start:end]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                print(f"❌ JSON parse error: {e}")
                return None
        return None

    def list_available_models(self):
        """List available free models"""
        print("\n📋 Available Free Models:")
        for name, model_id in self.models.items():
            print(f"  • {name}: {model_id}")


# ============= Main wrapper function =============
def prompt_to_automaton(prompt, region_names, llm_config=None):
    """
    Wrapper function for app.py to call

    Args:
        prompt: Natural language description
        region_names: List of region names
        llm_config: Optional LLM configuration

    Returns:
        Automaton JSON object (always returns something, never None)
    """
    try:
        # Create LLM interface
        llm = LLMInterface(llm_config)

        # Generate automaton (will use default if LLM fails)
        automaton_data = llm.generate_automaton(prompt, region_names)

        # Ensure we always return a valid automaton
        if automaton_data and _validate_automaton(automaton_data):
            print("✅ Using generated automaton")
            return automaton_data
        else:
            print("⚠️ Generated automaton invalid, using safe default")
            return _get_safe_default_automaton(region_names)

    except Exception as e:
        print(f"❌ Error in prompt_to_automaton: {e}")
        return _get_safe_default_automaton(region_names)


def _validate_automaton(automaton):
    """Validate that automaton has required fields"""
    required_fields = ["states", "initial", "accepting", "transitions"]

    if not isinstance(automaton, dict):
        return False

    for field in required_fields:
        if field not in automaton:
            print(f"❌ Missing required field: {field}")
            return False

    if automaton["initial"] not in automaton["states"]:
        print(f"❌ Initial state {automaton['initial']} not in states")
        return False

    for acc in automaton["accepting"]:
        if acc not in automaton["states"]:
            print(f"❌ Accepting state {acc} not in states")
            return False

    return True


def _get_safe_default_automaton(region_names):
    """Get the safest default automaton"""
    if not region_names:
        return {
            "states": ["q0"],
            "initial": "q0",
            "accepting": ["q0"],
            "transitions": {
                "q0": {}
            }
        }
    elif len(region_names) == 1:
        return {
            "states": ["q0", "q1"],
            "initial": "q0",
            "accepting": ["q1"],
            "transitions": {
                "q0": {region_names[0]: "q1"},
                "q1": {}
            }
        }
    else:
        return {
            "states": ["q0", "q1", "q2"],
            "initial": "q0",
            "accepting": ["q2"],
            "transitions": {
                "q0": {region_names[0]: "q1"},
                "q1": {region_names[1]: "q2"},
                "q2": {}
            }
        }


# Test script
if __name__ == "__main__":
    # Test with and without API key
    print("=" * 60)
    print("Testing LLM Interface with Default Fallbacks")
    print("=" * 60)

    # Test 1: With API key (if available)
    llm = LLMInterface()

    # Test connection
    if llm.api_key:
        llm.test_connection()

    # Test prompts
    test_cases = [
        ("Go to the red region", ["red"]),
        ("Go to red then blue", ["red", "blue"]),
        ("Avoid the red region", ["red", "blue"]),
        ("Stay in the blue region", ["red", "blue"]),
        ("Patrol between red and blue", ["red", "blue"]),
        ("Random nonsense prompt", ["red", "blue"])
    ]

    for i, (prompt, regions) in enumerate(test_cases):
        print(f"\n{'=' * 40}")
        print(f"Test {i + 1}: {prompt}")
        print(f"Regions: {regions}")
        print(f"{'=' * 40}")

        automaton = prompt_to_automaton(prompt, regions)
        print("\n📊 Resulting Automaton:")
        print(json.dumps(automaton, indent=2))