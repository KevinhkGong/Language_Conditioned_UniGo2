"""
Stage A — Structured Intent Extraction

Converts natural language commands into structured task specifications
using GPT-4 function calling / structured outputs.

Usage:
    python -m src.language.intent_parser "press the red button on the left panel"
"""

import json
import argparse
from dataclasses import dataclass, asdict
from typing import Optional

import yaml
from openai import OpenAI


# ---- Data Structures ----
@dataclass
class Target:
    category: str                    # e.g. "button", "box", "ball"
    color: Optional[str] = None      # e.g. "red", or None
    description: Optional[str] = None

@dataclass
class SpatialConstraints:
    goal_region: Optional[str] = None   # push destination: "left", "right", "forward", etc.
    location: Optional[str] = None      # where the target is: "on the left panel", etc.

@dataclass
class TaskSpec:
    interaction_type: str            # "press" or "push"
    target: Target
    spatial_constraints: SpatialConstraints


# ---- Function Schema for GPT-4 ----
EXTRACTION_FUNCTION = {
    "name": "extract_task_spec",
    "description": (
        "Extract a structured robot task specification from a natural language command. "
        "The robot is a quadruped that can press targets or push objects using its body and legs."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "interaction_type": {
                "type": "string",
                "enum": ["press", "push"],
                "description": "The type of physical interaction: press (make contact with a target) or push (move an object)."
            },
            "target": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "Object category, e.g. 'button', 'box', 'ball', 'switch'."
                    },
                    "color": {
                        "type": ["string", "null"],
                        "description": "Color descriptor if mentioned, else null."
                    },
                    "description": {
                        "type": ["string", "null"],
                        "description": "Any additional identifying information, else null."
                    }
                },
                "required": ["category"]
            },
            "spatial_constraints": {
                "type": "object",
                "properties": {
                    "goal_region": {
                        "type": ["string", "null"],
                        "description": "Where to push the object (e.g. 'left', 'right', 'forward'). Null for press tasks."
                    },
                    "location": {
                        "type": ["string", "null"],
                        "description": "Where the target is relative to the robot (e.g. 'on the left panel'). Null if not specified."
                    }
                },
                "required": []
            }
        },
        "required": ["interaction_type", "target", "spatial_constraints"]
    }
}


class IntentParser:
    """Parses natural language commands into structured TaskSpec using GPT-4 function calling."""

    def __init__(self, config_path: str = "configs/default.yaml"):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)["language"]

        self.model = cfg["model"]
        self.temperature = cfg["temperature"]
        self.max_tokens = cfg["max_tokens"]
        self.api_timeout = cfg["api_timeout"]
        self.system_prompt = cfg["system_prompt"]

        self.client = OpenAI()  # reads OPENAI_API_KEY from env

    def parse(self, command: str) -> TaskSpec:
        """
        Parse a natural language command into a structured TaskSpec.

        Args:
            command: Natural language instruction, e.g. "press the red button on the left panel"

        Returns:
            TaskSpec with interaction type, target attributes, and spatial constraints.

        Raises:
            ValueError: If GPT-4 does not return a valid function call.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.api_timeout,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": command}
            ],
            functions=[EXTRACTION_FUNCTION],
            function_call={"name": "extract_task_spec"}
        )

        # Extract the function call arguments
        message = response.choices[0].message
        if not message.function_call:
            raise ValueError(
                f"GPT-4 did not return a function call. Response: {message.content}"
            )

        args = json.loads(message.function_call.arguments)

        # Build TaskSpec from the parsed arguments
        target_data = args.get("target", {})
        spatial_data = args.get("spatial_constraints", {})

        return TaskSpec(
            interaction_type=args["interaction_type"],
            target=Target(
                category=target_data["category"],
                color=target_data.get("color"),
                description=target_data.get("description"),
            ),
            spatial_constraints=SpatialConstraints(
                goal_region=spatial_data.get("goal_region"),
                location=spatial_data.get("location"),
            ),
        )

    def parse_to_dict(self, command: str) -> dict:
        """Parse and return as a plain dictionary (useful for serialization)."""
        return asdict(self.parse(command))


# ---- CLI ----
def main():
    parser = argparse.ArgumentParser(
        description="Parse a natural language robot command into a structured task spec."
    )
    parser.add_argument(
        "command",
        type=str,
        help='Natural language command, e.g. "press the red button on the left panel"'
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config file (default: configs/default.yaml)"
    )
    args = parser.parse_args()

    intent_parser = IntentParser(config_path=args.config)
    result = intent_parser.parse(args.command)

    print("\n--- Parsed Task Specification ---")
    print(json.dumps(asdict(result), indent=2))


# ---- Quick Standalone Test (no API call) ----

def test_datastructures():
    """Verify dataclasses work correctly without calling the API."""
    spec = TaskSpec(
        interaction_type="press",
        target=Target(category="button", color="red", description="large circular"),
        spatial_constraints=SpatialConstraints(goal_region=None, location="on the left panel"),
    )
    d = asdict(spec)
    assert d["interaction_type"] == "press"
    assert d["target"]["color"] == "red"
    assert d["spatial_constraints"]["goal_region"] is None
    print("Datastructure test passed.")

    # Test roundtrip from dict
    spec2 = TaskSpec(
        interaction_type=d["interaction_type"],
        target=Target(**d["target"]),
        spatial_constraints=SpatialConstraints(**d["spatial_constraints"]),
    )
    assert asdict(spec2) == d
    print("Roundtrip test passed.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_datastructures()
    else:
        main()