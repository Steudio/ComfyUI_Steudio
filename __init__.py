import os
import importlib.util
from pathlib import Path
import tomllib

def get_version_from_pyproject():
    pyproject_path = Path(__file__).parent / "pyproject.toml"
    if not pyproject_path.exists():
        print("⚠️ pyproject.toml not found — version will be set to 'not found'")
        return "not found"

    try:
        with pyproject_path.open("rb") as f:
            data = tomllib.load(f)
    except Exception as e:
        print(f"❌ Failed to parse pyproject.toml: {e}")
        return "invalid"

    version = data.get("project", {}).get("version")
    if version is None:
        print("⚠️ pyproject.toml missing 'version' entry under [project] — defaulting to 'unknown'")
        return "unknown"

    return version

__version__ = get_version_from_pyproject()

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
WEB_DIRECTORY = "./js"

def load_nodes():
    current_dir = Path(__file__).parent

    for file in current_dir.rglob("*.py"):
        if file.stem == "__init__":
            continue

        try:
            spec = importlib.util.spec_from_file_location(file.stem, file)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                if hasattr(module, "NODE_CLASS_MAPPINGS"):
                    NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)
                if hasattr(module, "NODE_DISPLAY_NAME_MAPPINGS"):
                    NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)

        except Exception as e:
            print(f"❌ Error loading {file.name}: {e}")

load_nodes()

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]

print()
print(f'\033[96m[Steudio]\033[0m v{__version__} \033[92m✓\033[0m Loaded \033[93m{len(NODE_CLASS_MAPPINGS)} nodes\033[0m')
print()