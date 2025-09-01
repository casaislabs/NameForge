"""
NameForge Configuration (Simplified)
Only the settings actually used by the app are kept.
"""

import os
import logging
from pathlib import Path
from logging.handlers import RotatingFileHandler

# =============================================================================
# CORE APP CONFIG
# =============================================================================

# UI
UI_CONFIG = {
    "title": "NameForge - Intelligent Startup Name Generator",
}

# Sliders (used by app.py)
NAME_LENGTH = {"min": 3, "max": 12, "default": 6}
NAME_COUNT = {"min": 1, "max": 20, "default": 10}

# Styles (only the options and default are needed by app.py)
STYLE_OPTIONS = ["fun", "serious", "techy"]
DEFAULT_STYLE = "techy"

# Models (used by app.py dropdown)
AVAILABLE_MODELS = {
    "google/gemma-3-270m-it": {
        "name": "Gemma 3 (270M) - Instruction Tuned",
        "description": "Google's latest Gemma 3 model for creative text generation",
        "default": True,
    }
}
DEFAULT_MODEL = "google/gemma-3-270m-it"

# GPU (used by name_generator)
GPU_CONFIG = {
    "enabled": True,
    "force_cpu": False,
}

# Domain checking (used by domain_checker)
DOMAIN_CHECK = {
    "delay": 0.1,
    "timeout": 5.0,
    "max_concurrent": 5,
    "retry_attempts": 2,
    "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
}
DEFAULT_TLD = ".com"
TLD_OPTIONS = [".com", ".io", ".ai", ".co", ".net", ".org"]

# =============================================================================
# GRADIO ENVIRONMENT VARIABLES
# =============================================================================

# Production environment variables
os.environ.update({
    # Analytics and Telemetry (disabled for privacy)
    "GRADIO_ANALYTICS_ENABLED": "0",
    "GRADIO_TELEMETRY_ENABLED": "0",
    
    # Network and Security
    "GRADIO_CHECK_IP": "0",  # Disable IP checking for production
    "GRADIO_PKG_VERSION_CHECK": "0",  # Disable version checks
    "GRADIO_TUNNEL_REQUEST": "1",  # Enable tunnel requests for sharing
    
    # Logging and Debug
    "GRADIO_VERBOSE": "1",  # Enable verbose logging for production monitoring
    "GRADIO_DEBUG": "0",  # Disable debug mode in production
    
    # Performance optimizations
    "GRADIO_TEMP_DIR": os.path.join(os.getcwd(), "temp"),  # Custom temp directory
    "GRADIO_EXAMPLES_CACHE": "1",  # Enable examples caching
    
    # Security headers
    "GRADIO_ALLOW_FLAGGING": "never",  # Disable flagging in production
})

# =============================================================================
# SERVER CONFIG - Optimized for easy sharing
# =============================================================================

SERVER_CONFIG = {
    "host": "127.0.0.1",  # Local host
    "port": 7860,
    "share": True,  # Enable Gradio sharing for easy access
    "show_api": False,  # Keep API simple
    "show_error": True,  # Show errors for debugging
    "quiet": False,  # Show startup messages
    "inbrowser": True,  # Auto-open browser
}

def find_available_port(start_port=7860, max_attempts=10):
    import socket
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", port))
                return port
        except OSError:
            continue
    return None

def get_gradio_launch_kwargs(override_config=None):
    """Get Gradio launch configuration for easy sharing."""
    config = SERVER_CONFIG.copy()
    if override_config:
        config.update(override_config)

    # Check port availability
    if not find_available_port(config["port"], 1):
        available_port = find_available_port(7860, 10)
        if available_port:
            print(f"⚠️  Port {config['port']} is busy, using port {available_port}")
            config["port"] = available_port
        else:
            print("❌ No available ports found in range 7860-7869")

    # Status messages
    if config.get("share", False):
        print("🌐 Sharing enabled: Your app will be accessible via a public Gradio link")
        print("🔗 Perfect for sharing with others or running on Google Colab!")

    return {
        "server_name": config["host"],
        "server_port": config["port"],
        "share": config["share"],
        "show_error": config["show_error"],
        "quiet": config["quiet"],
        "inbrowser": config["inbrowser"],
        "show_api": config["show_api"],
        "prevent_thread_lock": False,
    }



# =============================================================================
# LOGGING (centralized and consistent)
# =============================================================================

DEBUG_CONFIG = {
    "enable_api_logging": True,
    "enable_gradio_logging": True,
    "enable_model_logging": True,
    "enable_parameter_logging": True,
    "log_all_parameters": True,
    "log_stack_traces": True,
    "max_log_size": "10MB",
    "backup_count": 5,
}

def _parse_size(size):
    if isinstance(size, int):
        return size
    if isinstance(size, str):
        s = size.strip().upper()
        units = {"KB": 1024, "MB": 1024**2, "GB": 1024**3}
        for unit, mult in units.items():
            if s.endswith(unit):
                try:
                    return int(float(s.replace(unit, "")) * mult)
                except ValueError:
                    break
        try:
            return int(s)
        except ValueError:
            return 10 * 1024 * 1024
    return 10 * 1024 * 1024

def setup_logging():
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    legacy_log = Path("nameforge.log")
    if legacy_log.exists():
        try:
            legacy_target = logs_dir / "nameforge_legacy.log"
            if not legacy_target.exists():
                legacy_log.rename(legacy_target)
        except Exception:
            pass

    log_format = "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(funcName)s() - %(message)s"
    max_bytes = _parse_size(DEBUG_CONFIG.get("max_log_size", "10MB"))
    backups = int(DEBUG_CONFIG.get("backup_count", 5))

    detailed_handler = RotatingFileHandler(
        logs_dir / "nameforge_detailed.log",
        maxBytes=max_bytes,
        backupCount=backups,
        encoding="utf-8",
    )
    detailed_handler.setLevel(logging.DEBUG)
    detailed_handler.setFormatter(logging.Formatter(log_format))

    error_handler = RotatingFileHandler(
        logs_dir / "nameforge_errors.log",
        maxBytes=max_bytes,
        backupCount=backups,
        encoding="utf-8",
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(logging.Formatter(log_format))

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

    logger = logging.getLogger("NameForge")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.handlers.clear()
    logger.addHandler(detailed_handler)
    logger.addHandler(error_handler)
    logger.addHandler(console_handler)

    return logger

# Global logger
logger = setup_logging()

API_CALLS = {
    "huggingface_calls": [],
    "gradio_endpoints": [],
    "model_generations": [],
    "errors": [],
}

def log_api_call(api_type: str, endpoint: str, parameters: dict, response: any = None, error: Exception = None):
    if not DEBUG_CONFIG["enable_api_logging"]:
        return
    import time, traceback
    entry = {
        "timestamp": time.time(),
        "api_type": api_type,
        "endpoint": endpoint,
        "parameters": parameters,
        "response": response,
        "error": str(error) if error else None,
        "stack_trace": traceback.format_exc() if error and DEBUG_CONFIG["log_stack_traces"] else None,
    }
    key = f"{api_type}_calls"
    if key not in API_CALLS:
        API_CALLS[key] = []
    API_CALLS[key].append(entry)
    logger.debug(f"API Call: {api_type} -> {endpoint}")
    logger.debug(f"Parameters: {parameters}")
    if error:
        logger.error(f"API Error: {error}")
        if entry["stack_trace"]:
            logger.error(f"Stack trace: {entry['stack_trace']}")
    elif response is not None:
        logger.debug(f"Response: {response}")

def log_gradio_endpoint(endpoint: str, inputs: list, outputs: list, error: Exception = None):
    if not DEBUG_CONFIG["enable_gradio_logging"]:
        return
    import time, traceback
    entry = {
        "timestamp": time.time(),
        "endpoint": endpoint,
        "inputs": inputs,
        "outputs": outputs,
        "error": str(error) if error else None,
        "stack_trace": traceback.format_exc() if error and DEBUG_CONFIG["log_stack_traces"] else None,
    }
    API_CALLS["gradio_endpoints"].append(entry)
    logger.debug(f"Gradio Endpoint: {endpoint}")
    logger.debug(f"Inputs: {inputs}")
    if error:
        logger.error(f"Gradio Error: {error}")
        if entry["stack_trace"]:
            logger.error(f"Stack trace: {entry['stack_trace']}")
    else:
        logger.debug(f"Outputs: {outputs}")

def log_model_generation(model_name: str, parameters: dict, result: any, error: Exception = None, timing: float = None):
    if not DEBUG_CONFIG["enable_model_logging"]:
        return
    import time, traceback
    entry = {
        "timestamp": time.time(),
        "model_name": model_name,
        "parameters": parameters,
        "result": result,
        "error": str(error) if error else None,
        "timing": timing,
        "stack_trace": traceback.format_exc() if error and DEBUG_CONFIG["log_stack_traces"] else None,
    }
    API_CALLS["model_generations"].append(entry)
    logger.debug(f"Model Generation: {model_name}")
    logger.debug(f"Parameters: {parameters}")
    if timing:
        logger.debug(f"Timing: {timing:.2f}s")
    if error:
        logger.error(f"Model Error: {error}")
        if entry["stack_trace"]:
            logger.error(f"Stack trace: {entry['stack_trace']}")
    else:
        logger.debug(f"Result: {result}")

def log_parameters(function_name: str, parameters: dict, error: Exception = None):
    if not DEBUG_CONFIG["enable_parameter_logging"]:
        return
    if DEBUG_CONFIG["log_all_parameters"]:
        logger.debug(f"Function: {function_name}")
        logger.debug(f"Parameters: {parameters}")
    if error:
        logger.error(f"Parameter Error in {function_name}: {error}")