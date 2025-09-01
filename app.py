"""
NameForge - Main Application
Interactive web application for generating startup names.
"""

import os
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import gradio as gr
import pandas as pd
from name_generator import NameGenerator
from domain_checker import check_domains_across_tlds
from config import (
    UI_CONFIG, STYLE_OPTIONS, AVAILABLE_MODELS,
    NAME_LENGTH, NAME_COUNT, DEFAULT_STYLE, DEFAULT_MODEL,
    get_gradio_launch_kwargs, log_gradio_endpoint,
    log_parameters, logger, DEFAULT_TLD, TLD_OPTIONS
)
import tempfile

# Custom CSS for professional styling
CUSTOM_CSS = """
/* Global Styles */
.gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    min-height: 100vh;
}

/* Main content area */
.main-container {
    background: rgba(255, 255, 255, 0.95) !important;
    backdrop-filter: blur(10px) !important;
    border-radius: 20px !important;
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1) !important;
    margin: 20px !important;
    padding: 30px !important;
}

/* Header styling */
.header-section {
    text-align: center;
    margin-bottom: 40px;
    padding: 30px 0;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 15px;
    color: white;
    box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
}

.header-section h1 {
    font-size: 3.5rem !important;
    font-weight: 700 !important;
    margin-bottom: 10px !important;
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
}

.header-section h3 {
    font-size: 1.4rem !important;
    font-weight: 400 !important;
    margin-bottom: 5px !important;
    opacity: 0.9;
}

.header-section p {
    font-size: 1rem !important;
    opacity: 0.8;
    margin: 0 !important;
}

/* Section headers */
.section-header {
    background: linear-gradient(90deg, #667eea, #764ba2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 1.3rem !important;
    font-weight: 600 !important;
    margin-bottom: 20px !important;
    padding-bottom: 10px;
    border-bottom: 2px solid #e9ecef;
}

/* Input styling */
.input-section {
    background: #f8f9fa;
    border-radius: 15px;
    padding: 25px;
    margin-bottom: 20px;
    border: 1px solid #e9ecef;
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.05);
}

/* Button styling */
.primary-button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 15px 30px !important;
    font-weight: 600 !important;
    font-size: 1.1rem !important;
    box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3) !important;
    transition: all 0.3s ease !important;
}

.primary-button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 12px 35px rgba(102, 126, 234, 0.4) !important;
}

.secondary-button {
    background: #6c757d !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 24px !important;
    font-weight: 500 !important;
    transition: all 0.3s ease !important;
}

.secondary-button:hover {
    background: #5a6268 !important;
    transform: translateY(-1px) !important;
}

/* Output styling */
.output-section {
    background: #ffffff;
    border-radius: 15px;
    padding: 25px;
    margin-bottom: 20px;
    border: 1px solid #e9ecef;
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.05);
}

/* Results area */
.results-area {
    background: #f8f9fa;
    border: 2px solid #e9ecef;
    border-radius: 12px;
    padding: 20px;
    font-family: 'Monaco', 'Menlo', monospace;
    line-height: 1.6;
}

/* Examples section */
.examples-section {
    background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
    border-radius: 15px;
    padding: 25px;
    margin-top: 30px;
    border: 2px solid #dee2e6;
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.08);
}

.examples-section h4 {
    color: #495057 !important;
    font-weight: 600 !important;
    margin-bottom: 15px !important;
    font-size: 1.2rem !important;
}

.examples-section li {
    margin-bottom: 12px;
    padding: 10px 0;
    color: #212529 !important;
    line-height: 1.5;
}

.examples-section strong {
    color: #343a40 !important;
    font-weight: 600;
}

.examples-section span {
    color: #495057 !important;
    font-weight: 400;
}

/* Loading states */
.loading-overlay {
    position: relative;
}

.loading-overlay::after {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(255, 255, 255, 0.8);
    border-radius: 12px;
    z-index: 10;
}

/* Responsive design */
@media (max-width: 768px) {
    .header-section h1 {
        font-size: 2.5rem !important;
    }
    
    .main-container {
        margin: 10px !important;
        padding: 20px !important;
    }
    
    .input-section, .output-section {
        padding: 20px;
    }
}

/* Animation for smooth transitions */
.fade-in {
    animation: fadeIn 0.5s ease-in;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}
"""


def generate_startup_names(description, style, name_length, name_count, check_domains, tlds, model_name):
    """Generate startup names based on user input with robust error handling."""
    try:
        # Log function call with all parameters
        logger.info(f"generate_startup_names called with parameters:")
        logger.info(f"  - description: {description}")
        logger.info(f"  - style: {style}")
        logger.info(f"  - name_length: {name_length} (type: {type(name_length)})")
        logger.info(f"  - name_count: {name_count} (type: {type(name_count)})")
        logger.info(f"  - check_domains: {check_domains} (type: {type(check_domains)})")
        logger.info(f"  - tlds: {tlds}")
        logger.info(f"  - model_name: {model_name}")
        
        # Log parameters for debugging
        log_parameters("generate_startup_names", {
            "description": description,
            "style": style,
            "name_length": name_length,
            "name_count": name_count,
            "check_domains": check_domains,
            "tlds": tlds,
            "model_name": model_name
        })
        
        # Input validation
        if not description or not isinstance(description, str) or not description.strip():
            error_msg = "Please enter a valid project description."
            logger.warning(f"Invalid description: {description}")
            return error_msg, "", ""
        
        # Validate parameters
        if not isinstance(name_length, int) or not isinstance(name_count, int):
            error_msg = "Invalid parameters. Please check your inputs."
            logger.error(f"Invalid parameter types: name_length={type(name_length)}, name_count={type(name_count)}")
            return error_msg, "", ""
        
        if name_length < 3 or name_length > 12:
            error_msg = f"Name length must be between 3 and 12 characters. Got: {name_length}"
            logger.warning(f"Invalid name length: {name_length}")
            return error_msg, "", ""
        
        if name_count < 1 or name_count > 20:
            error_msg = f"Name count must be between 1 and 20. Got: {name_count}"
            logger.warning(f"Invalid name count: {name_count}")
            return error_msg, "", ""
        
        logger.info(f"Input validation passed, calling generate_names...")
        
        # Generate names using the NameGenerator class
        generator = NameGenerator(model_name)
        names = generator.generate_names(description, style, name_length, name_count)
        
        if not names:
            error_msg = "No names could be generated. Please try different parameters."
            logger.warning("No names generated")
            return error_msg, "", ""
        
        logger.info(f"Generated {len(names)} names successfully")
        
        # Remove duplicates and format
        unique_names = list(dict.fromkeys(names))
        names_text = "\n".join([f"{i+1}. {name}" for i, name in enumerate(unique_names)])
        
        # Prepare CSV file and return its path
        csv_file_path = ""
        try:
            df = pd.DataFrame({
                "Name": unique_names,
                "Length": [len(name) for name in unique_names],
                "Style": [style] * len(unique_names),
                "Generated_At": [pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")] * len(unique_names)
            })
            
            # Use a more reliable temporary file creation method
            temp_dir = tempfile.gettempdir()
            temp_file = tempfile.NamedTemporaryFile(
                delete=False, 
                suffix=".csv", 
                mode="w", 
                encoding="utf-8", 
                newline="",
                dir=temp_dir
            )
            df.to_csv(temp_file.name, index=False)
            temp_file.close()
            csv_file_path = temp_file.name
            logger.debug(f"CSV file created at {csv_file_path}")
        except Exception as e:
            logger.error(f"Error creating CSV file: {e}")
            csv_file_path = ""
        
        # Check domain availability if requested
        domain_results = ""
        if check_domains and unique_names:
            try:
                logger.info(f"Starting multi-TLD domain availability check for {len(unique_names)} names")
                
                # Normalize tlds to a list
                tld_list = tlds
                if isinstance(tld_list, str):
                    tld_list = [tld_list]
                if not tld_list:
                    tld_list = [DEFAULT_TLD]
                
                results_by_tld = check_domains_across_tlds(unique_names, tlds=tld_list)
                
                # Build grouped output
                domain_results_lines = ["Domain Availability Results (grouped by TLD):", ""]
                for tld, results in results_by_tld.items():
                    available = [r["domain"] for r in results if r["available"] is True]
                    unavailable = [r["domain"] for r in results if r["available"] is False]
                    errors = [r["domain"] for r in results if r["available"] is None]
                    
                    domain_results_lines.append(f"{tld}:")
                    domain_results_lines.append(f"  Available ({len(available)}):")
                    domain_results_lines += [f"    • {d}" for d in available] if available else ["    None"]
                    domain_results_lines.append(f"  Unavailable ({len(unavailable)}):")
                    domain_results_lines += [f"    • {d}" for d in unavailable] if unavailable else ["    None"]
                    if errors:
                        domain_results_lines.append(f"  Errors ({len(errors)}):")
                        domain_results_lines += [f"    • {d}" for d in errors]
                    domain_results_lines.append("")
                
                domain_results = "\n".join(domain_results_lines).rstrip()
                logger.info("Domain check (multi-TLD) completed")
            except Exception as e:
                logger.error(f"Error checking domains: {e}")
                domain_results = f"Error checking domains: {str(e)}"
        
        # Log successful completion
        logger.info(f"generate_startup_names completed successfully")
        log_gradio_endpoint(
            endpoint="generate_startup_names",
            inputs=[description, style, name_length, name_count, check_domains, tlds, model_name],
            outputs=[names_text, csv_file_path, domain_results]
        )
        return names_text, csv_file_path, domain_results
    except Exception as e:
        error_msg = f"An unexpected error occurred: {str(e)}"
        logger.error(f"Unexpected error in generate_startup_names: {e}")
        log_gradio_endpoint(
            endpoint="generate_startup_names",
            inputs=[description, style, name_length, name_count, check_domains, tlds, model_name],
            outputs=[error_msg, "", ""],
            error=e
        )
        return error_msg, "", ""


def filter_results(names_text, min_length, max_length, original_names=None):
    """Filter generated names by length with robust error handling."""
    try:
        # Log function call
        logger.info(f"filter_results called with: names_text length={len(names_text) if names_text else 0}, min_length={min_length}, max_length={max_length}")
        
        # Log parameters for debugging
        log_parameters("filter_results", {
            "names_text_length": len(names_text) if names_text else 0,
            "min_length": min_length,
            "max_length": max_length
        })
        
        # Base: usar original_names si existe y es válido
        base_text = original_names if (isinstance(original_names, str) and original_names.strip()) else names_text

        # Input validation
        if not base_text or not isinstance(base_text, str):
            error_msg = "No names to filter."
            logger.warning("No names text provided for filtering")
            return error_msg
        
        if not isinstance(min_length, int) or not isinstance(max_length, int):
            error_msg = "Invalid filter parameters."
            logger.error(f"Invalid filter parameter types: min_length={type(min_length)}, max_length={type(max_length)}")
            return error_msg
        
        if min_length > max_length:
            logger.info(f"Swapping min_length ({min_length}) and max_length ({max_length})")
            min_length, max_length = max_length, min_length
        
        # Check for error messages
        if any(error in base_text for error in ["❌", "Error", "Please enter"]):
            logger.warning("Detected error message in names_text, returning as-is")
            return base_text
        
        # Parse and filter names
        lines = base_text.strip().split('\n')
        filtered_names = []
        
        for line in lines:
            if '. ' in line:
                try:
                    name = line.split('. ', 1)[1]
                    if min_length <= len(name) <= max_length:
                        filtered_names.append(line)
                except IndexError:
                    continue
        
        if filtered_names:
            result_msg = f"Filtered Results ({len(filtered_names)} names between {min_length}-{max_length} chars):\n\n" + "\n".join(filtered_names)
            logger.info(f"Filtering completed: {len(filtered_names)} names found")
            return result_msg
        else:
            result_msg = f"No names found between {min_length} and {max_length} characters."
            logger.info(f"Filtering completed: no names found in range {min_length}-{max_length}")
            return result_msg
            
    except Exception as e:
        error_msg = f"Error filtering names: {str(e)}"
        logger.error(f"Error filtering names: {e}")
        
        # Log the error
        log_gradio_endpoint(
            endpoint="filter_results",
            inputs=[names_text, min_length, max_length],
            outputs=[error_msg],
            error=e
        )
        
        return error_msg

# Move these small handlers ABOVE the UI construction so they're defined before use.
def on_generate_start():
    # Notificación de inicio + deshabilitar botón
    gr.Info("🚀 Generating names...")
    return gr.update(interactive=False, value="⏳ Generating...")

def on_generate_end(names_text: str):
    # Habilitar botón + notificación de resultado
    try:
        if (
            not names_text
            or names_text.startswith("Please enter")
            or names_text.startswith("Invalid")
            or "No names could be generated" in names_text
            or names_text.startswith("An unexpected error occurred")
        ):
            gr.Warning("⚠️ The generation did not produce valid results. Check your prompt and parameters.")
        else:
            gr.Info("✅ Generation successfully completed.")
    finally:
        return gr.update(interactive=True, value="🚀 Generate Names")

def store_original_names(names_text: str):
    # Save the complete text of generated names for subsequent filtering
    return names_text

# Create the Gradio interface
def export_csv_from_text(names_text: str):
    """
    Build a CSV file from the current names text (what the user sees), and return its file path.
    """
    try:
        # Early validation + notification
        if not names_text or not isinstance(names_text, str) or not names_text.strip():
            gr.Warning("📋 There are no names to export. Generate names first.")
            logger.warning("export_csv_from_text called with empty or invalid names_text")
            return None

        # Evitar exportar cuando el área contiene mensajes de error/placeholder
        error_markers = [
            "Please enter a valid project description.",
            "Invalid parameters.",
            "Name length must be between",
            "Name count must be between",
            "No names could be generated",
            "An unexpected error occurred",
            "No names to filter.",
            "Error",
            "❌"
        ]
        if any(marker in names_text for marker in error_markers):
            gr.Warning("⚠️ The current content is not exportable. Generate valid names first.")
            logger.info("export_csv_from_text found error/placeholder text, skipping export")
            return None

        # Extract names from numbered lines "1. Name"
        lines = [l for l in names_text.strip().split("\n") if ". " in l]
        names = []
        for line in lines:
            try:
                name = line.split(". ", 1)[1].strip()
                if name:
                    names.append(name)
            except Exception:
                continue

        if not names:
            gr.Warning("📋 No valid names to export.")
            logger.info("export_csv_from_text found no names to export")
            return None

        df = pd.DataFrame({
            "Name": names,
            "Length": [len(n) for n in names],
            "Exported_At": [pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")] * len(names)
        })

        temp_dir = tempfile.gettempdir()
        temp_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".csv",
            mode="w",
            encoding="utf-8",
            newline="",
            dir=temp_dir
        )
        df.to_csv(temp_file.name, index=False)
        temp_file.close()
        logger.debug(f"Export CSV created at {temp_file.name}")

        # Success notification
        gr.Info("📊 CSV generated successfully. Download will start automatically.")
        return temp_file.name
    except Exception as e:
        logger.error(f"Error exporting CSV: {e}")
        gr.Error(f"❌ Error exporting CSV: {str(e)}")
        return None


# Create custom theme
custom_theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="purple",
    neutral_hue="slate",
    font=["Inter", "ui-sans-serif", "system-ui", "sans-serif"]
).set(
    body_background_fill="*primary_50",
    body_background_fill_dark="*primary_900",
    button_primary_background_fill="*primary_500",
    button_primary_background_fill_hover="*primary_600",
    button_primary_text_color="white",
    block_background_fill="white",
    block_border_width="1px",
    block_border_color="*primary_200",
    block_radius="12px",
    input_background_fill="*neutral_50",
    input_border_color="*primary_200",
    input_border_width="2px"
)

with gr.Blocks(
    title=UI_CONFIG["title"], 
    theme=custom_theme,
    css=CUSTOM_CSS
) as iface:
    
    # Header Section
    gr.HTML("""
    <div class="header-section">
        <h1>🚀 NameForge</h1>
        <h3>Intelligent Startup Name Generator</h3>
        <p>Powered by Advanced AI Language Models</p>
    </div>
    """)
    
    with gr.Row(equal_height=True):
        # Left Column - Input Section
        with gr.Column(scale=1, elem_classes=["input-section"]):
            gr.HTML('<h3 class="section-header">📝 Project Configuration</h3>')
            
            description_input = gr.Textbox(
                label="💡 Describe your project/startup",
                placeholder="e.g., A mobile app for food delivery that connects local restaurants with customers",
                lines=4,
                elem_classes=["fade-in"]
            )
            
            with gr.Row():
                style_dropdown = gr.Dropdown(
                    choices=STYLE_OPTIONS,
                    value=DEFAULT_STYLE,
                    label="🎨 Style Preference",
                    elem_classes=["fade-in"]
                )
                
                model_dropdown = gr.Dropdown(
                    choices=list(AVAILABLE_MODELS.keys()),
                    value=DEFAULT_MODEL,
                    label="🤖 AI Model",
                    elem_classes=["fade-in"]
                )
            
            with gr.Row():
                name_length_slider = gr.Slider(
                    minimum=NAME_LENGTH["min"], maximum=NAME_LENGTH["max"], 
                    value=NAME_LENGTH["default"], step=1,
                    label="📏 Name Length (characters)",
                    elem_classes=["fade-in"]
                )
                
                name_count_slider = gr.Slider(
                    minimum=NAME_COUNT["min"], maximum=NAME_COUNT["max"], 
                    value=NAME_COUNT["default"], step=1,
                    label="🔢 Number of Names",
                    elem_classes=["fade-in"]
                )
            
            gr.HTML('<h4 class="section-header">🌐 Domain Options</h4>')
            with gr.Row():
                domain_checkbox = gr.Checkbox(
                    label="✅ Check Domain Availability",
                    value=False,
                    elem_classes=["fade-in"]
                )
                tld_dropdown = gr.Dropdown(
                    choices=TLD_OPTIONS,
                    value=[DEFAULT_TLD],
                    multiselect=True,
                    label="🔗 Top-Level Domains (TLDs)",
                    elem_classes=["fade-in"]
                )
            
            with gr.Row():
                generate_btn = gr.Button(
                    "🚀 Generate Names", 
                    variant="primary", 
                    size="lg",
                    elem_classes=["primary-button"]
                )
                clear_btn = gr.Button(
                    "🗑️ Clear All", 
                    variant="secondary",
                    elem_classes=["secondary-button"]
                )
        
        # Right Column - Results Section
        with gr.Column(scale=1, elem_classes=["output-section"]):
            gr.HTML('<h3 class="section-header">🎯 Generated Names</h3>')
            
            names_output = gr.Textbox(
                label="📋 Generated Names",
                lines=12,
                interactive=False,
                elem_classes=["results-area", "fade-in"]
            )
            
            with gr.Row():
                export_btn = gr.DownloadButton(
                    "📊 Export CSV", 
                    variant="secondary",
                    elem_classes=["secondary-button"]
                )
                filter_btn = gr.Button(
                    "🔍 Filter Results", 
                    variant="secondary",
                    elem_classes=["secondary-button"]
                )
            
            gr.HTML('<h4 class="section-header">🔍 Filter Options</h4>')
            with gr.Row():
                min_length_filter = gr.Slider(
                    minimum=NAME_LENGTH["min"], maximum=NAME_LENGTH["max"], 
                    value=NAME_LENGTH["min"], step=1,
                    label="📏 Min Length",
                    elem_classes=["fade-in"]
                )
                max_length_filter = gr.Slider(
                    minimum=NAME_LENGTH["min"], maximum=NAME_LENGTH["max"], 
                    value=NAME_LENGTH["max"], step=1,
                    label="📏 Max Length",
                    elem_classes=["fade-in"]
                )

            # Hold original unfiltered names for robust filtering
            original_names_state = gr.State("")
            
            gr.HTML('<h3 class="section-header">🌐 Domain Availability</h3>')
            domain_output = gr.Textbox(
                label="🔍 Domain Check Results",
                lines=8,
                interactive=False,
                elem_classes=["results-area", "fade-in"]
            )
            
    # Examples Section
    gr.HTML("""
    <div class="examples-section">
        <h4>💡 Example Descriptions</h4>
        <ul>
            <li><strong>🍕 Food Delivery:</strong> <span>A mobile app for food delivery that connects local restaurants with customers</span></li>
            <li><strong>🎓 Educational Games:</strong> <span>A children's educational game platform for learning math and science</span></li>
            <li><strong>💼 Business Consulting:</strong> <span>A professional consulting firm specializing in business strategy and growth</span></li>
            <li><strong>🏥 Healthcare Tech:</strong> <span>A telemedicine platform connecting patients with healthcare professionals</span></li>
            <li><strong>🌱 Sustainability:</strong> <span>An eco-friendly marketplace for sustainable and zero-waste products</span></li>
        </ul>
    </div>
    """)
    
    # Event handlers
    generate_btn.click(
        fn=on_generate_start,
        inputs=None,
        outputs=generate_btn
    ).then(
        fn=generate_startup_names,
        inputs=[description_input, style_dropdown, name_length_slider, name_count_slider, domain_checkbox, tld_dropdown, model_dropdown],
        outputs=[names_output, original_names_state, domain_output]
    ).then(
        fn=store_original_names,
        inputs=[names_output],
        outputs=original_names_state
    ).then(
        fn=on_generate_end,
        inputs=[names_output],
        outputs=generate_btn
    )
    
    filter_btn.click(
        fn=filter_results,
        inputs=[names_output, min_length_filter, max_length_filter, original_names_state],
        outputs=names_output
    )
    
    export_btn.click(
        fn=export_csv_from_text,
        inputs=[names_output],
        outputs=[export_btn]
    )
    
    clear_btn.click(
        fn=lambda: ("", "", "", ""),
        outputs=[description_input, names_output, domain_output, original_names_state]
    )


if __name__ == "__main__":
    try:
        logger.info("Starting NameForge...")
        print("🚀 Starting NameForge...")
        
        # Use flexible configuration with automatic port management
        launch_kwargs = get_gradio_launch_kwargs()
        logger.info(f"Launch configuration: {launch_kwargs}")
        
        print(f"🌐 Launching web interface on port {launch_kwargs['server_port']}...")
        logger.info(f"Launching web interface on port {launch_kwargs['server_port']}...")
        
        # New: show full local URL and share status
        url = f"http://{launch_kwargs['server_name']}:{launch_kwargs['server_port']}/"
        print(f"🔗 Open your browser at: {url}")
        if launch_kwargs.get("share"):
            print("🌍 Public sharing is enabled (Gradio will print the public link).")
        
        # Log the interface launch
        log_gradio_endpoint(
            endpoint="interface_launch",
            inputs=[launch_kwargs],
            outputs=["launching"]
        )
        
        iface.launch(**launch_kwargs)
        
    except Exception as e:
        error_msg = f"❌ Error starting NameForge: {e}"
        logger.error(f"Error starting NameForge: {e}")
        print(error_msg)
        
        # Log the error
        log_gradio_endpoint(
            endpoint="interface_launch",
            inputs=[launch_kwargs if 'launch_kwargs' in locals() else "unknown"],
            outputs=["failed"],
            error=e
        )
        
        print("🔄 Trying fallback configuration...")
        logger.info("Trying fallback configuration...")
        
        try:
            # Fallback with minimal settings
            fallback_config = {
                "server_name": "127.0.0.1",
                "server_port": 7861,  # Try different port
                "show_error": False,
                "quiet": True,
                "share": False,
                "show_api": False
            }
            
            logger.info(f"Fallback configuration: {fallback_config}")
            
            iface.launch(**fallback_config)
            
        except Exception as e2:
            error_msg2 = f"❌ Fallback also failed: {e2}"
            logger.error(f"Fallback also failed: {e2}")
            print(error_msg2)
            print("⚠️ Please check your configuration and try again.")




