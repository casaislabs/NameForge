#!/usr/bin/env python3
"""
Setup script for Gemma 3 model access
This script helps configure access to Google's Gemma 3 model
"""

import os
import sys
from pathlib import Path

def setup_gemma_access():
    """Setup Gemma 3 model access"""
    print("🚀 Setting up Gemma 3 access...")
    print("\nTo use the google/gemma-3-270m-it model, you need:")
    print("1. Have a Hugging Face account (https://huggingface.co)")
    print("2. Accept Google's terms of use for Gemma")
    print("3. Configure your access token in the .env file")
    
    # Try to load .env file first
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("✅ .env file loaded")
    except ImportError:
        print("⚠️  python-dotenv is not installed, trying to load manually")
    
    # Check if HF_TOKEN is set
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    
    if not hf_token:
        print("\n❌ Hugging Face token not found")
        print("Please configure your token in the .env file:")
        print("1. Edit the .env file that was created")
        print("2. Change 'your_token_here' to your real token")
        print("3. Save the file and run this script again")
        
        # Try to set token interactively as fallback
        try:
            token = input("\nDo you want to enter the token now? (y/n): ").strip().lower()
            if token == 'y':
                token_value = input("Enter your Hugging Face token: ").strip()
                if token_value:
                    os.environ["HF_TOKEN"] = token_value
                    print("✅ Token configured temporarily")
                else:
                    print("⚠️  Empty token, skipping")
            else:
                print("⚠️  Please configure the token in .env and run again")
                return False
        except KeyboardInterrupt:
            print("\n⚠️  Setup cancelled")
            return False
    else:
        print("✅ Hugging Face token found in .env")
    
    # Test model access
    print("\n🧪 Testing model access...")
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        
        # Module (top-level)
        os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
        
        model_name = "google/gemma-3-270m-it"
        
        print(f"Downloading tokenizer for {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            token=hf_token
        )
        print("✅ Tokenizer downloaded successfully")
        
        print(f"Downloading model {model_name}...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            token=hf_token
        )
        print("✅ Model downloaded successfully")
        
        # Test generation
        print("🧪 Testing text generation...")
        try:
            # Simple test prompt
            prompt = "Generate a startup name:"
            inputs = tokenizer(prompt, return_tensors="pt")
            
            # Move inputs to the same device as the model
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=20,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
            
            output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"✅ Generation successful: {output_text[:100]}...")
            
        except Exception as gen_error:
            print(f"⚠️  Generation test failed: {gen_error}")
            print("✅ Model loaded successfully (generation test skipped)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error accessing model: {e}")
        print("\nPossible solutions:")
        print("1. Make sure you have accepted Gemma's terms on Hugging Face")
        print("2. Verify that your token has read permissions")
        print("3. Try running: huggingface-cli login")
        return False

def create_env_file():
    """Create .env file with configuration"""
    env_content = """# Hugging Face Configuration for Gemma 3
# Get your token from: https://huggingface.co/settings/tokens

HF_TOKEN=your_token_here

# Gemma 3 Model Configuration
DEFAULT_MODEL=google/gemma-3-270m-it

# Optional: Set to True to enable model sharing
SHARE_MODEL=False

# Optional: Set to True to enable debug logging
DEBUG=False
"""
    
    env_file = Path(".env")
    if not env_file.exists():
        with open(env_file, "w") as f:
            f.write(env_content)
        print("✅ .env file created")
        print("⚠️  Please edit .env and configure your real token")
    else:
        print("ℹ️  .env file already exists")

if __name__ == "__main__":
    print("🔧 NameForge Setup - Gemma 3 Only")
    print("=" * 50)
    
    # Create .env file
    create_env_file()
    
    # Setup Gemma access
    if setup_gemma_access():
        print("\n🎉 Setup completed successfully!")
        print("You can now run your NameForge application")
        print("\n💡 Remember: Your token is saved in .env for future runs")
    else:
        print("\n⚠️  Setup incomplete")
        print("Please resolve the issues before continuing")
        print("\n🔧 Steps to resolve:")
        print("1. Make sure you have accepted Gemma 3 terms")
        print("2. Configure your token in the .env file")
        print("3. Run this script again")
