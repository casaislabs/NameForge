# 🚀 NameForge

**Intelligent Startup Name Generator**

NameForge is an interactive web application that uses advanced AI models to generate creative and unique names for startups. Powered by Google's Gemma 3 with a modern interface built with Gradio.

![NameForge Banner](https://img.shields.io/badge/NameForge-AI%20Powered-blue?style=for-the-badge&logo=rocket)

## ✨ Key Features

### 🤖 **AI-Powered Name Generation**
- **Gemma 3 Model**: Uses Google's `google/gemma-3-270m-it` model to generate creative names
- **Multiple Styles**: Generates names with different personalities:
  - 🎉 **Fun**: Playful and memorable names
  - 💼 **Serious**: Professional and corporate names
  - 🔧 **Techy**: Technical and innovative names

### 🌐 **Domain Verification**
- **Multiple TLDs**: Checks availability for `.com`, `.io`, `.ai`, `.co`, `.net`, `.org`
- **Real-time Verification**: Automatically checks domain availability
- **Detailed Results**: Clearly shows the status of each domain

### 🎨 **Modern Interface**
- **Responsive Design**: Optimized for desktop and mobile
- **Professional UI**: Elegant interface with gradients and animations
- **Intuitive Experience**: Easy to use for any user

### ⚡ **Optimized Performance**
- **GPU Support**: CUDA acceleration when available
- **CPU Fallback**: Works perfectly without GPU
- **Advanced Logging**: Detailed logging system for debugging

## 📋 System Requirements

### Minimum Requirements
- **Python**: 3.8 or higher
- **RAM**: 4GB minimum (8GB recommended)
- **Disk Space**: 2GB free
- **Internet**: Stable connection to download models

### Optional Requirements
- **GPU**: NVIDIA GPU with CUDA for better performance
- **VRAM**: 2GB+ for large models

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/casaislabs/NameForge.git
cd NameForge
```

### 2. Create Virtual Environment
```bash
# Windows
python -m venv nameforge_env
nameforge_env\Scripts\activate

# Linux/Mac
python3 -m venv nameforge_env
source nameforge_env/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Gemma 3 Access

#### Option A: Automatic Configuration (Recommended)
```bash
python setup_gemma.py
```

This script will guide you through the configuration process:
1. It will ask for your Hugging Face token
2. Automatically create the `.env` file
3. Verify that everything is configured correctly

#### Option B: Manual Configuration

1. **Get Hugging Face Token**:
   - Go to [huggingface.co](https://huggingface.co) and create an account
   - Go to Settings → Access Tokens
   - Create a new token with read permissions

2. **Accept Gemma Terms**:
   - Visit [google/gemma-3-270m-it](https://huggingface.co/google/gemma-3-270m-it)
   - Accept Google's terms of use

3. **Create .env file**:
   ```bash
   # Create .env file in project root
   echo "HF_TOKEN=your_token_here" > .env
   ```

## 🎯 Usage

### Local Development
```bash
python app.py
```

The application will start at `http://localhost:7860`

### 🌐 Share with Others

NameForge comes configured with **sharing enabled** by default, which means:

- ✅ **Automatic public link**: Gradio generates a public link you can share
- ✅ **Perfect for Google Colab**: Works without additional configuration
- ✅ **Easy to share**: Send the link to anyone to use your generator
- ✅ **No configuration**: No need to configure ports or domains

#### **Google Colab Usage**
```python
# In a Colab cell
!git clone https://github.com/casaislabs/NameForge.git
%cd NameForge
!pip install -r requirements.txt

# Configure token (only once)
import os
os.environ['HF_TOKEN'] = 'your_hugging_face_token'

# Run the application
!python app.py
```

When you run the application, you'll see a message like:
```
🌐 Sharing enabled: Your app will be accessible via a public Gradio link
🔗 Perfect for sharing with others or running on Google Colab!
Running on public URL: https://xxxxx.gradio.live
```

### Web Interface

1. **Business Description**: 
   - Describe your startup in 1-2 sentences
   - Example: "A mobile app for food delivery that connects local restaurants with customers"

2. **Select Style**:
   - **Fun**: For creative and playful startups
   - **Serious**: For professional companies
   - **Techy**: For technology startups

3. **Configure Parameters**:
   - **Quantity**: 1-20 names
   - **Length**: 3-12 characters
   - **TLD**: Select the domain to verify

4. **Generate Names**:
   - Click "🚀 Generate Names"
   - Wait for the AI to generate names
   - Review domain availability

### Usage Examples

#### Food Delivery Startup
```
Description: "A delivery platform that connects local restaurants with hungry customers"
Style: Fun
Results: FoodieRush, TastyDash, MunchExpress
```

#### Educational Platform
```
Description: "An educational gaming platform for children learning math and science"
Style: Techy
Results: EduCore, LearnLab, BrainBoost
```

#### Business Consulting
```
Description: "A consulting firm specialized in business strategy and growth"
Style: Serious
Results: StrategyPro, GrowthAxis, BusinessEdge
```

## 📁 Project Structure

```
NameForge/
├── 📄 app.py                 # Main Gradio application
├── 🤖 name_generator.py      # AI name generator
├── 🌐 domain_checker.py      # Domain checker
├── ⚙️ config.py             # Application configuration
├── 🔧 setup_gemma.py        # Gemma setup script
├── 📋 requirements.txt      # Python dependencies
├── 🔐 .env                  # Environment variables (create)
├── 📊 logs/                 # Log files
│   ├── nameforge_detailed.log
│   └── nameforge_errors.log
└── 🗂️ __pycache__/          # Python cache
```

## ⚙️ Advanced Configuration

### Environment Variables (.env)

#### **Basic Configuration**
```bash
# Hugging Face Token (REQUIRED)
HF_TOKEN=your_hugging_face_token
```

#### **Optional Configuration**
```bash
# GPU (optional, for better performance)
CUDA_VISIBLE_DEVICES=0
```

### Environment Variables

| Variable | Description | Default Value |
|----------|-------------|---------------|
| `HF_TOKEN` | Hugging Face token (required) | - |
| `CUDA_VISIBLE_DEVICES` | Available GPUs to use | 0 |

### Server Configuration
You can modify `config.py` to change:
- **Port**: Default 7860
- **Host**: Default 127.0.0.1
- **Share**: Enable public access

### GPU Configuration
To use GPU:
1. Install CUDA Toolkit
2. Install PyTorch with CUDA support:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

## 🔧 Troubleshooting

### Error: "Model not found"
**Solution**:
1. Verify your Hugging Face token
2. Make sure you've accepted Gemma's terms
3. Run `python setup_gemma.py` again

### Error: "CUDA out of memory"
**Solution**:
1. Reduce the number of names to generate
2. Close other applications using GPU
3. Use CPU instead of GPU by modifying `config.py`

### Error: "Connection timeout" in domain verification
**Solution**:
1. Check your internet connection
2. Increase timeout in `config.py`
3. Reduce the number of domains to verify

### Application is slow
**Optimizations**:
1. Use GPU if available
2. Reduce the number of generated names
3. Close other heavy applications

## 📊 Logs and Debugging

### Log Files
- `logs/nameforge_detailed.log`: Detailed log of all operations
- `logs/nameforge_errors.log`: Only errors and warnings

### Enable Debug
Modify `config.py`:
```python
DEBUG_CONFIG = {
    "enable_api_logging": True,
    "enable_model_logging": True,
    # ... more options
}
```

## 🤝 Contributing

1. Fork the project
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License. See `LICENSE` for more details.

## 🙏 Acknowledgments

- **Google**: For the Gemma 3 model
- **Hugging Face**: For the model platform
- **Gradio**: For the web interface
- **Open Source Community**: For the libraries used

## 📞 Support

If you have problems or questions:
1. Check the [Troubleshooting](#-troubleshooting) section
2. Review the logs in `logs/`
3. Open an issue on GitHub

---

**Made with ❤️ for the startup community!**

*NameForge - Where ideas become memorable names* 🚀