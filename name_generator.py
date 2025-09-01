"""
NameForge - Final Fixed Name Generation Module
Completely handles CUDA tensor issues with Gemma 3
"""

import re
import random
import string
import time
import logging
import os
from typing import List, Optional, Dict, Any

# Asegurar que TensorFlow no se cargue antes de importar transformers
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import torch

# Activar TF32 lo antes posible tras importar torch
try:
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
except Exception:
    pass

from transformers import AutoModelForCausalLM, AutoTokenizer

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from config import (
    GPU_CONFIG,
    logger,
)

class NameGenerator:
    """Main class for generating startup names using Google's Gemma 3 model."""
    
    def __init__(self, model_name: str = None, max_retries: int = 3):
        self.model_name = model_name or "google/gemma-3-270m-it"
        self.model = None
        self.tokenizer = None
        self.device = self._setup_device()
        self.model_loaded = False
        self._load_model()
    
    def _setup_device(self):
        """Setup the best available device for model execution."""
        # Ya usamos GPU_CONFIG importado arriba
        if GPU_CONFIG.get("force_cpu", False):
            logger.info("GPU usage forced to CPU by configuration")
            return "cpu"
        
        if torch.cuda.is_available() and GPU_CONFIG.get("enabled", True):
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"GPU detected: {gpu_name} with {gpu_memory:.1f}GB memory")
            
            if gpu_memory >= 2.0:
                logger.info("Using CUDA GPU for model execution")
                return "cuda"
            else:
                logger.warning(f"GPU memory ({gpu_memory:.1f}GB) may be insufficient, using CPU")
                return "cpu"
        else:
            logger.info("CUDA not available, using CPU")
            return "cpu"

    def _load_model(self):
        try:
            logger.info(f"Loading Gemma 3 model: {self.model_name}")
            
            hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            
            # Load tokenizer with proper configuration
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                token=hf_token,
                padding_side="left"  # Important for Gemma 3
            )
            
            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Always load on CPU first to avoid CUDA issues
            logger.info("Loading model on CPU first for stability...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.float32,
                device_map="cpu",
                token=hf_token,
                low_cpu_mem_usage=True
            )
            
            # Move to GPU only if explicitly requested and available
            if self.device == "cuda":
                logger.info("Moving model to GPU...")
                try:
                    self.model = self.model.to("cuda")
                    # Test GPU generation with a simple prompt
                    test_inputs = self.tokenizer("Hello", return_tensors="pt").to("cuda")
                    with torch.no_grad():
                        test_output = self.model.generate(
                            **test_inputs,
                            max_new_tokens=5,
                            pad_token_id=self.tokenizer.eos_token_id,
                            eos_token_id=self.tokenizer.eos_token_id
                        )
                    logger.info("GPU test successful, using CUDA")
                except Exception as e:
                    logger.warning(f"GPU test failed: {e}, falling back to CPU")
                    self.device = "cpu"
                    self.model = self.model.to("cpu")
            else:
                logger.info("Using CPU for model execution")
            
            self.model.eval()
            self.model_loaded = True
            logger.info(f"Successfully loaded Gemma 3 model: {self.model_name} on {self.device}")
            
        except Exception as e:
            logger.error(f"Error loading Gemma 3 model {self.model_name}: {e}")
            self.model = None
            self.tokenizer = None
            self.model_loaded = False

    def _build_prompt(self, description: str, style: str, count: int) -> str:
        """Build a prompt optimized for Gemma 3 instruction-tuned format."""
        # Get industry-specific keywords
        industry_keywords = self._get_industry_keywords(description)
        
        prompt = f"""<start_of_turn>user
Generate {count} creative startup names for: {description}

Style preference: {style}

Requirements:
- Names should be memorable, brandable, and easy to pronounce
- Each name MUST be 3-12 characters long without spaces
- Names should clearly reflect the business purpose and industry
- Avoid generic terms like "app", "tech", "hub", "ly", "name"
- Make names that sound professional and trustworthy
- IMPORTANT: Provide ONLY the names in a numbered list format (1. Name)
- DO NOT include any descriptions, explanations, or additional text
- DO NOT use any formatting like bold or italics
- DO NOT use generic words like "Name" or "Example"
- Focus on words related to: {industry_keywords}

Please provide exactly {count} names, one per line, starting with a number. DO NOT include any other text.
<end_of_turn>
<start_of_turn>model
"""
        return prompt

    def _get_industry_keywords(self, description: str) -> str:
        """Extract industry-specific keywords from description."""
        description_lower = description.lower()
        
        # Industry-specific keyword mapping
        industry_keywords = {
            'fitness': 'fitness, health, strength, energy, vitality, wellness, training, nutrition, workout, active',
            'tech': 'technology, innovation, digital, smart, connect, data, cloud, mobile, web, app',
            'finance': 'finance, money, wealth, invest, trade, bank, secure, trust, growth, capital',
            'health': 'health, medical, care, wellness, therapy, treatment, recovery, healing, medicine',
            'education': 'education, learn, teach, study, knowledge, skill, training, course, academy',
            'food': 'food, cuisine, taste, fresh, organic, healthy, delicious, cooking, restaurant',
            'travel': 'travel, journey, adventure, explore, discover, destination, trip, vacation',
            'fashion': 'fashion, style, design, trend, beauty, clothing, accessories, luxury',
            'business': 'business, enterprise, corporate, professional, service, solution, strategy',
            'entertainment': 'entertainment, fun, creative, media, content, gaming, music, film'
        }
        
        # Find the best matching industry
        best_match = 'business'  # default
        best_score = 0
        
        for industry, keywords in industry_keywords.items():
            score = sum(1 for keyword in keywords.split(', ') if keyword in description_lower)
            if score > best_score:
                best_score = score
                best_match = industry
        
        return industry_keywords.get(best_match, 'business, professional, service')

    def generate_names(self, description: str, style: str = "modern", length: int = 8, count: int = 10) -> List[str]:
        """Generate startup names using Gemma 3 model."""
        if not self.model_loaded:
            logger.error("Model not loaded, cannot generate names")
            return []
        
        logger.info("Starting name generation process...")
        
        # Try model generation first
        try:
            names = self._generate_with_model(description, style, count)
            if names:
                return names[:count]
        except Exception as e:
            logger.warning(f"Model generation failed: {e}")
        
        # Fallback to rule-based generation
        logger.info("Using fallback name generation method")
        return self._generate_fallback_names(description, style, length, count)

    def _generate_with_model(self, description: str, style: str, count: int) -> List[str]:
        """Generate names using the loaded model."""
        prompt = self._build_prompt(description, style, count)
        
        try:
            # Simple tokenization without padding to avoid CUDA issues
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                padding=False, 
                truncation=False
            ).to(self.device)
            
            logger.debug(f"Input shape: {inputs['input_ids'].shape}")
            
            with torch.no_grad():
                try:
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=200,  # Increased for better completion
                        temperature=0.7,     # Lower for more focused names
                        top_p=0.85,         # More focused sampling
                        do_sample=True,
                        repetition_penalty=1.2,  # Prevent repetitive names
                        eos_token_id=self.tokenizer.eos_token_id,
                        pad_token_id=self.tokenizer.eos_token_id,
                        num_return_sequences=1,
                        use_cache=True,
                        early_stopping=False  # Disable to avoid issues
                    )
                except RuntimeError as e:
                    if "CUDA out of memory" in str(e):
                        logger.error("CUDA out of memory error. Trying with reduced parameters.")
                        # Intentar con parámetros reducidos
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=100,  # Reducido
                            temperature=0.7,
                            top_p=0.9,
                            do_sample=True,
                            num_return_sequences=1,
                            use_cache=True
                        )
                    else:
                        raise
            
            output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
            logger.debug(f"Raw model output: {output_text}")
            
            # Extract names from output
            names = self._extract_names_from_output(output_text, count)
            logger.info(f"Model generated {len(names)} names")
            
            # Si no se encontraron suficientes nombres, intentar una segunda vez con temperatura más alta
            if len(names) < count:
                logger.warning(f"Only found {len(names)} names, trying again with higher temperature")
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=200,
                        temperature=0.9,  # Temperatura más alta para más creatividad
                        top_p=0.95,      # Más variedad
                        do_sample=True,
                        repetition_penalty=1.1,
                        num_return_sequences=1,
                        use_cache=True
                    )
                
                output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
                additional_names = self._extract_names_from_output(output_text, count - len(names))
                names.extend([name for name in additional_names if name not in names])
                logger.info(f"Second attempt generated {len(additional_names)} additional names")
            
            return names
            
        except Exception as e:
            logger.error(f"Error during model generation: {str(e)}")
            logger.exception("Detailed error information:")
            return []

    def _extract_names_from_output(self, output_text: str, count: int) -> List[str]:
        """Extract names from Gemma 3 output with numbered format."""
        import re
        logger.debug(f"Raw output text: {repr(output_text)}")
        
        # Eliminar texto antes de la primera línea numerada
        if "<start_of_turn>model" in output_text:
            output_text = output_text.split("<start_of_turn>model")[-1]
        if "Here are" in output_text:
            output_text = output_text.split("Here are")[-1]
        
        # Lista de palabras genéricas a excluir
        excluded_words = ['name', 'example', 'user', 'model', 'names', 'project', 'startup', 'creative', 
                          'here', 'are', 'the', 'for', 'your', 'this', 'that', 'with', 
                          'and', 'app', 'platform', 'service', 'company', 'tech', 'hub', 'ly']
        
        # Patrón simple para extraer nombres numerados (1. Nombre)
        pattern = r'\d+\.\s*([A-Za-z0-9 _-]+)'
        
        # Extraer todos los nombres que coincidan con el patrón
        names = []
        for line in output_text.strip().split('\n'):
            # Ignorar líneas en otros idiomas (que contienen caracteres no latinos)
            if re.search(r'[^\x00-\x7F]', line) and not re.search(r'[A-Za-z]', line):
                continue
                
            match = re.search(pattern, line)
            if match:
                name = match.group(1).strip()
                # Verificar que el nombre tenga al menos un carácter alfabético y no sea una palabra genérica
                if (any(c.isalpha() for c in name) and 
                    name.lower() not in excluded_words and 
                    len(name) >= 3):
                    names.append(name)
        
        # Si no se encontraron nombres, intentar con un patrón más flexible
        if not names:
            # Buscar cualquier palabra que parezca un nombre (primera letra mayúscula, al menos 3 caracteres)
            alt_pattern = r'\b([A-Z][A-Za-z0-9]{2,})\b'
            for line in output_text.strip().split('\n'):
                if re.search(r'[^\x00-\x7F]', line) and not re.search(r'[A-Za-z]', line):
                    continue
                    
                matches = re.findall(alt_pattern, line)
                for match in matches:
                    if (match.lower() not in excluded_words and 
                        len(match) >= 3 and len(match) <= 20):
                        names.append(match)
        
        # Eliminar duplicados manteniendo el orden
        unique_names = []
        for name in names:
            if name not in unique_names:
                unique_names.append(name)
                
        logger.debug(f"Extracted {len(unique_names)} names: {unique_names}")
        return unique_names[:count]

    def _generate_fallback_names(self, description: str, style: str, length: int, count: int) -> List[str]:
        """Generate fallback names using industry keywords."""
        logger.info("Using fallback name generation method")
        
        industry_keywords = self._get_industry_keywords(description).split(', ')
        keywords = re.findall(r'\b\w+\b', description.lower())
        keywords = [kw for kw in keywords if len(kw) >= 3 and kw not in ['the', 'and', 'for', 'with', 'this', 'that', 'app', 'platform', 'service', 'company']]
        
        all_words = industry_keywords + keywords
        all_words = [word for word in all_words if len(word) >= 3]
        
        names = []
        attempts = 0
        max_attempts = count * 20
        
        while len(names) < count and attempts < max_attempts:
            attempts += 1
            
            # Combine words creatively
            if len(all_words) >= 2:
                word1 = random.choice(all_words)
                word2 = random.choice(all_words)
                
                # Create combinations
                combinations = [
                    word1 + word2[:3],
                    word1[:4] + word2,
                    word1 + word2[-2:],
                    word1[:3] + word2[-3:]
                ]
                
                for combo in combinations:
                    if 3 <= len(combo) <= length + 2 and combo.isalnum():
                        names.append(combo)
            
            # Add some single words
            if random.random() < 0.3:
                word = random.choice(all_words)
                if 3 <= len(word) <= length + 2:
                    names.append(word)
        
        # Ensure uniqueness and proper count
        unique_names = list(dict.fromkeys(names))[:count]
        logger.info(f"Fallback generation produced {len(unique_names)} names")
        return unique_names

# Test the final generator
if __name__ == "__main__":
    print("🧪 Testing final NameGenerator...")
    
    generator = NameGenerator()
    
    if generator.model_loaded:
        print("✅ Model loaded successfully")
        
        # Test generation
        names = generator.generate_names(
            "A fitness tracking app that helps people monitor workouts and nutrition",
            style="techy",
            length=8,
            count=5
        )
        
        print(f"✅ Generated names: {names}")
    else:
        print("❌ Model failed to load")
