import asyncio
import re
from pathlib import Path
import json

import nest_asyncio
import torch
from googletrans import Translator
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# ============ CONFIGURATION ============
MODEL_OPTIONS = None
with open("model_config.json", "r", encoding="utf-8") as f:
    MODEL_OPTIONS = json.load(f)
    MODEL_OPTIONS = [model for model in MODEL_OPTIONS["models"] if model["available"]]


def get_available_models():
    """
    Get list of only available models

    Returns:
        list: List of available model configurations
    """
    return [model for model in MODEL_OPTIONS if model["available"]]


def get_model_display_names():
    """
    Get list of available model display names for UI

    Returns:
        list: List of display names for available models only
    """
    available_models = get_available_models()
    return [model["display_name"] for model in available_models]


def get_model_options():
    """
    Get list of available models with their display names

    Returns:
        list: List of model configurations
    """
    return MODEL_OPTIONS


def get_model_display_names():
    """
    Get list of model display names for UI

    Returns:
        list: List of display names
    """
    return [model["display_name"] for model in MODEL_OPTIONS]


# =====================================


class GlossToVietnameseTranslator:
    def __init__(self, model_config=None):
        """
        Initialize the translator with the fine-tuned LoRA model

        Args:
            model_config: Dictionary with model configuration from MODEL_OPTIONS
        """
        if model_config is None:
            # Default to first model if no config provided
            model_config = MODEL_OPTIONS[0]

        self.model_path = Path(model_config["model_dir"])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Use the configuration from passed config
        self.base_model_name = model_config["model_name"]
        self.is_direct = model_config["is_direct"]
        self.model_config = model_config

        # Initialize model references
        self.tokenizer = None
        self.base_model = None
        self.model = None

        print(f"🤖 {self.base_model_name} on {self.device}")

        # Load the model and tokenizer
        self._load_model()

    def cleanup_memory(self):
        """
        Clean up GPU memory by properly deleting model instances
        """
        try:
            # Delete model instances
            if hasattr(self, "model") and self.model is not None:
                del self.model
                self.model = None

            if hasattr(self, "base_model") and self.base_model is not None:
                del self.base_model
                self.base_model = None

            if hasattr(self, "tokenizer") and self.tokenizer is not None:
                del self.tokenizer
                self.tokenizer = None

            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"⚠️ Warning during memory cleanup: {e}")

    def reload_model(self, new_model_config):
        """
        Reload the model with a new configuration

        Args:
            new_model_config: New model configuration dictionary
        """
        # Cleanup existing model first
        self.cleanup_memory()

        # Update configuration
        self.model_path = Path(new_model_config["model_dir"])
        self.base_model_name = new_model_config["model_name"]
        self.is_direct = new_model_config["is_direct"]
        self.model_config = new_model_config

        print(f"🤖 {self.base_model_name} on {self.device}")

        # Load the new model
        self._load_model()

    def _load_model(self):
        """Load the base model and apply LoRA adapter"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
            self.base_model = AutoModelForSeq2SeqLM.from_pretrained(
                self.base_model_name
            )

            # Load LoRA adapter
            self.model = PeftModel.from_pretrained(self.base_model, self.model_path)

            # Move to device
            self.model.to(self.device)
            self.model.eval()

        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def clean_text(self, text, text_type="gloss"):
        """
        Clean text based on type

        Args:
            text: Input text to clean
            text_type: "gloss", "german", or "vietnamese"

        Returns:
            Cleaned text
        """
        if not text or str(text).strip() == "":
            return ""

        text = str(text).strip()

        if text_type == "gloss":
            # Clean German gloss
            text = text.upper()
            text = re.sub(r"[^\w\s-]", "", text)
            text = re.sub(r"(\w+)-(\w+)", r"\1 \2", text)
            text = re.sub(r"(\w+)(\d+)", r"\1", text)
            text = re.sub(r"\bIX\b", "PUNKT", text)
            text = re.sub(r"\b(\w+)\s+\1\b", r"\1", text)
            text = re.sub(r"\s+", " ", text)
            text = re.sub(r"^[-\s]+|[-\s]+$", "", text)

        elif text_type == "german":
            # Clean German text
            text = text.lower()
            text = re.sub(r'[♪♫…""`~@#$%^&*()_+=\[\]{}|\\:;"<>?/]', "", text)
            text = re.sub(r"[^\w\s.,!?-äöüÄÖÜß]", "", text)
            text = re.sub(r"[.,!?]{2,}", ".", text)
            text = re.sub(r"(\d+)\s*grad", r"\1 grad", text)
            text = re.sub(r"minus\s+(\d+)", r"minus \1", text)
            text = re.sub(r"\s+([.,!?])", r"\1", text)
            text = re.sub(r"([.,!?])([^\s])", r"\1 \2", text)
            text = re.sub(r"\s+", " ", text)

        elif text_type == "vietnamese":
            # Clean Vietnamese text
            text = re.sub(r'[♪♫…""`~@#$%^&*()_+=\[\]{}|\\:;"<>?/]', "", text)
            text = re.sub(r"[^\w\s.,!?-]", "", text)
            text = re.sub(r"\s+([.,!?])", r"\1", text)
            text = re.sub(r"([.,!?])([^\s])", r"\1 \2", text)
            text = re.sub(r"\s+", " ", text)

        return text.strip()

    def translate_with_model(self, gloss_text):
        """
        Function 1: Feed the model and get output (either Vietnamese or German)

        Args:
            gloss_text: German sign language gloss string

        Returns:
            Model output (German sentence or Vietnamese sentence depending on IS_DIRECT_TRANSLATION)
        """
        try:
            # Clean the input gloss
            cleaned_gloss = self.clean_text(gloss_text, "gloss")

            if not cleaned_gloss:
                return ""

            # Tokenize input
            inputs = self.tokenizer(
                cleaned_gloss,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.device)

            # Generate output
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=128,
                    num_beams=4,
                    early_stopping=True,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            # Decode output
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Clean the output based on IS_DIRECT_TRANSLATION setting
            if self.is_direct:
                # Model outputs Vietnamese directly
                cleaned_output = self.clean_text(generated_text, "vietnamese")
            else:
                # Model outputs German
                cleaned_output = self.clean_text(generated_text, "german")

            # Log input and output
            print(f"📝 Input: {cleaned_gloss}")
            print(f"📤 Output: {cleaned_output}")

            return cleaned_output

        except Exception as e:
            print(f"Error in model translation: {e}")
            return ""

    def translate_german_to_vietnamese(self, german_text):
        """
        Function 2: Use Google Translate to convert German to Vietnamese (optional)

        Args:
            german_text: German sentence string

        Returns:
            Vietnamese sentence string
        """
        try:
            if not german_text or str(german_text).strip() == "":
                return ""

            # Clean the input
            cleaned_german = self.clean_text(german_text, "german")

            # Reinitialize translator to avoid async issues
            translator = Translator()

            # Use Google Translate with async handling
            result = translator.translate(cleaned_german, src="de", dest="vi")

            # Check if it's a coroutine (async) and handle appropriately
            if hasattr(result, "__await__"):
                # Handle async result
                try:
                    nest_asyncio.apply()
                    loop = asyncio.get_event_loop()
                    vietnamese_text = loop.run_until_complete(result).text
                except:
                    # Fallback if async handling fails
                    vietnamese_text = cleaned_german  # Return German as fallback
            else:
                # Regular synchronous result
                vietnamese_text = result.text

            # Clean the Vietnamese output
            cleaned_vietnamese = self.clean_text(vietnamese_text, "vietnamese")

            return cleaned_vietnamese

        except Exception as e:
            print(f"Error in German-to-Vietnamese translation: {e}")
            # Fallback - return the German text if translation fails
            return self.clean_text(german_text, "german")

    def translate_gloss_to_vietnamese(self, gloss):
        """
        Main pipeline function - uses simple boolean to choose method

        Args:
            gloss: German sign language gloss string

        Returns:
            Dictionary with results
        """
        try:
            if self.is_direct:
                # Direct method: gloss -> vietnamese (one step)
                vietnamese_output = self.translate_with_model(gloss)

                if not vietnamese_output:
                    return {
                        "input_gloss": gloss,
                        "method": "direct",
                        "vietnamese_sentence": "",
                        "error": "Model failed to generate Vietnamese output",
                    }

                return {
                    "input_gloss": gloss,
                    "method": "direct",
                    "vietnamese_sentence": vietnamese_output,
                    "error": None,
                }
            else:
                # Two-stage method: gloss -> german -> vietnamese (two steps)
                german_output = self.translate_with_model(gloss)

                if not german_output:
                    return {
                        "input_gloss": gloss,
                        "method": "two_stage",
                        "german_sentence": "",
                        "vietnamese_sentence": "",
                        "error": "Model failed to generate German output",
                    }

                vietnamese_sentence = self.translate_german_to_vietnamese(german_output)

                return {
                    "input_gloss": gloss,
                    "method": "two_stage",
                    "german_sentence": german_output,
                    "vietnamese_sentence": vietnamese_sentence,
                    "error": None,
                }

        except Exception as e:
            return {
                "input_gloss": gloss,
                "method": "direct" if self.is_direct else "two_stage",
                "vietnamese_sentence": "",
                "error": f"Pipeline error: {str(e)}",
            }
