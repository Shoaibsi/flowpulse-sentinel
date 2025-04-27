"""
Gemini Analyzer module for FlowPulse Sentinel
Handles AI analysis using Google's Gemini API
"""
import time
import logging
from datetime import datetime
from typing import Dict, Optional, List
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from google.api_core import exceptions as google_exceptions
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    RetryError
)
from flowpulse.config import Config
from flowpulse.utils.sanitizer import ResponseSanitizer

logger = logging.getLogger("FlowPulseBot.GeminiAnalyzer")

class GeminiAnalyzer:
    """Handles all Gemini API interactions with robust error handling"""
    
    def __init__(self, api_key: Optional[str] = None):
        try:
            self.config = Config()
            genai.configure(api_key=api_key or self.config.GEMINI_API_KEY)
            
            # Initialize model with safety settings
            self.model = genai.GenerativeModel(
                model_name=self.config.GEMINI_MODEL_NAME,
                safety_settings=self._convert_safety_settings(self.config.GEMINI_SAFETY_SETTINGS),
                generation_config=self.config.GEMINI_GENERATION_CONFIG
            )
            
            # Usage tracking
            self._minute = datetime.now().minute
            self._requests_this_minute = 0
            self._consecutive_errors = 0
            
            # Test connection
            self._test_connection()
            logger.info("Gemini analyzer initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")
            raise

    def _convert_safety_settings(self, settings: Dict) -> List[Dict]:
        """Convert string settings to Gemini's enum format"""
        category_map = {
            "HARASSMENT": HarmCategory.HARM_CATEGORY_HARASSMENT,
            "HATE_SPEECH": HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            "SEXUAL": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            "DANGEROUS": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT
        }
        
        threshold_map = {
            "block_none": HarmBlockThreshold.BLOCK_NONE,
            "block_low": HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
            "block_med": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            "block_high": HarmBlockThreshold.BLOCK_ONLY_HIGH
        }
        
        return [
            {
                "category": category_map[k],
                "threshold": threshold_map[v]
            }
            for k, v in settings.items()
            if k in category_map and v in threshold_map
        ]

    @property
    def rate_limit_delay(self) -> float:
        """Dynamic delay based on usage patterns"""
        return min(2 ** self._consecutive_errors, 30)  # Max 30s delay

    def _track_usage(self):
        """Prevent budget overruns"""
        current_minute = datetime.now().minute
        if self._minute != current_minute:
            self._minute = current_minute
            self._requests_this_minute = 0
        
        self._requests_this_minute += 1
        if self._requests_this_minute > self.config.GEMINI_USAGE_LIMIT:
            raise RuntimeError("Gemini rate limit exceeded")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(
            (google_exceptions.GoogleAPIError, TimeoutError)
        )
    )
    def _test_connection(self) -> bool:
        """Verify API connectivity with exponential backoff"""
        try:
            response = self.model.generate_content(
                "Connection test",
                request_options={"timeout": 5}
            )
            return response.text == "Connection test"
        except Exception as e:
            logger.warning(f"Connection test failed: {e}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def generate_response(self, prompt: str) -> str:
        """Generate response with automatic retries and sanitization"""
        try:
            # Track usage
            self._track_usage()
            
            response = self.model.generate_content(
                prompt,
                request_options={"timeout": 10}
            )
            
            # Handle potential content blockers
            if not response.text:
                if response.prompt_feedback.block_reason:
                    raise ValueError(
                        f"Content blocked: {response.prompt_feedback.block_reason}"
                    )
                raise ValueError("Empty response from Gemini")
            
            # Reset error counter on success
            self._consecutive_errors = 0
            
            return ResponseSanitizer.sanitize(response.text)
            
        except genai.core.exceptions.InvalidArgument as e:
            logger.error(f"Invalid prompt structure: {e}")
            self._consecutive_errors += 1
            raise ValueError("Invalid analysis request") from e
        except google_exceptions.GoogleAPIError as e:
            logger.error(f"Generation failed: {e}")
            self._consecutive_errors += 1
            raise
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            self._consecutive_errors += 1
            raise
        except RetryError:
            logger.error("Max retries exceeded - cooling down")
            self._consecutive_errors += 1
            time.sleep(self.rate_limit_delay)
            raise
