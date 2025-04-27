"""
Response sanitizer module for FlowPulse Sentinel
Handles sanitization of AI-generated responses
"""
import re
import logging
from typing import Dict, Match, Callable

logger = logging.getLogger("FlowPulseBot.Sanitizer")

class ResponseSanitizer:
    """Utility class for sanitizing AI-generated responses"""
    
    @staticmethod
    def sanitize(response: str) -> str:
        """
        Securely sanitizes model output while preserving:
        - Price targets ($450.50, $SPY 500C)
        - Mathematical ranges (500-510, +3.2%)
        - Trading symbols (TSLA, SPY, QQQ)
        """
        if not response:
            return ""
            
        try:
            # Step 1: Temporarily protect trading patterns
            protected = []
            
            def protect_match(match):
                protected.append(match.group(0))
                return f"||PROTECTED_{len(protected)-1}||"
            
            # Protect price patterns before general sanitization
            response = re.sub(
                r'''
                (\$[A-Z]{1,5}\s?\d{0,5}[CP]?)|  # $SPY 500C
                (\$\d+\.?\d*\s?-\s?\$\d+\.?\d*)| # $450.50 - $455.50
                ([+-]?\d+\.?\d*%)                # +5.2%
                ''',
                protect_match,
                response,
                flags=re.VERBOSE
            )
            
            # Step 2: Standard sanitization
            response = re.sub(r'http[s]?://\S+', '[URL REMOVED]', response)  # URLs
            response = re.sub(r'\b(SELECT|INSERT|DROP|DELETE|FROM|WHERE)\b', '[SQL REMOVED]', response, flags=re.IGNORECASE)
            response = re.sub(r'```.*?```', '[CODE REMOVED]', response, flags=re.DOTALL)
            
            # Step 3: Restore protected trading patterns
            for i, pattern in enumerate(protected):
                response = response.replace(f"||PROTECTED_{i}||", pattern)
                
            return response
            
        except Exception as e:
            logger.error(f"Error in sanitization: {e}")
            return "Error in response sanitization"
            
    @staticmethod
    def create_placeholder(match: Match) -> str:
        """Helper method for creating placeholders"""
        return f"PLACEHOLDER_{hash(match.group(0))}"
