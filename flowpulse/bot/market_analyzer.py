"""
Market Analyzer module for FlowPulse Sentinel
Unified market analysis interface using Gemini API
"""
import logging
import re
from typing import Dict, List, Optional, Any
from flowpulse.config import Config
from .gemini_analyzer import GeminiAnalyzer

logger = logging.getLogger("FlowPulseBot.MarketAnalyzer")

class MarketAnalyzer:
    """Unified analysis interface (replaces DeepSeekAnalyzer)"""
    
    def __init__(self):
        self.config = Config()
        try:
            self.analyzer = GeminiAnalyzer(self.config.GEMINI_API_KEY)
            logger.info("Market analyzer initialized with Gemini")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini analyzer: {e}")
            self.analyzer = None
    
    def analyze_gamma_squeeze(self, ticker: str, call_oi: int, put_oi: int, gamma: float) -> Dict:
        """Analyze potential gamma squeeze"""
        if not self.analyzer:
            return self._fallback_analysis("gamma_squeeze", ticker)
            
        # Calculate derived metrics
        call_put_ratio = call_oi / max(put_oi, 1)  # Avoid division by zero
        volume_oi_ratio = 2.0  # Default placeholder
            
        prompt = self._create_gamma_prompt({
            "ticker": ticker,
            "call_oi": call_oi,
            "put_oi": put_oi,
            "gamma": gamma,
            "call_put_ratio": call_put_ratio,
            "volume_oi_ratio": volume_oi_ratio
        })
        
        try:
            response = self.analyzer.generate_response(prompt)
            return self._parse_response(response, "gamma_squeeze")
        except Exception as e:
            logger.error(f"Gamma squeeze analysis failed: {e}")
            return self._fallback_analysis("gamma_squeeze", ticker)
    
    def analyze_iv_crush(self, ticker: str, current_iv: float, historical_iv: float, 
                        days_to_earnings: int) -> Dict:
        """Analyze potential IV crush"""
        if not self.analyzer:
            return self._fallback_analysis("iv_crush", ticker)
            
        # Calculate derived metrics
        iv_percentile = min(100, max(0, (current_iv - historical_iv) / 
                           max(historical_iv, 0.01) * 100))
        iv_premium = current_iv - historical_iv
            
        prompt = self._create_iv_crush_prompt({
            "ticker": ticker,
            "current_iv": current_iv,
            "historical_iv": historical_iv,
            "days_to_earnings": days_to_earnings,
            "iv_percentile": iv_percentile,
            "iv_premium": iv_premium
        })
        
        try:
            response = self.analyzer.generate_response(prompt)
            return self._parse_response(response, "iv_crush")
        except Exception as e:
            logger.error(f"IV crush analysis failed: {e}")
            return self._fallback_analysis("iv_crush", ticker)
    
    def analyze_earnings_play(self, ticker: str, expected_move: float, 
                             tft_prediction: float, sentiment_score: float) -> Dict:
        """Analyze earnings play strategy"""
        if not self.analyzer:
            return self._fallback_analysis("earnings_play", ticker)
            
        # Add historical metrics (placeholders)
        hist_surprise = 2.5  # Default placeholder
        avg_move = 3.0  # Default placeholder
        iv_rank = 75.0  # Default placeholder
            
        prompt = self._create_earnings_prompt({
            "ticker": ticker,
            "expected_move": expected_move,
            "tft_prediction": tft_prediction,
            "sentiment_score": sentiment_score,
            "hist_surprise": hist_surprise,
            "avg_move": avg_move,
            "iv_rank": iv_rank
        })
        
        try:
            response = self.analyzer.generate_response(prompt)
            return self._parse_response(response, "earnings_play")
        except Exception as e:
            logger.error(f"Earnings play analysis failed: {e}")
            return self._fallback_analysis("earnings_play", ticker)
    
    def _create_gamma_prompt(self, params: Dict) -> str:
        """Create prompt for gamma squeeze analysis"""
        return f"""
        [SYSTEM]: You are an expert options trader analyzing gamma squeeze potential.
        [DATA]:
        - Ticker: {params['ticker']}
        - Call OI: {params['call_oi']:,}
        - Put OI: {params['put_oi']:,}
        - Call/Put Ratio: {params['call_put_ratio']:.2f}
        - Gamma Value: {params['gamma']:.4f}
        - Options Volume/OI Ratio: {params['volume_oi_ratio']:.2f}
        
        [INSTRUCTIONS]:
        1. Analyze squeeze probability (1-10 scale)
        2. Identify key price levels
        3. Provide price target range
        4. Suggest risk management strategy
        5. Format as JSON with keys: squeeze_probability, price_target, recommended_hedge, analysis_type
        """
    
    def _create_iv_crush_prompt(self, params: Dict) -> str:
        """Create prompt for IV crush analysis"""
        return f"""
        [SYSTEM]: You are an elite options strategist analyzing volatility and earnings events.
        [DATA]:
        - Ticker: {params['ticker']}
        - Current IV: {params['current_iv']:.2f}%
        - Historical IV Average: {params['historical_iv']:.2f}%
        - IV Percentile: {params['iv_percentile']:.2f}%
        - Days Until Earnings: {params['days_to_earnings']}
        - IV Premium: {params['iv_premium']:.2f}%
        
        [INSTRUCTIONS]:
        1. Analyze IV crush probability (1-10 scale)
        2. Estimate expected post-earnings IV level
        3. Recommend optimal strategy
        4. Format as JSON with keys: crush_probability, expected_iv, recommended_strategy, analysis_type
        """
    
    def _create_earnings_prompt(self, params: Dict) -> str:
        """Create prompt for earnings play analysis"""
        return f"""
        [SYSTEM]: You are a veteran market strategist analyzing earnings events and options positioning.
        [DATA]:
        - Ticker: {params['ticker']}
        - Expected Move (Options Implied): {params['expected_move']:.2f}%
        - TFT Model Prediction: {params['tft_prediction']:.2f}%
        - Market Sentiment Score: {params['sentiment_score']:.2f}
        - Historical Earnings Surprise: {params['hist_surprise']:.2f}%
        - Average Post-Earnings Move: {params['avg_move']:.2f}%
        - IV Rank: {params['iv_rank']:.2f}%
        
        [INSTRUCTIONS]:
        1. Determine directional bias (bullish/bearish/neutral)
        2. Assess confidence level (1-10 scale)
        3. Recommend optimal position strategy
        4. Format as JSON with keys: directional_bias, confidence_score, recommended_position, analysis_type
        """
    
    def _parse_response(self, response: str, analysis_type: str) -> Dict:
        """Parse response into structured data"""
        try:
            # Try to parse as JSON first
            import json
            try:
                result = json.loads(response)
                # Add analysis_type if not present
                if "analysis_type" not in result:
                    result["analysis_type"] = analysis_type
                return result
            except json.JSONDecodeError:
                # Fall back to text parsing
                pass
            
            # Extract structured data based on analysis type
            if analysis_type == "gamma_squeeze":
                squeeze_prob = self._extract_value_by_keyword(response, ["squeeze probability", "probability", "likelihood"], default=5.0)
                price_target = self._extract_text_by_keyword(response, ["price target", "target zone", "price range"])
                hedge_strategy = self._extract_text_by_keyword(response, ["recommended hedge", "hedging strategy", "hedge"])
                
                return {
                    "squeeze_probability": squeeze_prob,
                    "price_target": price_target,
                    "recommended_hedge": hedge_strategy,
                    "analysis_type": "gamma_squeeze",
                    "raw_analysis": self._extract_summary(response)
                }
            elif analysis_type == "iv_crush":
                crush_prob = self._extract_value_by_keyword(response, ["iv crush probability", "crush probability", "probability"], default=5.0)
                expected_iv = self._extract_text_by_keyword(response, ["expected iv", "post-earnings iv", "iv level"])
                strategy = self._extract_text_by_keyword(response, ["recommended strategy", "optimal strategy", "strategy"])
                
                return {
                    "crush_probability": crush_prob,
                    "expected_iv": expected_iv,
                    "recommended_strategy": strategy,
                    "analysis_type": "iv_crush",
                    "raw_analysis": self._extract_summary(response)
                }
            elif analysis_type == "earnings_play":
                bias = self._extract_text_by_keyword(response, ["directional bias", "bias", "direction"])
                confidence = self._extract_value_by_keyword(response, ["confidence score", "confidence", "conviction"], default=5.0)
                position = self._extract_text_by_keyword(response, ["recommended position", "position", "strategy"])
                
                return {
                    "directional_bias": bias,
                    "confidence_score": confidence,
                    "recommended_position": position,
                    "analysis_type": "earnings_play",
                    "raw_analysis": self._extract_summary(response)
                }
            else:
                return self._fallback_analysis(analysis_type, "")
        except Exception as e:
            logger.error(f"Error parsing response: {e}")
            return self._fallback_analysis(analysis_type, "")
    
    def _extract_value_by_keyword(self, text: str, keywords: List[str], default: float = 0.0) -> float:
        """Extract numerical value from text based on keywords"""
        lines = text.split("\n")
        for line in lines:
            line_lower = line.lower()
            for keyword in keywords:
                if keyword.lower() in line_lower:
                    # Extract numerical value
                    numbers = re.findall(r'\d+\.?\d*', line)
                    if numbers:
                        return float(numbers[0])
        return default
    
    def _extract_text_by_keyword(self, text: str, keywords: List[str]) -> str:
        """Extract text based on keywords"""
        lines = text.split("\n")
        for line in lines:
            line_lower = line.lower()
            for keyword in keywords:
                if keyword.lower() in line_lower:
                    # Extract text after colon or number with dot
                    parts = re.split(r':\s*|\d+\.\s+', line, 1)
                    if len(parts) > 1:
                        return parts[1].strip()
                    return line.strip()
        return ""
    
    def _extract_summary(self, text: str, max_length: int = 200) -> str:
        """Extract a concise summary from the full analysis"""
        # Get the first few sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        summary = ""
        
        for sentence in sentences:
            if len(summary) + len(sentence) <= max_length:
                summary += sentence + " "
            else:
                break
                
        return summary.strip()
    
    def _fallback_analysis(self, analysis_type: str, ticker: str) -> Dict:
        """Provide fallback analysis when model fails"""
        if analysis_type == "gamma_squeeze":
            return {
                "squeeze_probability": 5.0,
                "price_target": f"{ticker} likely to remain within recent range",
                "recommended_hedge": "Consider collar strategy for protection",
                "analysis_type": "gamma_squeeze",
                "is_fallback": True
            }
        elif analysis_type == "iv_crush":
            return {
                "crush_probability": 5.0,
                "expected_iv": "Expect 30-40% IV reduction post-earnings",
                "recommended_strategy": "Consider iron condors or calendar spreads",
                "analysis_type": "iv_crush",
                "is_fallback": True
            }
        elif analysis_type == "earnings_play":
            return {
                "directional_bias": "Neutral",
                "confidence_score": 5.0,
                "recommended_position": "Consider iron condor or straddle based on IV",
                "analysis_type": "earnings_play",
                "is_fallback": True
            }
        else:
            return {
                "analysis_type": analysis_type,
                "is_fallback": True,
                "message": "Analysis not available"
            }
