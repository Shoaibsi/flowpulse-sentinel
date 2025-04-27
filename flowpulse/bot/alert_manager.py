"""
Alert Manager module for FlowPulse Sentinel
Handles sending alerts via Telegram
"""
import logging
import time
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from telegram import Bot
from telegram.error import TelegramError

logger = logging.getLogger("FlowPulseBot.AlertManager")

class TelegramAlertManager:
    """Step 31-40: Alert System Implementation"""
    def __init__(self, token: str, chat_id: Optional[str] = None, max_alerts_per_hour: int = 10):
        self.token = token
        self.chat_id = chat_id
        self.max_alerts_per_hour = max_alerts_per_hour
        self.alert_history = []
        self.digest_mode = False
        self.digest_alerts = []
        self.bot = None
        self.bot_info = None
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        # Initialize bot
        self.bot = self._sync_initialize()
            
    def _sync_initialize(self):
        """Initialize bot synchronously using asyncio.run"""
        try:
            return self.loop.run_until_complete(self._initialize_bot())
        except Exception as e:
            logger.error(f"Failed to initialize Telegram bot: {e}")
            return None
            
    async def _initialize_bot(self):
        """Initialize bot asynchronously"""
        try:
            bot = Bot(token=self.token)
            me = await bot.get_me()
            logger.info(f"Telegram bot initialized: @{me.username}")
            self.bot_info = me
            return bot
        except Exception as e:
            logger.error(f"Telegram init error: {e}")
            return None
            
    def send_alert(self, ticker: str, flow: Dict, tft_pred: float, analysis: str) -> bool:
        """Send an alert via Telegram"""
        if self.bot is None:
            logger.error("Telegram bot not initialized, cannot send alert")
            return False
            
        # Check rate limiting
        if not self._check_rate_limit():
            logger.warning("Rate limit exceeded, alert not sent")
            
            # Add to digest if in digest mode
            if self.digest_mode:
                self.digest_alerts.append({
                    'ticker': ticker,
                    'flow': flow,
                    'tft_pred': tft_pred,
                    'analysis': analysis,
                    'timestamp': datetime.now()
                })
            return False
            
        # Format the alert message
        message = self._format_alert(ticker, flow, tft_pred, analysis)
        
        try:
            # Send message using asyncio.run
            if self.chat_id:
                success = self._send_telegram_message(message)
                if not success:
                    logger.error("Failed to send Telegram message")
                    return False
            else:
                logger.warning("No chat_id provided, alert not sent")
                return False
                
            # Record alert in history
            self.alert_history.append({
                'ticker': ticker,
                'timestamp': datetime.now(),
                'priority': self._calculate_priority(ticker, flow, tft_pred)
            })
            
            logger.info(f"Alert sent for {ticker}")
            return True
        except Exception as e:
            logger.error(f"Failed to send Telegram alert: {e}")
            return False
            
    def _send_telegram_message(self, message: str) -> bool:
        """Send a message via Telegram using a new event loop"""
        try:
            return self.loop.run_until_complete(
                self._async_send_message(message)
            )
        except Exception as e:
            logger.error(f"Telegram send error: {e}")
            return False
            
    async def _async_send_message(self, message: str) -> bool:
        """Send a message via Telegram asynchronously"""
        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode="MarkdownV2"
            )
            return True
        except Exception as e:
            logger.error(f"Telegram send error: {e}")
            # Try again without markdown if there's a parsing error
            if "can't parse entities" in str(e).lower():
                try:
                    logger.warning("Retrying without markdown formatting")
                    await self.bot.send_message(
                        chat_id=self.chat_id,
                        text=message.replace('*', '').replace('_', ''),
                        parse_mode=None
                    )
                    return True
                except Exception as e2:
                    logger.error(f"Telegram plain text send error: {e2}")
            return False
            
    def _format_alert(self, ticker: str, flow: Dict, tft_pred: Optional[float], analysis: str) -> str:
        """Format the alert message"""
        # Format prediction safely
        if isinstance(tft_pred, (int, float)):
            prediction_str = f"{tft_pred:.2f}%"
        else:
            prediction_str = "N/A" # Or some other placeholder
            
        # Format the message with Markdown
        message = f"""
*{ticker} Alert*

*Price Prediction:* {prediction_str}
*Options Flow:* {flow.get('volume', 0)}/{flow.get('open_interest', 0)} ({flow.get('ratio', 0):.1f}x)
*IV Percentile:* {flow.get('iv_rank', 0):.0%}

*Analysis:*
{self._escape_markdown_v2(analysis)}

_Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_
"""
        return message
        
    def _escape_markdown_v2(self, text: str) -> str:
        """Escape special characters for Markdown V2 formatting in Telegram
        
        Based on Telegram's MarkdownV2 requirements:
        https://core.telegram.org/bots/api#markdownv2-style
        """
        escape_chars = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
        for char in escape_chars:
            text = text.replace(char, f"\\{char}")
        return text
        
    def _check_rate_limit(self) -> bool:
        """Step 34: Rate limiting implementation"""
        # Remove alerts older than 1 hour
        now = datetime.now()
        self.alert_history = [
            alert for alert in self.alert_history 
            if now - alert['timestamp'] < timedelta(hours=1)
        ]
        
        # Check if we've exceeded the rate limit
        return len(self.alert_history) < self.max_alerts_per_hour
        
    def _calculate_priority(self, ticker: str, flow: Dict, tft_pred: float) -> float:
        """Calculate alert priority (0-1 scale)"""
        # Factors that increase priority:
        # 1. High volume/OI ratio
        # 2. Strong TFT prediction
        # 3. High IV rank
        
        volume = flow.get('volume', 0)
        open_interest = flow.get('open_interest', 1)
        ratio = flow.get('ratio', volume / open_interest if open_interest > 0 else 0)
        iv_rank = flow.get('iv_rank', 0)
        
        # Normalize ratio (cap at 10x)
        ratio_score = min(ratio / 10, 1.0)
        
        # Normalize TFT prediction (cap at ±5%)
        tft_score = min(abs(tft_pred) / 0.05, 1.0)
        
        # Calculate priority score (weighted average)
        priority = (0.4 * ratio_score) + (0.3 * tft_score) + (0.3 * iv_rank)
        
        return priority
        
    def send_digest(self) -> bool:
        """Step 35: Send digest of accumulated alerts"""
        if not self.digest_mode or not self.digest_alerts:
            return False
            
        try:
            # Group alerts by ticker
            ticker_groups = {}
            for alert in self.digest_alerts:
                ticker = alert['ticker']
                if ticker not in ticker_groups:
                    ticker_groups[ticker] = []
                ticker_groups[ticker].append(alert)
                
            # Format digest message
            message = "*FlowPulse Sentinel - Hourly Digest*\n\n"
            
            for ticker, alerts in ticker_groups.items():
                message += f"*{self._escape_markdown_v2(ticker)}* - {len(alerts)} alerts\n"
                
                # Add the highest priority alert details
                highest_priority = max(alerts, key=lambda x: self._calculate_priority(
                    x['ticker'], x['flow'], x['tft_pred']
                ))
                
                flow = highest_priority['flow']
                tft_pred = highest_priority['tft_pred']
                
                volume = flow.get("volume", 0)
                open_interest = flow.get("open_interest", 1)
                
                message += f"  📈 TFT: {self._escape_markdown_v2(f'{tft_pred:.2%}')}\\n"  
                message += f"  📊 Top Flow: {self._escape_markdown_v2(f'{volume}/{open_interest}')} ratio\\n\\n"
                
            # Send digest
            if self.chat_id:
                success = self._send_telegram_message(message)
                if not success:
                    logger.error("Failed to send digest message")
                    return False
                
            # Clear digest alerts
            self.digest_alerts = []
            
            logger.info("Digest alert sent")
            return True
        except Exception as e:
            logger.error(f"Failed to send digest: {e}")
            return False
            
    def set_digest_mode(self, enabled: bool):
        """Enable or disable digest mode"""
        self.digest_mode = enabled
        logger.info(f"Digest mode {'enabled' if enabled else 'disabled'}")
        
    def set_chat_id(self, chat_id: str):
        """Set the chat ID for sending alerts"""
        self.chat_id = chat_id
        logger.info(f"Chat ID set to {chat_id}")
        
    def send_error_notification(self, error_message: str) -> bool:
        """Step 37: Send error notification"""
        if self.bot is None or not self.chat_id:
            return False
            
        try:
            message = f"*ERROR ALERT*\n\n{self._escape_markdown_v2(error_message)}"
            
            success = self._send_telegram_message(message)
            if not success:
                logger.error("Failed to send error notification message")
                return False
            
            return True
        except Exception as e:
            logger.error(f"Failed to send error notification: {e}")
            return False

    def send_test_message(self, message: str) -> bool:
        """Send a test message via Telegram"""
        if not self.chat_id:
            logger.error("No chat ID configured for Telegram")
            return False
            
        try:
            # Escape special characters for MarkdownV2
            message_esc = self._escape_markdown_v2(message)
            
            # Add formatting
            formatted_message = f"*Test Message*\n\n{message_esc}"
            
            # Send message
            return self._send_telegram_message(formatted_message)
        except Exception as e:
            logger.error(f"Failed to send test message: {e}")
            return False
