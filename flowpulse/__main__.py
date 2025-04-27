"""
Main entry point for FlowPulse Sentinel
"""
import argparse
import logging
from flowpulse.bot import FlowPulseBot

def main():
    """Main entry point"""
    # Set up argument parser
    parser = argparse.ArgumentParser(description="FlowPulse Sentinel - AI Trading Bot")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--dry-run", action="store_true", help="Run in dry-run mode (no alerts sent)")
    parser.add_argument("--tickers", type=str, help="Comma-separated list of tickers to monitor")
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Initialize and run bot
    bot = FlowPulseBot()
    
    # Override default tickers if specified
    if args.tickers:
        bot.config.DEFAULT_TICKERS = args.tickers.split(",")
    
    # Run the bot
    bot.run()

if __name__ == "__main__":
    main()
