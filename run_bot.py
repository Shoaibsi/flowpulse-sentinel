"""
Main entry point for FlowPulse Sentinel
"""
import sys
import os
import logging
import asyncio
from flowpulse.bot import FlowPulseBot

async def main():
    """Asynchronous main function to initialize and run the bot."""
    logging.info("Starting FlowPulse Sentinel (Async Main)...")
    bot = FlowPulseBot() # Instantiate the bot
    
    try:
        # Asynchronously initialize the data fetcher
        logging.info("Initializing data fetcher...")
        fetcher = await bot._init_data_fetcher() 
        bot.data_fetcher = fetcher # Assign the initialized fetcher
        logging.info("Data fetcher initialized successfully.")
        
        # --- Add initializations for other async components if needed ---
        # Example: await bot._init_other_async_component()

        # Now that async setup is done, run the main synchronous loop
        logging.info("Starting bot's main run loop...")
        await bot.run() # Await the async run method
        logging.info("Bot run loop finished.")
        
    except RuntimeError as e:
        logging.critical(f"Failed to initialize critical components: {e}")
        # Exit if critical components failed (like data fetcher)
    except Exception as e:
        logging.critical(f"An unexpected error occurred during bot execution: {e}", exc_info=True)

if __name__ == "__main__":
    # Add project root to path
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s", 
        handlers=[
            logging.FileHandler("flowpulse.log"),
            logging.StreamHandler()
        ]
    )
    
    # Run the async main function
    asyncio.run(main())
