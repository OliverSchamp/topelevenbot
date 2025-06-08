# Top Eleven Bot

An automated bot for Top Eleven game that handles auctions, training, and ad watching.

## Project Structure
```
OOPBot/
├── config/
│   ├── __init__.py
│   ├── auction_config.py     # Auction-specific configuration
│   ├── training_config.py    # Training-specific configuration
│   └── ad_config.py         # Ad watching configuration
├── utils/
│   ├── __init__.py
│   ├── image_processing.py  # Image processing utilities
│   ├── ocr.py              # OCR related functions
│   └── logging_utils.py     # Enhanced logging functionality
├── core/
│   ├── __init__.py
│   ├── bot.py              # Main bot class
│   ├── auction.py          # Auction functionality
│   ├── training.py         # Training functionality
│   └── ad_watch.py         # Ad watching functionality
└── main.py                 # Entry point
```

## Features
- Automatic player auctions with configurable budgets
- Automatic training management
- Automatic ad watching
- Enhanced logging and error handling
- Configurable parameters
- Image recognition and OCR capabilities

## Usage
1. Configure the settings in the config files
2. Run main.py to start the bot
3. Select desired automation mode 