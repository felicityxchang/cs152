# Discord Moderation Bot with Gemini AI Integration

This Discord bot includes automatic detection of suicide and self-harm content using Google's Gemini AI.

## Setup

1. Install required packages:
   ```
   pip install -r requirements.txt
   ```

2. Configure tokens:
   - Edit `tokens.json` and add your Discord token and Gemini API key
   ```json
   {
     "discord": "YOUR_DISCORD_TOKEN_HERE",
     "gemini": "YOUR_GEMINI_API_KEY_HERE"
   }
   ```

3. Run the bot:
   ```
   python bot.py
   ```

## Features

- Manual user reporting system for harmful content
- Automatic detection of suicide/self-harm content using Gemini AI
- Human moderator review workflow
- Automatic report generation for detected harmful content

## Suicide/Self-Harm Detection

The bot automatically:
1. Analyzes all messages in the server using Gemini AI
2. Detects suicide and self-harm related content
3. Creates pre-filled moderator reports for review
4. Sends reports to the mod channel for human verification

## Getting a Gemini API Key

1. Go to the [Google AI Studio](https://ai.google.dev/) and sign in
2. Navigate to the API keys section
3. Create a new API key
4. Add the key to your `tokens.json` file 