# Discord Moderation Bot with Gemini AI Integration

This Discord bot includes automatic detection of suicide and self-harm content using Google's Gemini AI.

## Setup

1. Install required packages:
   ```
   pip install -r requirements.txt
   ```

2. Configure tokens:
   - Edit `tokens.json` and add your Discord token
   ```json
   {
     "discord": "YOUR_DISCORD_TOKEN_HERE",
   }
   ```
3. Follow this guide for Gemini API Setup: https://cloud.google.com/vertex-ai/generative-ai/docs/start/quickstarts/quickstart-multimodal#gen-ai-sdk-for-python

4. Run the bot:
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