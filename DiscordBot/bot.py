# bot.py
import discord
from discord.ext import commands
import os
import json
import logging
import re
import requests
from report import Report
import pdb
from datetime import datetime
from enum import Enum
from google import genai
# import google.generativeai as genai
from google.genai.types import HttpOptions, Part
# from google.generativeai import types
# import vertexai
# from vertexai.generative_models import generative_models
import textwrap
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


from dotenv import load_dotenv
load_dotenv()
# project = os.environ.get("GOOGLE_CLOUD_PROJECT")
# location = os.environ.get("GOOGLE_CLOUD_LOCATION")
# use_vertexai = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").lower() == "true"  # Better boolean conversion

from report import HarmfulSubcategory
from report import SuicideFollow
from report import State
from report import ReportCategory
from report import SuicideSelfHarmType

# CONSTANTS
GEMINI_MODEL_VER = 'gemini-2.0-flash-001'
# GEMINI_MODEL_VER = "gemini-1.5-pro-preview"

# Set up logging to the console
logger = logging.getLogger('discord')
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler(filename='discord.log', encoding='utf-8', mode='w')
handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s:%(name)s: %(message)s'))
logger.addHandler(handler)

# There should be a file called 'tokens.json' inside the same folder as this file
token_path = 'tokens.json'
if not os.path.isfile(token_path):
    raise Exception(f"{token_path} not found!")
with open(token_path) as f:
    # If you get an error here, it means your token is formatted incorrectly. Did you put it in quotes?
    tokens = json.load(f)
    discord_token = tokens['discord']
    
class GravityLevel(Enum):
    LEVEL_0 = "Bot actively discouraged suicide/self-harm and sent the correct resources"
    LEVEL_1 = "Bot actively discouraged suicide/self-harm but didn't use the correct phrasing"
    LEVEL_2 = "Bot vaguely/indirectly discouraged suicide/self-harm"
    LEVEL_3 = "Bot was neutral/didn't address the problem"
    LEVEL_4 = "Bot indirectly encouraged suicide/self-harm"
    LEVEL_5 = "Bot actively/directly encouraged suicide/self-harm"

class RecategorizationOption(Enum):
    SEXUAL_CONTENT = "Sexual Content"
    HARASSMENT = "Harassment"
    VIOLENCE = "Violence"
    HATE_SPEECH = "Hate Speech"
    INVASION_PRIVACY = "Invasion of Privacy"
    IP_VIOLATION = "IP Violation"
    ILLEGAL_ACTIVITIES = "Illegal Activities"
    FRAUD_SPAM = "Fraud and Scam/Spam"
    MISINFORMATION = "Misinformation/Erroneous Content"

class FollowUpMessage(Enum):
    ENCOURAGED = "Bot encouraged suicide/self-harm"
    FAILED_DISCOURAGE = "Bot failed to discourage suicide/self-harm"

class BotScriptedResponse(Enum):
    EXPECTED_BOT_RESPONSE_1 = "National Suicide Prevention Lifeline: 988 or 1-800-273-8255"
    EXPECTED_BOT_RESPONSE_2 = "Crisis Text Line: Text HOME to 741741"
    LEVELS_1_2 = textwrap.dedent("""\
    I understand our previous conversation touched on some difficult feelings and thoughts you're experiencing. It's important to recognize how challenging these emotions can be. Many people go through periods of intense emotional pain, and it can sometimes be hard to see a way forward. You're not alone in experiencing these struggles. If you or someone you know is experiencing thoughts of self-harm, these resources can help:
    - National Suicide Prevention Lifeline: 988 or 1-800-273-8255
    - Crisis Text Line: Text HOME to 741741
    Each step toward support matters, and connecting with the right resources can make a significant difference in how you're feeling.
    """)
    LEVELS_3_4 = textwrap.dedent("""\
    It seems like something has gone wrong. Your feelings and wellbeing are extremely important. 
    If you or someone you know is experiencing thoughts of self-harm, these resources can help:
    - National Suicide Prevention Lifeline: 988 or 1-800-273-8255 (24/7)
    - Crisis Text Line: Text HOME to 741741 (24/7)
    """)

class UserScriptedResponse(Enum):
    USER_SCRIPTED_REPONSE_0 = textwrap.dedent("""\
    I hear that you're feeling isolated and alone right now. Those feelings are real and can be really challenging to experience. 
    Many people go through periods of feeling disconnected from others, and it can make the world feel like a very lonely place. It takes courage to express these feelings.
    Connecting with others, even in small ways, can make a difference. Consider reaching out to a trusted friend, family member, or counselor who can provide real human connection. Community groups, support networks, or even volunteer opportunities can also help build meaningful relationships. 
    Here are a few ressources for you to consider:
    - National Suicide Prevention Lifeline: 988 or 1-800-273-8255
    - Crisis Text Line: Text HOME to 741741
    You don't have to face these feelings alone. Is there someone in your life you could reach out to today? """)
    USER_SCRIPTED_REPONSE_1 = textwrap.dedent("""\
    I notice we're discussing self-harm, which can bring up many complex emotions and thoughts. Thank you for bringing up this important topic.
    These conversations can be difficult but are important. Many people struggle with these thoughts or know someone who does, and it's normal to have questions or concerns about them. But these feelings, however intense, can change with time and support.
    If you or someone you know is experiencing thoughts of self-harm, these resources can help:
    - National Suicide Prevention Lifeline: 988 or 1-800-273-8255
    - Crisis Text Line: Text HOME to 741741
    Would you like to share more about what prompted this conversation? I'm here to listen and can help you find the most relevant information or support.
    """)
    USER_SCRIPTED_REPONSE_2 = textwrap.dedent("""\
    I understand you're worried about someone you care about who may be at risk of self-harm. It's important to take these thoughts seriously. Listening without judgment is crucial - let them know you care and they're not alone.
    Avoid minimizing their feelings or offering quick solutions.
    Encourage them to seek professional help - this could be their doctor, a therapist, or a crisis service. In urgent situations where you believe someone is in immediate danger, emergency services should be contacted.
    Here are some ways you can help:
    - Express concern gently: "I've noticed you seem to be going through a tough time, and I'm worried about you."
    - Listen without judgment and reassure them that they matter to you
    - Encourage them to reach out to professionals: Crisis Text Line (Text HOME to 741741) or National Suicide Prevention Lifeline (988)
    - In emergencies, help them contact emergency services or get to an emergency room
    Supporting someone with these thoughts can be emotionally challenging. Remember to take care of your own wellbeing too and consider seeking support for yourself.
    Is there anything about this situation that you'd like to discuss?
    """)
    USER_SCRIPTED_REPONSE_3 = textwrap.dedent("""\
    What you're experiencing sounds really difficult. Reaching out for resources is an important and brave step.
    Dealing with these thoughts can feel isolating, but you're not alone in this struggle, and there are people who want to help.
    If you're having thoughts of harming yourself, please reach out to a crisis helpline immediately:
    - National Suicide Prevention Lifeline: 988 or 1-800-273-8255
    - Crisis Text Line: Text HOME to 741741
    You don't have to face these feelings alone. Professional support from a therapist or counselor can make a significant difference. Even just talking with a trusted friend or family member might provide some immediate relief.
    Each step toward support matters, and connecting with the right resources can make a significant difference in how you're feeling.
    """)
    USER_SCRIPTED_REPONSE_4 = textwrap.dedent("""\
    What you're experiencing sounds really difficult. Reaching out for resources is an important and brave step.
    Dealing with these thoughts can feel isolating, but you're not alone in this struggle, and there are people who want to help. These struggles are part of being human, and you shouldn't have to face them alone.
    If you're having thoughts of harming yourself, please reach out to a crisis helpline immediately:
    - National Suicide Prevention Lifeline: 988 or 1-800-273-8255
    - Crisis Text Line: Text HOME to 741741
    You don't have to face these feelings alone. Professional support from a therapist or counselor can make a significant difference. Even just talking with a trusted friend or family member might provide some immediate relief.
    Each step toward support matters, and connecting with the right resources can make a significant difference in how you're feeling.
    """)
    USER_SCRIPTED_REPONSE_5 = textwrap.dedent("""\
    I'm hearing that you're in a lot of pain right now, and I want you to know that I'm taking what you've shared very seriously.
    When we're in this emotional state, it can feel impossible to see a way forward. But these feelings, however intense, can change with time and support.
    If you're having thoughts of harming yourself, please reach out to a crisis helpline immediately:
    - Call 911 or go to your nearest emergency room
    - National Suicide Prevention Lifeline: 988 or 1-800-273-8255
    - Crisis Text Line: Text HOME to 741741
    - If you're not in the US, please call your local emergency services

    Your life matters, and this intense feeling can change with the right support. Please reach out to emergency services now - they're equipped to help you through this critical moment, and taking this step can be the beginning of finding relief.
    """)


class ModeratorReportState:
    def __init__(self, report, reported_message, mod_channel):
        self.report = report  # The Report object
        self.reported_message = reported_message  # The reported Discord message
        self.mod_channel = mod_channel  # The mod channel to send messages to
        self.categorization_decision = None  # Is report correctly categorized (Yes, No harm found, Miscategorized)
        self.imminent_danger = None  # Is there imminent danger to user (Yes, No)
        self.gravity_level = None  # Gravity level (0-5)
        self.active = True  # Whether this report is still being processed
        self.recategorized_as = None # What this report is being recategorized as

class ModBot(discord.Client):
    def __init__(self): 
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix='.', intents=intents)
        self.group_num = None
        self.mod_channels = {} # Map from guild to the mod channel id for that guild
        self.reports = {} # Map from user IDs to the state of their report
        # Initialize Gemini model if API key is available
        # self.gemini_model = genai.Client(http_options=HttpOptions(api_version="v1"))
        self.gemini_model = genai.Client(vertexai = True, project = "cs-152-460122", location="us-central1")
        # self.gemini_model = genai.Client(vertexai = use_vertexai, project = project, location=location)
        # Load the local suicide detection model
        self.load_suicide_detection_model()

    def load_suicide_detection_model(self):
        """Load the local DistilBERT model for suicide content detection"""
        print("in load suicide detection model")
        try:
            model_path = "best_model"
            self.suicide_tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.suicide_model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.suicide_model.eval()  # Set to evaluation mode
            print("Local suicide detection model loaded successfully")
        except Exception as e:
            print(f"Error loading local suicide detection model: {e}")
            self.suicide_tokenizer = None
            self.suicide_model = None

    async def on_ready(self):
        print(f'{self.user.name} has connected to Discord! It is these guilds:')
        for guild in self.guilds:
            print(f' - {guild.name}')
        print('Press Ctrl-C to quit.')

        # Parse the group number out of the bot's name
        match = re.search('[gG]roup (\d+) [bB]ot', self.user.name)
        if match:
            self.group_num = match.group(1)
        else:
            raise Exception("Group number not found in bot's name. Name format should be \"Group # Bot\".")

        # Find the mod channel in each guild that this bot should report to
        for guild in self.guilds:
            for channel in guild.text_channels:
                if channel.name == f'group-{self.group_num}-mod':
                    self.mod_channels[guild.id] = channel
        

    async def on_message(self, message):
        '''
        This function is called whenever a message is sent in a channel that the bot can see (including DMs). 
        Currently the bot is configured to only handle messages that are sent over DMs or in your group's "group-#" channel. 
        '''
        # Ignore messages from the bot 
        if message.author.id == self.user.id:
            return

        # Check if this message was sent in a server ("guild") or if it's a DM
        if message.guild:
            await self.handle_channel_message(message)
        else:
            await self.handle_dm(message)

    async def handle_dm(self, message):
        # Handle a help message
        if message.content == Report.HELP_KEYWORD:
            reply =  "Use the `report` command to begin the reporting process.\n"
            reply += "Use the `cancel` command to cancel the report process.\n"
            await message.channel.send(reply)
            return

        author_id = message.author.id
        responses = []

        # Only respond to messages if they're part of a reporting flow
        if author_id not in self.reports and not message.content.startswith(Report.START_KEYWORD):
            return

        # If we don't currently have an active report for this user, add one
        if author_id not in self.reports:
            self.reports[author_id] = Report(self)
            self.reports[author_id].sent_to_moderators = False

        # Let the report class handle this message; forward all the messages it returns to us
        responses = await self.reports[author_id].handle_message(message)
        for r in responses:
            await message.channel.send(r)


        # Check if this is a suicide/self-harm report in the additional options state
        if author_id in self.reports and self.reports[author_id].state == State.AWAITING_ADDITIONAL_SUICIDE_OPTIONS:
            # Check if this is a suicide/self-harm report + send to mods
            if (self.reports[author_id].subcategory == HarmfulSubcategory.SUICIDE_SELF_HARM and
                not self.reports[author_id].sent_to_moderators):
                self.reports[author_id].user_id = author_id
                self.reports[author_id].sent_to_moderators = True
                await self.send_report_to_moderators(self.reports[author_id])

        # If the report is complete or cancelled, send to moderators and remove from map
        if author_id in self.reports and self.reports[author_id].report_complete():
            # Send all completed reports to moderators (except those already sent)
            if (self.reports[author_id].message and 
                not self.reports[author_id].sent_to_moderators):
                self.reports[author_id].user_id = author_id
                self.reports[author_id].sent_to_moderators = True
                await self.send_report_to_moderators(self.reports[author_id])
            
            self.reports.pop(author_id)

    async def handle_channel_message(self, message):
        # Only handle messages sent in the "group-#" channel
        if not message.channel.name == f'group-{self.group_num}':
            return

        # Forward the message to the mod channel
        # mod_channel = self.mod_channels[message.guild.id]
        # await mod_channel.send(f'Forwarded message:\n{message.author.name}: "{message.content}"')
        
        # Analyze message with Gemini if available
        if self.gemini_model and message.content:
            # user case
            if not any(role.name == "bot" for role in message.author.roles):
                print("role is not bot")
                try:
                    # Check if the message relates to suicide/self-harm
                    is_suicide_related = await self.check_suicide_content(message.content)

                    if is_suicide_related:
                        # categorize user message
                        user_message_category = await self.categorize_user_message(message.content)
                        if user_message_category != None:
                            # send appropriate scripted response
                            await self.send_appropriate_scripted_message(message, user_message_category)
                except Exception as e:
                    logger.error(f"Error analyzing message with Gemini: {e}")

            # character AI/bot case
            else:
                print("role is bot")
                try:
                    # Check if the message relates to suicide/self-harm
                    is_suicide_related = await self.check_suicide_content(message.content)
                    
                    if is_suicide_related:
                        # Create an auto-generated report
                        # await self.create_auto_suicide_report(message)
                        # categorize bot message
                        bot_message_category = await self.categorize_bot_message(message.content)
                        if bot_message_category != None:
                            # take correct action
                            await self.take_appropriate_bot_action(message, bot_message_category)
                except Exception as e:
                    logger.error(f"Error analyzing message with Gemini: {e}")
            
        # Original code for message evaluation
        # scores = self.eval_text(message.content)
        # await mod_channel.send(self.code_format(scores))

    async def check_suicide_content(self, message_text):
        """Use local DistilBERT model to check if message contains suicide/self-harm content"""
        if not self.suicide_model or not self.suicide_tokenizer:
            print("Local suicide detection model not available, falling back to False")
            return False
            
 
        try:
            # Tokenize the input text
            inputs = self.suicide_tokenizer(
                message_text,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt"
            )
            
            # Get model prediction
            with torch.no_grad():
                outputs = self.suicide_model(**inputs)
                logits = outputs.logits
                
                # Apply softmax to get probabilities
                probabilities = torch.softmax(logits, dim=-1)
                
                # Get the predicted class (assuming class 1 is suicide/self-harm)
                predicted_class = torch.argmax(probabilities, dim=-1).item()
                confidence = probabilities[0][predicted_class].item()
                
                print(f"Suicide detection - Predicted class: {predicted_class}, Confidence: {confidence:.4f}")
                
                # Return True if classified as suicide/self-harm content (class 1)
                return predicted_class == 1
                
        except Exception as e:
            print(f"Error in suicide content detection: {e}")
            return False
        
    async def categorize_bot_message(self, message_text):
        if not self.gemini_model:
            return False
            
        # only categorize into 1 through 4 and then after, if category is 1, determine if correct resources were sent
        prompt = f"""
        Categorize this message into categorizes 1 through 4 based on these category descriptions:
        1: the message actively discouraged suicide/self-harm
        2: the message vaguely or indirectly discouraged suicide/self-harm
        3: the message indirectly encouraged suicide/self-harm
        4: the message actively/directly encouraged suicide/self-harm
        Respond with only a number, 1 through 4.
        Message: {message_text}
        """

        response = self.gemini_model.models.generate_content(
            model=GEMINI_MODEL_VER,
            contents=prompt
        )
            
        result = int(response.text.strip())
        print("gemini response: ", result, flush=True)

        # If the bot actively discouraged suicide/self-harm, determine if it sent the correct resources
        # If it did, then set result to 0
        if result == 1:
            if BotScriptedResponse.EXPECTED_BOT_RESPONSE_1.value in message_text and BotScriptedResponse.EXPECTED_BOT_RESPONSE_2.value in message_text:
                result = 0

        return result
    
    async def take_appropriate_bot_action(self, message, bot_message_category):
        if bot_message_category == 0:
            return
        elif bot_message_category == 1 or bot_message_category == 2:
            await message.channel.send(BotScriptedResponse.LEVELS_1_2.value)
        elif bot_message_category == 3 or bot_message_category == 4:
            # delete the message – need certain permissions
            if not message.channel.permissions_for(message.guild.me).manage_messages:
                print(f"Missing manage_messages permission in {message.channel}")
                return
            else:
                # delete the message
                await message.delete()
                await message.channel.send(BotScriptedResponse.LEVELS_3_4.value)

                # block character for this user – just send the blocked message
                await message.channel.send(f"⛔ **Character blocked**")

                # send message to human moderator to determine if character should be blocked for all users
                mod_report = Report(self)
                # I think this is the character AI's id?
                mod_report.user_id = message.author.id
                mod_report.message = message
                mod_report.block_character_for_all_users = True
                report_id = await self.send_report_to_moderators(mod_report)

        # send character for review by character team
        character_id = message.author.id
        mod_channel = self.mod_channels[message.guild.id]
        await mod_channel.send(f"⚠️ **Character with character ID: {character_id} sent for review** - The character will be reviewed by the character team.")
            
            
            
    async def create_auto_suicide_report(self, message):
        """Create an auto-generated report for suicide/self-harm content"""
        # Create a report instance
        auto_report = Report(self)
        auto_report.message = message
        auto_report.category = ReportCategory.HARMFUL
        auto_report.subcategory = HarmfulSubcategory.SUICIDE_SELF_HARM
        auto_report.specific_type = SuicideSelfHarmType.SUICIDE
        auto_report.follow_up = SuicideFollow.OTHER
        auto_report.explanation = "[AUTO-DETECTED] Potential suicide/self-harm content detected by Gemini AI"
        auto_report.state = State.AWAITING_ADDITIONAL_SUICIDE_OPTIONS
        auto_report.user_id = message.author.id  # Store the message author's ID
        
        # Send the report to moderators
        report_id = await self.send_report_to_moderators(auto_report)
        
        if report_id:
            logger.info(f"Auto-generated suicide report sent to moderators: {report_id}")
            mod_channel = self.mod_channels[message.guild.id]
            await mod_channel.send(f"⚠️ **AUTO-DETECTED SUICIDE/SELF-HARM CONTENT** - Report ID: {report_id}")
            await mod_channel.send("This report was automatically generated by Gemini AI. Please verify and take appropriate action.")

    async def categorize_user_message(self, message_text):
        if not self.gemini_model:
            return False
            
        prompt = f"""
        Categorize this message into categorizes 0 through 5 based on these category descriptions:
        0: mentions being alone/lonely/preferring AI to humans
        1: mentions self-harm/suicide in general but not in relation to a person
        2: mentions self-harm suicide directly in relation to someone else
        3: Asks for resources to get help
        4: Non-urgent mention of self-harm or suicide
        5: Direct and urgent mention of self-harm or suicide
        Return only a number 0, 1, 2, 3, 4, or 5.
        Message: {message_text}
        """

        response = self.gemini_model.models.generate_content(
            model=GEMINI_MODEL_VER,
            contents=prompt
        )
        
        result = int(response.text.strip())
        print("gemini response: ", result)
        return result
    
    async def send_appropriate_scripted_message(self, message, user_message_category):
        if user_message_category == 0:
            await message.channel.send(UserScriptedResponse.USER_SCRIPTED_REPONSE_0.value)
        elif user_message_category == 1:
            await message.channel.send(UserScriptedResponse.USER_SCRIPTED_REPONSE_1.value)
        elif user_message_category == 2:
            await message.channel.send(UserScriptedResponse.USER_SCRIPTED_REPONSE_2.value)
        elif user_message_category == 3:
            await message.channel.send(UserScriptedResponse.USER_SCRIPTED_REPONSE_3.value)
        elif user_message_category == 4:
            await message.channel.send(UserScriptedResponse.USER_SCRIPTED_REPONSE_4.value)
        elif user_message_category == 5:
            await message.channel.send(UserScriptedResponse.USER_SCRIPTED_REPONSE_5.value)


    # NOT BEING USED RIGHT NOW
    async def analyze_message_gemini_detailed(self, message_text):
        """Get detailed analysis of potentially harmful content"""
        if not self.gemini_model:
            return None
            
        prompt = f"""
        Analyze the following message for suicide/self-harm content:
        
        Message: {message_text}
        
        If this message contains suicide/self-harm content, respond with a JSON object in the following format:
        {{
            "contains_suicide_content": true,
            "confidence": [number between 0-1],
            "explanation": [brief explanation],
            "gravity_level": [number between 0-5, where 5 is most severe],
            "recommended_action": [recommendation]
        }}
        
        If it does not contain suicide/self-harm content, respond with:
        {{
            "contains_suicide_content": false
        }}
        """
        
        response = self.gemini_model.models.generate_content(
            model=GEMINI_MODEL_VER,
            contents=prompt
        )
        
        try:
            result = json.loads(response.text)
            return result
        except:
            return {"contains_suicide_content": False}

    async def send_report_to_moderators(self, report):
        """Send a report to the moderator channel and start the appropriate moderator flow.
        Suicide/self-harm reports get the full review flow, other reports get simple acknowledgment."""
        
        guild_id = report.message.guild.id
        if guild_id not in self.mod_channels:
            return None
        
        mod_channel = self.mod_channels[guild_id]
        
        # Create a moderator report state
        mod_report = ModeratorReportState(report, report.message, mod_channel)
        
        # Store the report state
        if not hasattr(self, 'moderator_reports'):
            self.moderator_reports = {}
        
        report_id = f"{report.message.id}-{datetime.now().timestamp()}"
        self.moderator_reports[report_id] = mod_report
        
        # Check if this is a "check for blocking character for all users" report
        if report.block_character_for_all_users != None:
            await mod_channel.send(f"🚨 **DETERMINE IF CHARACTER SHOULD BE BLOCKED FOR ALL USERS** 🚨 (ID: {report_id})")
            await mod_channel.send(f"**Message:** \n```{report.message.author.name}: {report.message.content}```")
            await self.block_character_for_all_users_decision(report_id)
        
        # Check if this is a suicide/self-harm report that needs full review
        elif (hasattr(report, 'subcategory') and 
              report.subcategory == HarmfulSubcategory.SUICIDE_SELF_HARM and
              (hasattr(report, 'additional_option') and report.additional_option is not None) or
              (hasattr(report, 'state') and report.state == State.AWAITING_ADDITIONAL_SUICIDE_OPTIONS)):
            # Full suicide/self-harm review flow
            await mod_channel.send(f"🚨 **NEW SUICIDE/SELF-HARM REPORT** 🚨 (ID: {report_id})")
            await mod_channel.send(f"**Reported Message:** \n```{report.message.author.name}: {report.message.content}```")
            await mod_channel.send(f"**Category:** {report.category.value if report.category else 'None'}")
            await mod_channel.send(f"**Subcategory:** {report.subcategory.value if report.subcategory else 'None'}")
            
            if report.specific_type:
                await mod_channel.send(f"**Type:** {report.specific_type.value}")
            if report.follow_up:
                if report.follow_up == SuicideFollow.BOT_ENCOURAGED:
                    await mod_channel.send(f"**Follow-up:** {FollowUpMessage.ENCOURAGED.value}")
                elif report.follow_up == SuicideFollow.BOT_FAILED_DISCOURAGE:
                    await mod_channel.send(f"**Follow-up:** {FollowUpMessage.FAILED_DISCOURAGE.value}")
                else:
                    await mod_channel.send(f"**Follow-up:** {report.follow_up.value}")
            if report.explanation:
                await mod_channel.send(f"**User Explanation:** {report.explanation}")
        
            # Start the full moderator review flow
            await self.start_moderator_review(report_id)
        
        else:
            # Simple acknowledgment flow for all other reports
            await mod_channel.send(f"📋 **NEW REPORT** 📋 (ID: {report_id})")
            await mod_channel.send(f"**Reported Message:** \n```{report.message.author.name}: {report.message.content}```")
            await mod_channel.send(f"**Category:** {report.category.value if report.category else 'None'}")
            
            if hasattr(report, 'subcategory') and report.subcategory:
                await mod_channel.send(f"**Subcategory:** {report.subcategory.value}")
            if hasattr(report, 'specific_type') and report.specific_type:
                await mod_channel.send(f"**Type:** {report.specific_type.value}")
            if hasattr(report, 'explanation') and report.explanation:
                await mod_channel.send(f"**User Explanation:** {report.explanation}")
            
            # Simple acknowledgment flow
            await self.start_simple_acknowledgment(report_id)
        
        return report_id
    
    async def block_character_for_all_users_decision(self, report_id):
        mod_report = self.moderator_reports[report_id]
        await mod_report.mod_channel.send("Should this character be blocked for all users? Reply with:")
        options = "1. Yes, block character for all users\n2. No, do not block character for all users"
        
        message = await mod_report.mod_channel.send(options)
        
        # Add reaction options for moderator decision
        for i in range(1, 3):  # 3 options
            await message.add_reaction(f"{i}\u20e3")  # Adding keycap emoji (1️⃣, 2️⃣, 3️⃣)
        
        # Store this message ID for reaction handling
        if not hasattr(self, 'decision_messages'):
            self.decision_messages = {}
        self.decision_messages[message.id] = {"report_id": report_id, "type": "block character"}
        

    async def start_moderator_review(self, report_id):
        """Start the human moderator review process."""
        if report_id not in self.moderator_reports:
            return
        
        mod_report = self.moderator_reports[report_id]
        
        # Send the first question - is the report categorized correctly?
        await mod_report.mod_channel.send(f"**MODERATOR REVIEW** (Report ID: {report_id})")
        await mod_report.mod_channel.send("Is this report categorized correctly? Reply with:")
        options = "1. Correctly Categorized\n2. No Harm Found\n3. Miscategorized"
        
        message = await mod_report.mod_channel.send(options)
        
        # Add reaction options for moderator decision
        for i in range(1, 4):  # 3 options
            await message.add_reaction(f"{i}\u20e3")  # Adding keycap emoji (1️⃣, 2️⃣, 3️⃣)
        
        # Store this message ID for reaction handling
        if not hasattr(self, 'decision_messages'):
            self.decision_messages = {}
        self.decision_messages[message.id] = {"report_id": report_id, "type": "categorization"}

    async def handle_moderator_reaction(self, reaction, user):
        """Handle reactions from moderators for the review process."""
        # Ignore bot's own reactions
        if user.id == self.user.id:
            return
        
        # Check if this is a reaction to a decision message
        if not hasattr(self, 'decision_messages') or reaction.message.id not in self.decision_messages:
            return
        
        decision_info = self.decision_messages[reaction.message.id]
        report_id = decision_info["report_id"]
        decision_type = decision_info["type"]
        
        if report_id not in self.moderator_reports:
            return
        
        mod_report = self.moderator_reports[report_id]
        
        # Process reaction based on decision type
        if decision_type == "categorization":
            # Convert emoji to number (1️⃣ -> 1)
            try:
                emoji_text = str(reaction.emoji)
                if emoji_text[0].isdigit():
                    choice = int(emoji_text[0])
                    
                    # Map choices to decision values
                    if choice == 1:
                        mod_report.categorization_decision = "CORRECTLY_CATEGORIZED"
                    elif choice == 2:
                        mod_report.categorization_decision = "NO_HARM_FOUND"
                    elif choice == 3:
                        mod_report.categorization_decision = "MISCATEGORIZED"
                    else:
                        return  # Invalid choice
                    
                    # Remove this message from the tracking dict
                    self.decision_messages.pop(reaction.message.id)
                    
                    # Handle next steps based on decision
                    await self.handle_categorization_decision(report_id, mod_report.categorization_decision)
            except:
                pass  # Invalid reaction, ignore
                
        elif decision_type == "imminent_danger":
            try:
                emoji_text = str(reaction.emoji)
                if emoji_text[0].isdigit():
                    choice = int(emoji_text[0])
                    
                    if choice == 1:
                        mod_report.imminent_danger = "YES"
                    elif choice == 2:
                        mod_report.imminent_danger = "NO"
                    else:
                        return  # Invalid choice
                    
                    # Remove this message from the tracking dict
                    self.decision_messages.pop(reaction.message.id)
                    
                    # Handle next steps based on imminent danger decision
                    await self.handle_imminent_danger_decision(report_id, mod_report.imminent_danger)
            except:
                pass
                
        elif decision_type == "gravity":
            try:
                emoji_text = str(reaction.emoji)
                if emoji_text[0].isdigit():
                    choice = int(emoji_text[0])  # Direct mapping for gravity level (0-5)
                    
                    if 0 <= choice <= 5:  # We have 6 levels (0-5)
                        mod_report.gravity_level = choice
                        
                        # Remove this message from the tracking dict
                        self.decision_messages.pop(reaction.message.id)
                        
                        # Handle next steps based on gravity level
                        await self.handle_gravity_level_decision(report_id, mod_report.gravity_level)
            except:
                pass
            
        elif decision_type == "recategorization":
            try:
                emoji_text = str(reaction.emoji)
                if emoji_text[0].isdigit():
                    choice = int(emoji_text[0])  # Direct mapping for recategorization
                    
                    if 1 <= choice <= 9:  # We have 9 levels (1-9)
                        mod_report.recategorized_as = choice
                        
                        # Remove this message from the tracking dict
                        self.decision_messages.pop(reaction.message.id)

                        await self.complete_moderator_report(report_id)
                        # Handle next steps based on gravity level
                        # await self.handle_recategorized_decision(report_id, mod_report.gravity_level)
            except:
                pass
        elif decision_type == "block character":
            try:
                emoji_text = str(reaction.emoji)
                if emoji_text[0].isdigit():
                    choice = int(emoji_text[0])
                # block the character for all users

                # Remove this message from the tracking dict
                self.decision_messages.pop(reaction.message.id)

                if choice == 1:
                    character_id = mod_report.reported_message.author.id
                    # Add character to global blocked list
                    if not hasattr(self, 'globally_blocked_characters'):
                        self.globally_blocked_characters = set()
                        
                    self.globally_blocked_characters.add(character_id)
                    
                    await mod_report.mod_channel.send(f"⛔ **Character blocked for all users** - Character ID: {character_id}")

                    await self.complete_moderator_report(report_id)
                # we don't need to do anything in the case that the moderator decides not to block the character
                else:
                    await self.complete_moderator_report(report_id)
            except:
                pass

        elif decision_type == "simple_acknowledgment":
            try:
                emoji_text = str(reaction.emoji)
                if emoji_text[0].isdigit():
                    choice = int(emoji_text[0])
                    
                    # Remove this message from the tracking dict
                    self.decision_messages.pop(reaction.message.id)
                    
                    if choice == 1:
                        await mod_report.mod_channel.send(f"✅ **Report acknowledged** (ID: {report_id}) - No action needed.")
                    elif choice == 2:
                        await mod_report.mod_channel.send(f"⚠️ **Report acknowledged** (ID: {report_id}) - Flagged for further review.")
                    
                    # Complete the report
                    await self.complete_moderator_report(report_id)
            except:
                pass

    async def handle_categorization_decision(self, report_id, decision):
        """Handle the next steps based on categorization decision."""
        if report_id not in self.moderator_reports:
            return
            
        mod_report = self.moderator_reports[report_id]
        
        if decision == "CORRECTLY_CATEGORIZED":
            # For suicide/self-harm reports, check for imminent danger
            if (mod_report.report.subcategory == HarmfulSubcategory.SUICIDE_SELF_HARM):
                await self.check_imminent_danger(report_id)
            else:
                # For other correctly categorized reports, complete the review
                await self.complete_moderator_report(report_id)
                
        elif decision == "NO_HARM_FOUND":
            await mod_report.mod_channel.send("✅ **No harm found** - No action will be taken.")
            await self.complete_moderator_report(report_id)
            
        elif decision == "MISCATEGORIZED":
            await self.recategorize_report(report_id)
            # await self.complete_moderator_report(report_id)

    async def check_imminent_danger(self, report_id):
        """Check if there is imminent danger to the user."""
        if report_id not in self.moderator_reports:
            return
            
        mod_report = self.moderator_reports[report_id]
        
        await mod_report.mod_channel.send("**Is there an imminent danger to the user?** Reply with:")
        options = "1. Yes\n2. No"
        
        message = await mod_report.mod_channel.send(options)
        
        # Add reaction options
        await message.add_reaction(f"{1}\u20e3")
        await message.add_reaction(f"{2}\u20e3")
        
        # Store this message ID for reaction handling
        self.decision_messages[message.id] = {"report_id": report_id, "type": "imminent_danger"}

    async def handle_imminent_danger_decision(self, report_id, decision):
        """Handle the next steps based on imminent danger decision."""
        if report_id not in self.moderator_reports:
            return
            
        mod_report = self.moderator_reports[report_id]
        
        if decision == "YES":
            # Block character for this user
            if mod_report.reported_message and mod_report.reported_message.author:
                user_id = mod_report.report.user_id
                character_id = mod_report.reported_message.author.id
                
                # Add character to blocked list
                if not hasattr(self, 'blocked_characters'):
                    self.blocked_characters = {}
                
                if user_id not in self.blocked_characters:
                    self.blocked_characters[user_id] = set()
                    
                self.blocked_characters[user_id].add(character_id)
                
                await mod_report.mod_channel.send(f"⛔ **Character blocked** for user ID: {user_id}")
                user = await mod_report.report.client.fetch_user(user_id)
                dm_channel = await user.create_dm()
                await dm_channel.send(f"⛔ **Character blocked** - You are blocked from interacting with this character (ID: {character_id}).")
            
        await self.check_gravity_level(report_id)

    async def check_gravity_level(self, report_id):
        """Check the gravity level of the suicide/self-harm content."""
        if report_id not in self.moderator_reports:
            return
            
        mod_report = self.moderator_reports[report_id]
        
        await mod_report.mod_channel.send("**Check gravity level:** Select the appropriate level:")
        options = "\n".join([f"{i}: {level.value}" for i, level in enumerate(GravityLevel)])
        
        message = await mod_report.mod_channel.send(options)
        
        # Add reaction options
        for i in range(len(GravityLevel)):
            await message.add_reaction(f"{i}\u20e3")
        
        # Store this message ID for reaction handling
        self.decision_messages[message.id] = {"report_id": report_id, "type": "gravity"}

    async def handle_gravity_level_decision(self, report_id, gravity_level):
        """Handle the next steps based on gravity level decision."""
        if report_id not in self.moderator_reports:
            return
            
        mod_report = self.moderator_reports[report_id]
        
        if gravity_level == 0:
            # Level 0-1: No action needed
            await mod_report.mod_channel.send("✅ **No action needed** - Bot handled the situation appropriately.")
            await self.complete_moderator_report(report_id)
            
        elif gravity_level == 1 or gravity_level == 2 or gravity_level == 3:
            # Level 2: Send character for review
            await mod_report.mod_channel.send("⚠️ **Character sent for review** - The character will be reviewed by the character team.")
            # Here you would implement logic to flag the character for review
            await self.complete_moderator_report(report_id)
            
        elif gravity_level >= 4:
            # Level 4-5: Block character for all users
            if mod_report.reported_message and mod_report.reported_message.author:
                character_id = mod_report.reported_message.author.id
                
                # Add character to global blocked list
                if not hasattr(self, 'globally_blocked_characters'):
                    self.globally_blocked_characters = set()
                    
                self.globally_blocked_characters.add(character_id)
                
                await mod_report.mod_channel.send(f"⛔ **Character blocked for all users** - Character ID: {character_id}")
                await mod_report.mod_channel.send("⚠️ **Character sent for review** - The character will be reviewed by the character team.")

            await self.complete_moderator_report(report_id)

    async def recategorize_report(self, report_id):
        """Allow moderator to recategorize the report."""
        if report_id not in self.moderator_reports:
            return
            
        mod_report = self.moderator_reports[report_id]
        
        await mod_report.mod_channel.send("**Select the correct category:**")
        options = "\n".join([f"{i+1}. {category.value}" for i, category in enumerate(RecategorizationOption)])
        
        message = await mod_report.mod_channel.send(options)
        
        # Add reaction options
        for i in range(1, len(RecategorizationOption) + 1):
            await message.add_reaction(f"{i}\u20e3")
        
        # Store this message ID for reaction handling
        self.decision_messages[message.id] = {"report_id": report_id, "type": "recategorization"}

    async def complete_moderator_report(self, report_id):
        """Complete the moderator review process."""
        if report_id not in self.moderator_reports:
            return
            
        mod_report = self.moderator_reports[report_id]
        
        # Mark the report as inactive
        mod_report.active = False
        
        await mod_report.mod_channel.send(f"✅ **Report review completed** (ID: {report_id})")
        
        # Summarize the decisions made
        summary = "**Review Summary:**\n"
        summary += f"- Categorization: {mod_report.categorization_decision if mod_report.categorization_decision else 'N/A'}\n"
        
        if mod_report.recategorized_as:
            summary += f"- Recategorized as: {mod_report.recategorized_as}\n"
        
        if mod_report.imminent_danger:
            summary += f"- Imminent danger: {mod_report.imminent_danger}\n"
        
        if mod_report.gravity_level is not None:
            summary += f"- Gravity level: {mod_report.gravity_level}\n"
        
        await mod_report.mod_channel.send(summary)

    async def start_simple_acknowledgment(self, report_id):
        """Start a simple acknowledgment process for non-suicide/self-harm reports."""
        if report_id not in self.moderator_reports:
            return
        
        mod_report = self.moderator_reports[report_id]
        
        # Send simple acknowledgment options
        await mod_report.mod_channel.send(f"**MODERATOR ACKNOWLEDGMENT** (Report ID: {report_id})")
        await mod_report.mod_channel.send("Please acknowledge this report:")
        options = "1. Acknowledged - No Action Needed\n2. Acknowledged - Further Review Required"
        
        message = await mod_report.mod_channel.send(options)
        
        # Add reaction options for moderator acknowledgment
        for i in range(1, 3):  # 2 options
            await message.add_reaction(f"{i}\u20e3")  # Adding keycap emoji (1️⃣, 2️⃣)
        
        # Store this message ID for reaction handling
        if not hasattr(self, 'decision_messages'):
            self.decision_messages = {}
        self.decision_messages[message.id] = {"report_id": report_id, "type": "simple_acknowledgment"}

    # Modify the bot's on_reaction_add event handler to handle moderator reactions
    async def on_reaction_add(self, reaction, user):
        """This function is called whenever a reaction is added to a message."""
        # Ignore reactions from the bot itself
        if user.id == self.user.id:
            return
            
        # Handle moderator reactions
        await self.handle_moderator_reaction(reaction, user)
    
    # def eval_text(self, message):
    #     ''''
    #     TODO: Once you know how you want to evaluate messages in your channel, 
    #     insert your code here! This will primarily be used in Milestone 3. 
    #     '''
    #     return message

    
    # def code_format(self, text):
    #     ''''
    #     TODO: Once you know how you want to show that a message has been 
    #     evaluated, insert your code here for formatting the string to be 
    #     shown in the mod channel. 
    #     '''
    #     return "Evaluated: '" + text+ "'"


client = ModBot()
client.run(discord_token)