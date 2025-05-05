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

from report import HarmfulSubcategory

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
    LEVEL_0 = "0: Bot actively discouraged suicide/self-harm and sent the correct resources"
    LEVEL_1 = "1: Bot actively discouraged suicide/self-harm but didn't use the correct phrasing"
    LEVEL_2 = "2: Bot vaguely/indirectly discouraged suicide/self-harm"
    LEVEL_3 = "3: Bot was neutral/didn't address the problem"
    LEVEL_4 = "4: Bot indirectly encouraged suicide/self-harm"
    LEVEL_5 = "5: Bot actively/directly encouraged suicide/self-harm"

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

class ModeratorReportState:
    def __init__(self, report, reported_message, mod_channel):
        self.report = report  # The Report object
        self.reported_message = reported_message  # The reported Discord message
        self.mod_channel = mod_channel  # The mod channel to send messages to
        self.categorization_decision = None  # Is report correctly categorized (Yes, No harm found, Miscategorized)
        self.imminent_danger = None  # Is there imminent danger to user (Yes, No)
        self.gravity_level = None  # Gravity level (0-5)
        self.active = True  # Whether this report is still being processed

class ModBot(discord.Client):
    def __init__(self): 
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix='.', intents=intents)
        self.group_num = None
        self.mod_channels = {} # Map from guild to the mod channel id for that guild
        self.reports = {} # Map from user IDs to the state of their report

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

        # Let the report class handle this message; forward all the messages it returns to us
        responses = await self.reports[author_id].handle_message(message)
        for r in responses:
            await message.channel.send(r)

        report = self.reports[author_id]
        # If the report is complete or cancelled, remove it from our map
        if report.report_complete():
            # # Check if this is a suicide/self-harm report + send to mods
            if (report.subcategory == HarmfulSubcategory.SUICIDE_SELF_HARM):
                await self.send_report_to_moderators(report)

            self.reports.pop(author_id)

    async def handle_channel_message(self, message):
        # Only handle messages sent in the "group-#" channel
        if not message.channel.name == f'group-{self.group_num}':
            return

        # Forward the message to the mod channel
        mod_channel = self.mod_channels[message.guild.id]
        await mod_channel.send(f'Forwarded message:\n{message.author.name}: "{message.content}"')
        scores = self.eval_text(message.content)
        await mod_channel.send(self.code_format(scores))

    async def send_report_to_moderators(self, report):
        """Send a report to the moderator channel and start the human moderator flow.
        Only sends reports that involve suicide/self-harm."""
        
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
        
        # Send initial report information
        await mod_channel.send(f"🚨 **NEW SUICIDE/SELF-HARM REPORT** 🚨 (ID: {report_id})")
        await mod_channel.send(f"**Reported Message:** \n```{report.message.author.name}: {report.message.content}```")
        await mod_channel.send(f"**Category:** {report.category.value if report.category else 'None'}")
        await mod_channel.send(f"**Subcategory:** {report.subcategory.value if report.subcategory else 'None'}")
        
        if report.specific_type:
            await mod_channel.send(f"**Type:** {report.specific_type.value}")
        if report.follow_up:
            await mod_channel.send(f"**Follow-up:** {report.follow_up.value}")
        if report.explanation:
            await mod_channel.send(f"**User Explanation:** {report.explanation}")
        
        # Start the moderator review flow
        await self.start_moderator_review(report_id)
        
        return report_id

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
            await self.complete_moderator_report(report_id)

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
                user_id = mod_report.report.message.author.id
                character_id = mod_report.reported_message.author.id
                
                # Add character to blocked list
                if not hasattr(self, 'blocked_characters'):
                    self.blocked_characters = {}
                
                if user_id not in self.blocked_characters:
                    self.blocked_characters[user_id] = set()
                    
                self.blocked_characters[user_id].add(character_id)
                
                await mod_report.mod_channel.send(f"⛔ **Character blocked** for user ID: {user_id}")
            
            await self.complete_moderator_report(report_id)
            
        elif decision == "NO":
            # Check gravity level for suicide/self-harm reports
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
        
        # Get the index of the gravity level
        gravity_index = list(GravityLevel).index(gravity_level)
        
        if gravity_index <= 1:
            # Level 0-1: No action needed
            await mod_report.mod_channel.send("✅ **No action needed** - Bot handled the situation appropriately.")
            await self.complete_moderator_report(report_id)
            
        elif gravity_index == 2:
            # Level 2: Send character for review
            await mod_report.mod_channel.send("⚠️ **Character sent for review** - The character will be reviewed by the character team.")
            # Here you would implement logic to flag the character for review
            await self.complete_moderator_report(report_id)
            
        elif gravity_index == 3:
            # Level 3: Send character for review
            await mod_report.mod_channel.send("⚠️ **Character sent for review** - The character will be reviewed by the character team.")
            # Here you would implement logic to flag the character for review
            await self.complete_moderator_report(report_id)
            
        elif gravity_index >= 4:
            # Level 4-5: Block character for all users
            if mod_report.reported_message and mod_report.reported_message.author:
                character_id = mod_report.reported_message.author.id
                
                # Add character to global blocked list
                if not hasattr(self, 'globally_blocked_characters'):
                    self.globally_blocked_characters = set()
                    
                self.globally_blocked_characters.add(character_id)
                
                await mod_report.mod_channel.send(f"⛔ **Character blocked for all users** - Character ID: {character_id}")
            
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
        summary += f"- Categorization: {mod_report.decision.value if mod_report.decision else 'N/A'}\n"
        
        if mod_report.recategorization:
            summary += f"- Recategorized as: {mod_report.recategorization.value}\n"
        
        if mod_report.imminent_danger:
            summary += f"- Imminent danger: {mod_report.imminent_danger.value}\n"
        
        if mod_report.gravity_level:
            summary += f"- Gravity level: {mod_report.gravity_level.value}\n"
        
        await mod_report.mod_channel.send(summary)


    # Modify the bot's on_reaction_add event handler to handle moderator reactions
    async def on_reaction_add(self, reaction, user):
        """This function is called whenever a reaction is added to a message."""
        # Ignore reactions from the bot itself
        if user.id == self.user.id:
            return
            
        # Handle moderator reactions
        await self.handle_moderator_reaction(reaction, user)
    
    def eval_text(self, message):
        ''''
        TODO: Once you know how you want to evaluate messages in your channel, 
        insert your code here! This will primarily be used in Milestone 3. 
        '''
        return message

    
    def code_format(self, text):
        ''''
        TODO: Once you know how you want to show that a message has been 
        evaluated, insert your code here for formatting the string to be 
        shown in the mod channel. 
        '''
        return "Evaluated: '" + text+ "'"


client = ModBot()
client.run(discord_token)