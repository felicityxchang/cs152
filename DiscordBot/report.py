from enum import Enum, auto
import discord
import re

class ReportCategory(Enum):
    HARMFUL = "Harmful or Unsafe Content"
    ILLEGAL = "Illegal or False Content"

class HarmfulSubcategory(Enum):
    SEXUAL_CONTENT = "Sexual Content"
    HARASSMENT = "Harassment"
    VIOLENCE = "Violence"
    HATE_SPEECH = "Hate Speech"
    SUICIDE_SELF_HARM = "Suicide and Self-Harm"
    INVASION_PRIVACY = "Invasion of Privacy"

class SexualContentType(Enum):
    CHILD_EXPLOITATION = "Child Exploitation"
    SEXUAL_HARASSMENT = "Sexual Harassment"
    OTHER = "Other Sexual Content"

class SuicideSelfHarmType(Enum):
    SUICIDE = "Suicide or Self-Harm"
    EATING_DISORDER = "Eating Disorder"

class SuicideFollow(Enum):
    BOT_ENCOURAGED = "Did the bot encourage suicide/self-harm?"
    BOT_FAILED_DISCOURAGE = "Did the bot fail to discourage suicide/self-harm?"
    OTHER = "Other (please explain)"

class EatingDisorderFollow(Enum):
    BOT_ENCOURAGED = "Did the bot encourage eating disorder behavior?"
    BOT_FAILED_DISCOURAGE = "Did the bot fail to discourage eating disorder behavior?"
    OTHER = "Other (please explain)"

class IllegalSubcategory(Enum):
    ILLEGAL_ACTIVITIES = "Illegal Activities"
    FRAUD_SPAM = "Fraud and Scam/Spam"
    IP_VIOLATION = "IP Violation"
    MISINFORMATION = "Misinformation/Erroneous Content"

class IllegalActivityType(Enum):
    DRUGS = "Drugs"
    ALCOHOL = "Alcohol"
    OTHER = "Other Illegal Activity"

class MisinformationType(Enum):
    IMPERSONATION = "Impersonating a Person"
    OTHER = "Other Misinformation"

class AdditionalSuicideOptions(Enum):
    RESOURCES = "Suicide and self-harm resources"
    BLOCK_CHARACTER = "Block this character (don't show me this character again)"
    DONT_SHOW_TOPIC = "Don't bring up this topic again"
    STOP_HERE = "No further action needed"

class AdditionalEDOptions(Enum):
    RESOURCES = "Eating disorder resources"
    BLOCK_CHARACTER = "Block this character (don't show me this character again)"
    DONT_SHOW_TOPIC = "Don't bring up this topic again"
    STOP_HERE = "No further action needed"

class State(Enum):
    REPORT_START = auto()
    AWAITING_MESSAGE = auto()
    MESSAGE_IDENTIFIED = auto()
    AWAITING_CATEGORY = auto()
    AWAITING_HARMFUL_SUBCATEGORY = auto()
    AWAITING_ILLEGAL_SUBCATEGORY = auto()
    AWAITING_SEXUAL_CONTENT_TYPE = auto()
    AWAITING_SUICIDE_SELF_HARM_TYPE = auto()
    AWAITING_SUICIDE_FOLLOW_UP = auto()
    AWAITING_ED_FOLLOW_UP = auto()
    AWAITING_ILLEGAL_ACTIVITY_TYPE = auto()
    AWAITING_MISINFORMATION_TYPE = auto()
    AWAITING_ADDITIONAL_SUICIDE_OPTIONS = auto()
    AWAITING_ADDITIONAL_ED_OPTIONS = auto()
    AWAITING_EXPLANATION = auto()
    REPORT_COMPLETE = auto()
    REPORT_COMPLETE_SUICIDE = auto()

class Report:
    START_KEYWORD = "report"
    CANCEL_KEYWORD = "cancel"
    HELP_KEYWORD = "help"

    def __init__(self, client):
        self.state = State.REPORT_START
        self.client = client
        self.message = None
        self.category = None
        self.subcategory = None
        self.specific_type = None
        self.follow_up = None
        self.explanation = None
        self.additional_option = None
        self.blocked_characters = set()
        self.blocked_topics = set()
        self.user_id = None
        self.block_character_for_all_users = None # should I have done this??
        
    async def handle_message(self, message):
        '''
        This function makes up the meat of the user-side reporting flow. It defines how we transition between states and what 
        prompts to offer at each of those states.
        '''

        if message.content.lower() == self.CANCEL_KEYWORD:
            self.state = State.REPORT_COMPLETE
            return ["Report cancelled."]
        
        if message.content.lower() == self.HELP_KEYWORD:
            return [self._get_help_message()]
        
        if self.state == State.REPORT_START:
            reply = "Thank you for starting the reporting process. "
            reply += "Say `help` at any time for more information.\n\n"
            reply += "Please copy paste the link to the message you want to report.\n"
            reply += "You can obtain this link by right-clicking the message and clicking `Copy Message Link`."
            self.state = State.AWAITING_MESSAGE
            return [reply]
        
        if self.state == State.AWAITING_MESSAGE:
            # Parse out the three ID strings from the message link
            m = re.search(r'https?://(?:canary\.|ptb\.)?discord(?:app)?\.com/channels/(\d+)/(\d+)/(\d+)', message.content)
            if not m:
                return ["I'm sorry, I couldn't read that link. Please try again or say `cancel` to cancel."]
            guild = self.client.get_guild(int(m.group(1)))
            if not guild:
                return ["I cannot accept reports of messages from guilds that I'm not in. Please have the guild owner add me to the guild and try again."]
            channel = guild.get_channel(int(m.group(2)))
            if not channel:
                return ["It seems this channel was deleted or never existed. Please try again or say `cancel` to cancel."]
            try:
                reported_message = await channel.fetch_message(int(m.group(3)))
            except discord.errors.NotFound:
                return ["It seems this message was deleted or never existed. Please try again or say `cancel` to cancel."]

            self.message = reported_message
            self.state = State.MESSAGE_IDENTIFIED
            return ["I found this message:", "```" + reported_message.author.name + ": " + reported_message.content + "```", 
                   "Please select a category for your report:", self._get_categories()]
        
        if self.state == State.MESSAGE_IDENTIFIED:
            try:
                category_index = int(message.content) - 1
                categories = list(ReportCategory)
                if 0 <= category_index < len(categories):
                    self.category = categories[category_index]
                    self.state = State.AWAITING_CATEGORY
                    return self._handle_category_selection()
                else:
                    return ["Invalid selection. Please enter a number between 1 and " + str(len(categories)) + "."]
            except ValueError:
                return ["Please enter a number to select a category."]
        
        if self.state == State.AWAITING_HARMFUL_SUBCATEGORY:
            try:
                subcategory_index = int(message.content) - 1
                subcategories = list(HarmfulSubcategory)
                if 0 <= subcategory_index < len(subcategories):
                    self.subcategory = subcategories[subcategory_index]
                    return self._handle_harmful_subcategory_selection()
                else:
                    return ["Invalid selection. Please enter a number between 1 and " + str(len(subcategories)) + "."]
            except ValueError:
                return ["Please enter a number to select a subcategory."]
                
        if self.state == State.AWAITING_ILLEGAL_SUBCATEGORY:
            try:
                subcategory_index = int(message.content) - 1
                subcategories = list(IllegalSubcategory)
                if 0 <= subcategory_index < len(subcategories):
                    self.subcategory = subcategories[subcategory_index]
                    return self._handle_illegal_subcategory_selection()
                else:
                    return ["Invalid selection. Please enter a number between 1 and " + str(len(subcategories)) + "."]
            except ValueError:
                return ["Please enter a number to select a subcategory."]
                
        if self.state == State.AWAITING_SEXUAL_CONTENT_TYPE:
            try:
                type_index = int(message.content) - 1
                content_types = list(SexualContentType)
                if 0 <= type_index < len(content_types):
                    self.specific_type = content_types[type_index]
                    self.state = State.REPORT_COMPLETE
                    return ["Report sent. ✅"]
                else:
                    return ["Invalid selection. Please enter a number between 1 and " + str(len(content_types)) + "."]
            except ValueError:
                return ["Please enter a number to select a type."]
                
        if self.state == State.AWAITING_SUICIDE_SELF_HARM_TYPE:
            try:
                type_index = int(message.content) - 1
                content_types = list(SuicideSelfHarmType)
                if 0 <= type_index < len(content_types):
                    self.specific_type = content_types[type_index]
                    if self.specific_type == SuicideSelfHarmType.SUICIDE:
                        self.state = State.AWAITING_SUICIDE_FOLLOW_UP
                        return ["Please specify:", self._get_suicide_follow_up_options()]
                    elif self.specific_type == SuicideSelfHarmType.EATING_DISORDER:
                        self.state = State.AWAITING_ED_FOLLOW_UP
                        return ["Please specify:", self._get_ed_follow_up_options()]
                else:
                    return ["Invalid selection. Please enter a number between 1 and " + str(len(content_types)) + "."]
            except ValueError:
                return ["Please enter a number to select a type."]
                
        if self.state == State.AWAITING_SUICIDE_FOLLOW_UP:
            try:
                option_index = int(message.content) - 1
                options = list(SuicideFollow)
                if 0 <= option_index < len(options):
                    self.follow_up = options[option_index]
                    if self.follow_up == SuicideFollow.OTHER:
                        self.state = State.AWAITING_EXPLANATION
                        return ["Please provide more details about your report:"]
                    else:
                        self.state = State.AWAITING_ADDITIONAL_SUICIDE_OPTIONS
                        return ["Report sent. ✅", 
                                "Thank you for reporting this issue. Is there any other action you would like to take?", 
                                self._get_additional_suicide_options()]
                else:
                    return ["Invalid selection. Please enter a number between 1 and " + str(len(options)) + "."]
            except ValueError:
                return ["Please enter a number to select an option."]
                
        if self.state == State.AWAITING_ED_FOLLOW_UP:
            try:
                option_index = int(message.content) - 1
                options = list(EatingDisorderFollow)
                if 0 <= option_index < len(options):
                    self.follow_up = options[option_index]
                    if self.follow_up == EatingDisorderFollow.OTHER:
                        self.state = State.AWAITING_EXPLANATION
                        return ["Please provide more details about your report:"]
                    else:
                        self.state = State.AWAITING_ADDITIONAL_ED_OPTIONS
                        return ["Report sent. ✅", 
                                "Thank you for reporting this issue. Is there any other action you would like to take?", 
                                self._get_additional_ed_options()]
                else:
                    return ["Invalid selection. Please enter a number between 1 and " + str(len(options)) + "."]
            except ValueError:
                return ["Please enter a number to select an option."]
                
        if self.state == State.AWAITING_ILLEGAL_ACTIVITY_TYPE:
            try:
                type_index = int(message.content) - 1
                content_types = list(IllegalActivityType)
                if 0 <= type_index < len(content_types):
                    self.specific_type = content_types[type_index]
                    self.state = State.REPORT_COMPLETE
                    return ["Report sent. ✅"]
                else:
                    return ["Invalid selection. Please enter a number between 1 and " + str(len(content_types)) + "."]
            except ValueError:
                return ["Please enter a number to select a type."]
                
        if self.state == State.AWAITING_MISINFORMATION_TYPE:
            try:
                type_index = int(message.content) - 1
                content_types = list(MisinformationType)
                if 0 <= type_index < len(content_types):
                    self.specific_type = content_types[type_index]
                    self.state = State.REPORT_COMPLETE
                    return ["Report sent. ✅"]
                else:
                    return ["Invalid selection. Please enter a number between 1 and " + str(len(content_types)) + "."]
            except ValueError:
                return ["Please enter a number to select a type."]
                
        if self.state == State.AWAITING_EXPLANATION:
            self.explanation = message.content
            
            if self.specific_type == SuicideSelfHarmType.SUICIDE:
                self.state = State.AWAITING_ADDITIONAL_SUICIDE_OPTIONS
                return ["Report sent. ✅", 
                        "Thank you for reporting this issue. Is there any other action you would like to take?", 
                        self._get_additional_suicide_options()]
            elif self.specific_type == SuicideSelfHarmType.EATING_DISORDER:
                self.state = State.AWAITING_ADDITIONAL_ED_OPTIONS
                return ["Report sent. ✅", 
                        "Thank you for reporting this issue. Is there any other action you would like to take?", 
                        self._get_additional_ed_options()]
            else:
                self.state = State.REPORT_COMPLETE
                return ["Report sent. ✅"]
                
        if self.state == State.AWAITING_ADDITIONAL_SUICIDE_OPTIONS:
            try:
                option_index = int(message.content) - 1
                options = list(AdditionalSuicideOptions)
                if 0 <= option_index < len(options):
                    self.additional_option = options[option_index]
                    self.state = State.REPORT_COMPLETE
                    if self.additional_option == AdditionalSuicideOptions.RESOURCES:
                        return ["Here are some resources that might help:",
                                "• National Suicide Prevention Lifeline: 1-800-273-8255",
                                "• Crisis Text Line: Text HOME to 741741",
                                "• International Association for Suicide Prevention: https://www.iasp.info/resources/Crisis_Centres/",
                                "Thank you for your report. We will review this message and take appropriate action."]
                    elif self.additional_option == AdditionalSuicideOptions.BLOCK_CHARACTER:
                        # Add character to blocked list
                        if self.message:
                            self.blocked_characters.add(self.message.author.id)
                        return ["⛔ Character blocked. You will not see messages from this character anymore.",
                                "Thank you for your report. We will review this message and take appropriate action."]
                    elif self.additional_option == AdditionalSuicideOptions.DONT_SHOW_TOPIC:
                        # Add topic to blocked list
                        self.blocked_topics.add("suicide")
                        return ["⛔ Topic blocked. This topic will not be shown to you anymore.",
                                "Thank you for your report. We will review this message and take appropriate action."]
                    elif self.additional_option == AdditionalSuicideOptions.STOP_HERE:
                        return ["Thank you for your report. We will review this message and take appropriate action."]
                else:
                    return ["Invalid selection. Please enter a number between 1 and " + str(len(options)) + "."]
            except ValueError:
                return ["Please enter a number to select an option."]
                
        if self.state == State.AWAITING_ADDITIONAL_ED_OPTIONS:
            try:
                option_index = int(message.content) - 1
                options = list(AdditionalEDOptions)
                if 0 <= option_index < len(options):
                    self.additional_option = options[option_index]
                    self.state = State.REPORT_COMPLETE
                    if self.additional_option == AdditionalEDOptions.RESOURCES:
                        return ["Here are some resources that might help:",
                                "• National Eating Disorders Association (NEDA): 1-800-931-2237",
                                "• NEDA Crisis Text Line: Text NEDA to 741741",
                                "• Eating Disorders Anonymous: http://www.eatingdisordersanonymous.org/",
                                "Thank you for your report. We will review this message and take appropriate action."]
                    elif self.additional_option == AdditionalEDOptions.BLOCK_CHARACTER:
                        # Add character to blocked list
                        if self.message:
                            self.blocked_characters.add(self.message.author.id)
                        return ["⛔ Character blocked. You will not see messages from this character anymore.",
                                "Thank you for your report. We will review this message and take appropriate action."]
                    elif self.additional_option == AdditionalEDOptions.DONT_SHOW_TOPIC:
                        # Add topic to blocked list
                        self.blocked_topics.add("eating disorder")
                        return ["⛔ Topic blocked. This topic will not be shown to you anymore.",
                                "Thank you for your report. We will review this message and take appropriate action."]
                    elif self.additional_option == AdditionalEDOptions.STOP_HERE:
                        return ["Thank you for your report. We will review this message and take appropriate action."]
                else:
                    return ["Invalid selection. Please enter a number between 1 and " + str(len(options)) + "."]
            except ValueError:
                return ["Please enter a number to select an option."]

        return ["I'm not sure how to handle that. Please try again or say `help` for more information."]

    def report_complete(self):
        return self.state == State.REPORT_COMPLETE
        
    def _handle_category_selection(self):
        if self.category == ReportCategory.HARMFUL:
            self.state = State.AWAITING_HARMFUL_SUBCATEGORY
            return ["What type of harmful content are you reporting?", self._get_harmful_subcategories()]
        elif self.category == ReportCategory.ILLEGAL:
            self.state = State.AWAITING_ILLEGAL_SUBCATEGORY
            return ["What type of illegal or false content are you reporting?", self._get_illegal_subcategories()]
        return ["Invalid category."]
        
    def _handle_harmful_subcategory_selection(self):
        if self.subcategory == HarmfulSubcategory.SEXUAL_CONTENT:
            self.state = State.AWAITING_SEXUAL_CONTENT_TYPE
            return ["Please specify the type of sexual content:", self._get_sexual_content_types()]
        elif self.subcategory == HarmfulSubcategory.SUICIDE_SELF_HARM:
            self.state = State.AWAITING_SUICIDE_SELF_HARM_TYPE
            return ["Please provide more details about the suicide/self-harm content:", self._get_suicide_self_harm_types()]
        else:
            self.state = State.REPORT_COMPLETE
            return ["Report sent. ✅"]
            
    def _handle_illegal_subcategory_selection(self):
        if self.subcategory == IllegalSubcategory.ILLEGAL_ACTIVITIES:
            self.state = State.AWAITING_ILLEGAL_ACTIVITY_TYPE
            return ["Please specify the type of illegal activity:", self._get_illegal_activity_types()]
        elif self.subcategory == IllegalSubcategory.MISINFORMATION:
            self.state = State.AWAITING_MISINFORMATION_TYPE
            return ["Please specify the type of misinformation:", self._get_misinformation_types()]
        else:
            self.state = State.REPORT_COMPLETE
            return ["Report sent. ✅"]
    
    def _get_help_message(self):
        help_message = "**Reporting Process Help**\n\n"
        help_message += "The reporting process allows you to report messages that violate our community guidelines. "
        help_message += "You will be guided through a series of steps to provide information about the message you are reporting.\n\n"
        help_message += "**Commands:**\n"
        help_message += "- `cancel`: Cancel the reporting process at any time.\n"
        help_message += "- `help`: Display this help message.\n\n"
        help_message += "Follow the prompts and select the appropriate options to complete your report."
        return help_message
        
    def _get_categories(self):
        options = ""
        for i, category in enumerate(ReportCategory):
            options += f"{i+1}. {category.value}\n"
        return options
        
    def _get_harmful_subcategories(self):
        options = ""
        for i, subcategory in enumerate(HarmfulSubcategory):
            options += f"{i+1}. {subcategory.value}\n"
        return options
        
    def _get_illegal_subcategories(self):
        options = ""
        for i, subcategory in enumerate(IllegalSubcategory):
            options += f"{i+1}. {subcategory.value}\n"
        return options
        
    def _get_sexual_content_types(self):
        options = ""
        for i, content_type in enumerate(SexualContentType):
            options += f"{i+1}. {content_type.value}\n"
        return options
        
    def _get_suicide_self_harm_types(self):
        options = ""
        for i, content_type in enumerate(SuicideSelfHarmType):
            options += f"{i+1}. {content_type.value}\n"
        return options
        
    def _get_suicide_follow_up_options(self):
        options = ""
        for i, option in enumerate(SuicideFollow):
            options += f"{i+1}. {option.value}\n"
        return options
        
    def _get_ed_follow_up_options(self):
        options = ""
        for i, option in enumerate(EatingDisorderFollow):
            options += f"{i+1}. {option.value}\n"
        return options
        
    def _get_illegal_activity_types(self):
        options = ""
        for i, content_type in enumerate(IllegalActivityType):
            options += f"{i+1}. {content_type.value}\n"
        return options
        
    def _get_misinformation_types(self):
        options = ""
        for i, content_type in enumerate(MisinformationType):
            options += f"{i+1}. {content_type.value}\n"
        return options
        
    def _get_additional_suicide_options(self):
        options = ""
        for i, option in enumerate(AdditionalSuicideOptions):
            options += f"{i+1}. {option.value}\n"
        return options
        
    def _get_additional_ed_options(self):
        options = ""
        for i, option in enumerate(AdditionalEDOptions):
            options += f"{i+1}. {option.value}\n"
        return options