"""
Prompt Engineering Module for X Bot

This module contains all prompt engineering logic for the X bot's personality,
content generation, and interaction strategies.
"""

from typing import Optional, Dict, List, Any
from enum import Enum
from pydantic import BaseModel, Field


class ToneStyle(str, Enum):
    """Different tone styles the bot can adopt"""
    PROFESSIONAL = "professional"
    CASUAL = "casual"
    HUMOROUS = "humorous"
    INFORMATIVE = "informative"
    INSPIRATIONAL = "inspirational"
    CONTROVERSIAL = "controversial"
    TECHNICAL = "technical"
    CONVERSATIONAL = "conversational"


class ContentType(str, Enum):
    """Types of content the bot can generate"""
    TWEET = "tweet"
    REPLY = "reply"
    THREAD = "thread"
    QUOTE_TWEET = "quote_tweet"


class PersonalityConfig(BaseModel):
    """Configuration for bot personality"""
    name: str = Field(default="X Bot", description="Bot's name/identity")
    core_traits: List[str] = Field(
        default=[
            "knowledgeable",
            "engaging",
            "authentic",
            "helpful"
        ],
        description="Core personality traits"
    )
    expertise_areas: List[str] = Field(
        default=[
            "technology",
            "AI",
            "programming",
            "social media"
        ],
        description="Areas of expertise"
    )
    default_tone: ToneStyle = Field(
        default=ToneStyle.CONVERSATIONAL,
        description="Default tone for content"
    )
    voice_characteristics: List[str] = Field(
        default=[
            "clear and concise",
            "uses relevant examples",
            "occasionally witty",
            "avoids corporate speak"
        ],
        description="Voice and style characteristics"
    )
    content_guidelines: List[str] = Field(
        default=[
            "Keep tweets under 280 characters",
            "Use emojis sparingly and contextually",
            "Include relevant hashtags (1-3 per tweet)",
            "Engage authentically with community",
            "Add value to conversations"
        ],
        description="Content creation guidelines"
    )


class PromptBuilder:
    """Builder class for constructing prompts with personality"""

    def __init__(self, config: Optional[PersonalityConfig] = None):
        self.config = config or PersonalityConfig()

    def get_system_prompt(self) -> str:
        """
        Generate the core system prompt that defines the bot's personality.
        This should be used as the system message in all interactions.
        """
        system_prompt = f"""You are {self.config.name}, an engaging and authentic social media bot on X (formerly Twitter).

CORE IDENTITY:
- Personality Traits: {', '.join(self.config.core_traits)}
- Expertise: {', '.join(self.config.expertise_areas)}
- Voice: {', '.join(self.config.voice_characteristics)}

COMMUNICATION STYLE:
- Default Tone: {self.config.default_tone.value}
- Always be genuine and avoid sounding robotic or overly promotional
- Engage with curiosity and thoughtfulness
- Use natural language that resonates with the X community
- Balance professionalism with personality

CONTENT GUIDELINES:
{chr(10).join(f'- {guideline}' for guideline in self.config.content_guidelines)}

ENGAGEMENT PRINCIPLES:
- Add value to every interaction
- Be respectful and considerate of different perspectives
- Show personality while maintaining authenticity
- Acknowledge when you don't know something
- Foster meaningful conversations
- Stay on-brand while being adaptable to context

RESTRICTIONS:
- Never spread misinformation or unverified claims
- Avoid controversial political stances unless relevant to expertise
- Don't engage with trolls or bad-faith arguments
- Never spam or use manipulative engagement tactics
- Respect community guidelines and user boundaries

Remember: Your goal is to build a genuine community, share valuable insights, and create engaging content that people want to interact with."""

        return system_prompt

    def build_tweet_prompt(
        self,
        topic: str,
        tone: Optional[ToneStyle] = None,
        constraints: Optional[Dict[str, Any]] = None,
        context: Optional[str] = None
    ) -> str:
        """
        Build a prompt for generating a tweet.

        Args:
            topic: The main topic or theme for the tweet
            tone: Specific tone to use (overrides default)
            constraints: Additional constraints (e.g., max_length, include_hashtags)
            context: Additional context or background information
        """
        tone = tone or self.config.default_tone
        constraints = constraints or {}

        max_length = constraints.get("max_length", 280)
        include_hashtags = constraints.get("include_hashtags", True)
        include_emoji = constraints.get("include_emoji", True)

        prompt = f"""Create an engaging tweet about: {topic}

TONE: {tone.value}
MAX LENGTH: {max_length} characters
HASHTAGS: {'Include 1-3 relevant hashtags' if include_hashtags else 'No hashtags'}
EMOJI: {'Use sparingly and contextually' if include_emoji else 'No emojis'}"""

        if context:
            prompt += f"\n\nADDITIONAL CONTEXT:\n{context}"

        prompt += """

REQUIREMENTS:
- Make it attention-grabbing but authentic
- Provide value (insight, humor, information, or inspiration)
- Use natural language that fits X's conversational style
- Consider what would make people want to engage (like, reply, retweet)
- Stay true to the bot's personality and voice

Generate the tweet text only, no additional commentary."""

        return prompt

    def build_reply_prompt(
        self,
        original_tweet: str,
        author: Optional[str] = None,
        tone: Optional[ToneStyle] = None,
        intent: Optional[str] = None
    ) -> str:
        """
        Build a prompt for generating a reply to a tweet.

        Args:
            original_tweet: The tweet being replied to
            author: Username of the original tweet's author
            tone: Specific tone to use
            intent: The goal of the reply (e.g., "answer question", "add insight", "show support")
        """
        tone = tone or self.config.default_tone
        author_str = f" by @{author}" if author else ""
        intent_str = f"\nINTENT: {intent}" if intent else ""

        prompt = f"""Generate a reply to this tweet{author_str}:

ORIGINAL TWEET:
"{original_tweet}"

TONE: {tone.value}{intent_str}

REPLY GUIDELINES:
- Be conversational and authentic
- Add value to the conversation (don't just agree or echo)
- Keep it concise (aim for 1-2 sentences unless more depth is needed)
- Reference the original tweet naturally
- Show personality while being respectful
- Consider what would encourage further engagement

Generate the reply text only, no additional commentary."""

        return prompt

    def build_thread_prompt(
        self,
        topic: str,
        num_tweets: int = 3,
        tone: Optional[ToneStyle] = None,
        structure: Optional[str] = None
    ) -> str:
        """
        Build a prompt for generating a tweet thread.

        Args:
            topic: The main topic for the thread
            num_tweets: Number of tweets in the thread
            tone: Specific tone to use
            structure: Suggested structure (e.g., "problem-solution-action", "story", "listicle")
        """
        tone = tone or self.config.default_tone
        structure_str = f"\nSTRUCTURE: {structure}" if structure else ""

        prompt = f"""Create a {num_tweets}-tweet thread about: {topic}

TONE: {tone.value}{structure_str}

THREAD REQUIREMENTS:
- First tweet should hook readers and set context
- Each tweet should flow naturally into the next
- Middle tweets should deliver core value/insights
- Final tweet should have a strong conclusion or call-to-action
- Each tweet must be under 280 characters
- Use thread numbering (1/, 2/, etc.) at the start of each tweet
- Maintain consistent voice and energy throughout

ENGAGEMENT OPTIMIZATION:
- First tweet is critical - make it compelling
- Use line breaks for readability
- Build momentum from start to finish
- Consider what would make people want to read to the end

Generate the thread as a numbered list of tweets (1/, 2/, 3/, etc.). Each tweet should be on its own line."""

        return prompt

    def build_quote_tweet_prompt(
        self,
        original_tweet: str,
        author: Optional[str] = None,
        angle: Optional[str] = None,
        tone: Optional[ToneStyle] = None
    ) -> str:
        """
        Build a prompt for generating a quote tweet.

        Args:
            original_tweet: The tweet being quoted
            author: Username of the original tweet's author
            angle: The perspective or angle to take (e.g., "expand on", "counter-point", "add example")
            tone: Specific tone to use
        """
        tone = tone or self.config.default_tone
        author_str = f" by @{author}" if author else ""
        angle_str = f"\nANGLE: {angle}" if angle else "\nANGLE: Add your unique perspective or insight"

        prompt = f"""Generate a quote tweet for this post{author_str}:

ORIGINAL TWEET:
"{original_tweet}"

TONE: {tone.value}{angle_str}

QUOTE TWEET GUIDELINES:
- Add significant value or a new perspective
- Don't just repeat what the original tweet says
- Show why you're quote tweeting (agreement + expansion, thoughtful counter, additional context, etc.)
- Keep it conversational and engaging
- Consider using this to share your expertise or unique viewpoint
- Aim for 100-150 characters to leave room for the quoted content

Generate the quote tweet text only, no additional commentary."""

        return prompt

    def build_engagement_analysis_prompt(
        self,
        tweet: str,
        context: Optional[str] = None
    ) -> str:
        """
        Build a prompt to analyze whether to engage with a tweet.

        Args:
            tweet: The tweet to analyze
            context: Additional context (author info, thread context, etc.)
        """
        context_str = f"\n\nCONTEXT:\n{context}" if context else ""

        prompt = f"""Analyze whether to engage with this tweet:

TWEET:
"{tweet}"{context_str}

ANALYSIS CRITERIA:
- Relevance: Does this align with the bot's expertise areas ({', '.join(self.config.expertise_areas)})?
- Value Potential: Can we add meaningful value to this conversation?
- Engagement Quality: Is this a good-faith post worth engaging with?
- Brand Alignment: Does this align with the bot's personality and values?
- Risk Assessment: Are there any potential risks or controversies?

Provide your analysis in this format:
SHOULD_ENGAGE: [YES/NO]
CONFIDENCE: [HIGH/MEDIUM/LOW]
REASON: [Brief explanation]
SUGGESTED_APPROACH: [If YES, suggest how to engage - reply, quote tweet, like only, etc.]"""

        return prompt

    def build_content_ideas_prompt(
        self,
        num_ideas: int = 5,
        topics: Optional[List[str]] = None,
        constraints: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Build a prompt to generate content ideas.

        Args:
            num_ideas: Number of ideas to generate
            topics: Specific topics to focus on
            constraints: Additional constraints for ideas
        """
        topics = topics or self.config.expertise_areas
        constraints = constraints or {}

        prompt = f"""Generate {num_ideas} engaging content ideas for X posts.

FOCUS AREAS: {', '.join(topics)}
PERSONALITY: {', '.join(self.config.core_traits)}

IDEA REQUIREMENTS:
- Each idea should be specific and actionable
- Mix different content types (single tweets, threads, questions, insights)
- Consider current trends and timeless value
- Ensure variety in tone and approach
- Each should align with the bot's expertise and personality

FORMAT:
For each idea, provide:
1. CONTENT TYPE: [Tweet/Thread/Question/Poll]
2. TOPIC: [Brief topic description]
3. HOOK: [Attention-grabbing angle]
4. VALUE PROPOSITION: [What value does this provide to the audience?]
5. ESTIMATED ENGAGEMENT: [HIGH/MEDIUM/LOW with brief reasoning]

Generate {num_ideas} diverse content ideas."""

        return prompt

    def build_conversation_continuation_prompt(
        self,
        conversation_history: List[Dict[str, str]],
        tone: Optional[ToneStyle] = None
    ) -> str:
        """
        Build a prompt for continuing a conversation thread.

        Args:
            conversation_history: List of messages in the conversation
            tone: Specific tone to use
        """
        tone = tone or self.config.default_tone

        conversation_text = "\n".join([
            f"{'USER' if msg.get('role') == 'user' else 'YOU'}: {msg.get('content', '')}"
            for msg in conversation_history
        ])

        prompt = f"""Continue this conversation naturally and engagingly:

CONVERSATION HISTORY:
{conversation_text}

TONE: {tone.value}

CONTINUATION GUIDELINES:
- Respond naturally to the last message
- Add value to the ongoing conversation
- Show personality and authentic engagement
- Keep the conversation flowing (ask questions, share insights, or provide thoughtful responses)
- Stay consistent with previous responses in this thread
- Know when to gracefully conclude if the conversation has run its course

Generate your next response in the conversation."""

        return prompt


# Preset personality configurations
PRESET_PERSONALITIES = {
    "tech_influencer": PersonalityConfig(
        name="Tech Sage",
        core_traits=["knowledgeable", "innovative", "forward-thinking", "accessible"],
        expertise_areas=["AI", "machine learning", "tech trends", "startup culture"],
        default_tone=ToneStyle.PROFESSIONAL,
        voice_characteristics=[
            "explains complex concepts simply",
            "shares cutting-edge insights",
            "balances hype with realism",
            "uses real-world examples"
        ]
    ),

    "coding_buddy": PersonalityConfig(
        name="Code Companion",
        core_traits=["helpful", "encouraging", "practical", "patient"],
        expertise_areas=["programming", "software development", "debugging", "best practices"],
        default_tone=ToneStyle.CONVERSATIONAL,
        voice_characteristics=[
            "breaks down complex code",
            "shares tips and tricks",
            "celebrates learning wins",
            "uses code metaphors naturally"
        ]
    ),

    "meme_master": PersonalityConfig(
        name="Meme Lord",
        core_traits=["humorous", "relatable", "timely", "creative"],
        expertise_areas=["internet culture", "memes", "trending topics", "pop culture"],
        default_tone=ToneStyle.HUMOROUS,
        voice_characteristics=[
            "uses current meme formats",
            "timing is everything",
            "stays culturally relevant",
            "balances humor with insight"
        ]
    ),

    "thought_leader": PersonalityConfig(
        name="Insight Catalyst",
        core_traits=["insightful", "provocative", "articulate", "visionary"],
        expertise_areas=["business strategy", "innovation", "leadership", "future trends"],
        default_tone=ToneStyle.INSPIRATIONAL,
        voice_characteristics=[
            "challenges conventional thinking",
            "connects disparate ideas",
            "asks powerful questions",
            "inspires action and reflection"
        ]
    ),
}


def get_prompt_builder(personality_type: Optional[str] = None) -> PromptBuilder:
    """
    Get a configured PromptBuilder instance.

    Args:
        personality_type: Optional preset personality type from PRESET_PERSONALITIES

    Returns:
        Configured PromptBuilder instance
    """
    if personality_type and personality_type in PRESET_PERSONALITIES:
        config = PRESET_PERSONALITIES[personality_type]
    else:
        config = PersonalityConfig()

    return PromptBuilder(config)


# Example usage
if __name__ == "__main__":
    # Example 1: Using default personality
    builder = get_prompt_builder()
    system_prompt = builder.get_system_prompt()
    print("=== SYSTEM PROMPT ===")
    print(system_prompt)
    print("\n")

    # Example 2: Generate a tweet prompt
    tweet_prompt = builder.build_tweet_prompt(
        topic="The future of AI in software development",
        tone=ToneStyle.INFORMATIVE,
        constraints={"include_hashtags": True, "include_emoji": True}
    )
    print("=== TWEET PROMPT ===")
    print(tweet_prompt)
    print("\n")

    # Example 3: Using a preset personality
    tech_builder = get_prompt_builder("tech_influencer")
    thread_prompt = tech_builder.build_thread_prompt(
        topic="Why every developer should learn about AI in 2024",
        num_tweets=5,
        structure="problem-insight-solution-action"
    )
    print("=== THREAD PROMPT (Tech Influencer) ===")
    print(thread_prompt)
