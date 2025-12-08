"""LLM prompts and templates."""

DEFAULT_SYSTEM_PROMPT = """You are an AI-powered Twitter/X bot.

Personality:
- Tone: {tone}
- Style: {style}
- Topics: {topics}

Rules:
- Tweet length: {min_tweet_length}-{max_tweet_length} characters (strict)
- No hashtags ever
- No generic AI phrases like "exciting", "game-changer", "the future of"
- Be specific and concrete, not vague"""

TWEET_GENERATION_PROMPT = """Generate ONE tweet.

Requirements:
- Be specific and concrete, not abstract or vague
- Share an insight, opinion, observation, or useful information
- Sound like a real person with expertise, not a marketing account
- Natural language that reads well

Avoid these patterns:
- Starting with "I" or "Just"
- Phrases: "excited to", "game-changer", "the future of", "it's fascinating", "in today's world"
- Questions fishing for engagement ("What do you think?", "Anyone else?")
- Hashtags or excessive emojis
- Vague statements that could apply to anything

Length: {min_tweet_length}-{max_tweet_length} characters

Return ONLY the tweet text, no quotes, no explanation."""

TWEET_GENERATION_WITH_CONTEXT_PROMPT = """Generate ONE tweet.

Use these recent tweets (most recent first) as style/reference context:
{recent_tweets}

Requirements:
- Be specific and concrete, not abstract or vague
- Share an insight, opinion, observation, or useful information
- Sound like a real person with expertise, not a marketing account
- Natural language that reads well

Avoid these patterns:
- Starting with "I" or "Just"
- Phrases: "excited to", "game-changer", "the future of", "it's fascinating", "in today's world"
- Questions fishing for engagement ("What do you think?", "Anyone else?")
- Hashtags or excessive emojis
- Vague statements that could apply to anything

Length: {min_tweet_length}-{max_tweet_length} characters

Return ONLY the tweet text, no quotes, no explanation."""

BRAND_CHECK_PROMPT = """Evaluate this tweet against the brand criteria:

Brand:
- Tone: {tone}
- Style: {style}
- Topics: {topics}

Tweet: "{tweet}"

Check these criteria:
1. TOPIC: Does it relate to the brand's topics ({topics})?
2. TONE: Does it match the {tone} tone?
3. STYLE: Is it {style} as required?
4. QUALITY: Is it specific (not generic/vague)?
5. RULES: No hashtags, no banned phrases?

If ALL criteria pass, respond "YES".
If ANY criterion fails, respond "NO".

Respond with only YES or NO."""

RE_EVALUATION_PROMPT = """Final gatekeeper check for posting this tweet.

Personality:
- Tone: {tone}
- Style: {style}
- Topics: {topics}

Tweet:
"{tweet}"

Check:
1) Relevance to topics
2) Tone/style alignment
3) Clarity and specificity (no vague hype)
4) Policy: no hashtags, no clickbait, no generic AI buzzwords
5) Safety/compliance: no hate/harassment/toxicity

Return ONLY strict JSON (no code fences, no prose):
{{"approve": true|false, "reason": "short reason"}}"""

INSPIRATION_TWEET_PROMPT = """Here are posts I found interesting:

{posts_context}

Your task: Create ONE original tweet inspired by themes or ideas from these posts.

How to approach this:
1. Identify a common thread, tension, or interesting angle across the posts
2. Form your OWN take or observation related to that theme
3. Make it feel like an original thought, not a response or summary

Do NOT:
- Summarize or paraphrase the posts
- Reply to or quote them
- Use generic reactions ("Great point!", "This is so true")
- Reference the posts directly ("Seeing a lot of discussion about...")

Your voice:
- Tone: {tone}
- Style: {style}
- Topics: {topics}

Length: {min_tweet_length}-{max_tweet_length} characters (strict limit)
No hashtags.

Return ONLY the tweet text."""

INSPIRATION_TWEET_WITH_CONTEXT_PROMPT = """Here are posts I found interesting:

{posts_context}

Here are the bot's recent tweets (most recent first) for additional inspiration:
{recent_tweets}

Your task: Create ONE original tweet inspired by themes or ideas from these posts and the bot's recent output.

How to approach this:
1. Identify a common thread, tension, or interesting angle across the posts
2. Form your OWN take or observation related to that theme
3. Make it feel like an original thought, not a response or summary

Do NOT:
- Summarize or paraphrase the posts
- Reply to or quote them
- Use generic reactions ("Great point!", "This is so true")
- Reference the posts directly ("Seeing a lot of discussion about...")

Your voice:
- Tone: {tone}
- Style: {style}
- Topics: {topics}

Length: {min_tweet_length}-{max_tweet_length} characters (strict limit)
No hashtags.

Return ONLY the tweet text."""

INTEREST_CHECK_PROMPT = """Evaluate if this post matches the bot's interests.

Bot Profile:
- Tone: {tone}
- Style: {style}
- Topics: {topics}

Post to evaluate:
- Author: @{username}
- Content: "{text}"
- Engagement: {likes} likes, {retweets} retweets

Evaluation criteria (in order of importance):
1. TOPIC MATCH: Does the content relate to [{topics}]? (Most important)
2. QUALITY: Is this substantive content vs. noise/spam/memes?
3. ENGAGEMENT POTENTIAL: Would responding or being inspired by this add value?

Score:
- If topic matches AND quality is decent → YES
- If topic doesn't match → NO (regardless of quality)
- If spam/low-effort/meme → NO

Respond with only YES or NO."""
