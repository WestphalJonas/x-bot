"""
Twitter scraper using twscrape library.

This module provides async functions to scrape Twitter data without official API keys.
"""

import asyncio
from typing import List, Optional, Dict, Any
from datetime import datetime
from twscrape import API, gather
from twscrape.models import Tweet, User


class TwitterScraper:
    """
    Twitter scraper using twscrape.

    This class provides methods to:
    - Add and manage Twitter accounts for scraping
    - Scrape tweets from specific users
    - Search tweets by query
    - Get user profiles
    """

    def __init__(self, db_path: str = "accounts.db"):
        """
        Initialize the Twitter scraper.

        Args:
            db_path: Path to the SQLite database for storing account credentials
        """
        self.api = API(db_path)

    async def add_account(
        self,
        username: str,
        password: str,
        email: str,
        email_password: str
    ) -> None:
        """
        Add a Twitter account for scraping.

        Args:
            username: Twitter username
            password: Twitter password
            email: Email associated with the account
            email_password: Email password for verification
        """
        await self.api.pool.add_account(
            username=username,
            password=password,
            email=email,
            email_password=email_password
        )
        await self.api.pool.login_all()

    async def login_accounts(self) -> None:
        """Login all accounts in the pool."""
        await self.api.pool.login_all()

    async def get_user_tweets(
        self,
        username: str,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Get tweets from a specific user.

        Args:
            username: Twitter username (without @)
            limit: Maximum number of tweets to retrieve

        Returns:
            List of tweet dictionaries with relevant information
        """
        tweets = await gather(self.api.user_tweets(username, limit=limit))

        return [
            {
                "id": tweet.id,
                "url": tweet.url,
                "date": tweet.date,
                "user": tweet.user.username,
                "text": tweet.rawContent,
                "reply_count": tweet.replyCount,
                "retweet_count": tweet.retweetCount,
                "like_count": tweet.likeCount,
                "quote_count": tweet.quoteCount,
                "view_count": tweet.viewCount,
                "bookmark_count": tweet.bookmarkCount,
                "hashtags": tweet.hashtags,
                "mentions": [m.username for m in tweet.mentionedUsers] if tweet.mentionedUsers else [],
                "links": tweet.links,
                "media": [
                    {
                        "type": m.type,
                        "url": m.url if hasattr(m, 'url') else None
                    }
                    for m in (tweet.media or [])
                ],
                "is_retweet": tweet.retweetedTweet is not None,
                "is_reply": tweet.inReplyToTweetId is not None,
                "is_quote": tweet.quotedTweet is not None,
                "language": tweet.lang,
            }
            for tweet in tweets
        ]

    async def search_tweets(
        self,
        query: str,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Search tweets by query.

        Args:
            query: Search query (supports Twitter search operators)
            limit: Maximum number of tweets to retrieve

        Returns:
            List of tweet dictionaries

        Examples:
            - "python programming" - search for tweets containing these words
            - "from:elonmusk" - tweets from specific user
            - "#AI" - tweets with hashtag
            - "bitcoin since:2024-01-01" - tweets since a date
        """
        tweets = await gather(self.api.search(query, limit=limit))

        return [
            {
                "id": tweet.id,
                "url": tweet.url,
                "date": tweet.date,
                "user": tweet.user.username,
                "text": tweet.rawContent,
                "reply_count": tweet.replyCount,
                "retweet_count": tweet.retweetCount,
                "like_count": tweet.likeCount,
                "quote_count": tweet.quoteCount,
                "view_count": tweet.viewCount,
                "hashtags": tweet.hashtags,
                "mentions": [m.username for m in tweet.mentionedUsers] if tweet.mentionedUsers else [],
                "language": tweet.lang,
            }
            for tweet in tweets
        ]

    async def get_user_info(self, username: str) -> Dict[str, Any]:
        """
        Get user profile information.

        Args:
            username: Twitter username (without @)

        Returns:
            Dictionary with user information
        """
        user = await self.api.user_by_login(username)

        return {
            "id": user.id,
            "username": user.username,
            "display_name": user.displayname,
            "description": user.rawDescription,
            "followers_count": user.followersCount,
            "following_count": user.friendsCount,
            "tweets_count": user.statusesCount,
            "created": user.created,
            "verified": user.verified,
            "profile_image_url": user.profileImageUrl,
            "profile_banner_url": user.profileBannerUrl,
            "location": user.location,
            "url": user.url,
        }

    async def get_tweet_replies(
        self,
        tweet_id: int,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Get replies to a specific tweet.

        Args:
            tweet_id: Tweet ID
            limit: Maximum number of replies to retrieve

        Returns:
            List of reply tweet dictionaries
        """
        replies = await gather(self.api.tweet_replies(tweet_id, limit=limit))

        return [
            {
                "id": tweet.id,
                "url": tweet.url,
                "date": tweet.date,
                "user": tweet.user.username,
                "text": tweet.rawContent,
                "reply_count": tweet.replyCount,
                "retweet_count": tweet.retweetCount,
                "like_count": tweet.likeCount,
            }
            for tweet in replies
        ]

    async def get_user_followers(
        self,
        username: str,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Get followers of a user.

        Args:
            username: Twitter username (without @)
            limit: Maximum number of followers to retrieve

        Returns:
            List of follower user dictionaries
        """
        user_id = (await self.api.user_by_login(username)).id
        followers = await gather(self.api.followers(user_id, limit=limit))

        return [
            {
                "id": user.id,
                "username": user.username,
                "display_name": user.displayname,
                "followers_count": user.followersCount,
                "verified": user.verified,
            }
            for user in followers
        ]

    async def get_user_following(
        self,
        username: str,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Get users that a user is following.

        Args:
            username: Twitter username (without @)
            limit: Maximum number of following to retrieve

        Returns:
            List of following user dictionaries
        """
        user_id = (await self.api.user_by_login(username)).id
        following = await gather(self.api.following(user_id, limit=limit))

        return [
            {
                "id": user.id,
                "username": user.username,
                "display_name": user.displayname,
                "followers_count": user.followersCount,
                "verified": user.verified,
            }
            for user in following
        ]


# Convenience functions for synchronous usage
def run_async(coro):
    """Helper function to run async coroutines synchronously."""
    return asyncio.run(coro)


# Example usage functions
async def example_scrape_user_tweets():
    """Example: Scrape tweets from a user."""
    scraper = TwitterScraper()

    # Get tweets from a user
    tweets = await scraper.get_user_tweets("elonmusk", limit=10)

    for tweet in tweets:
        print(f"@{tweet['user']}: {tweet['text'][:100]}...")
        print(f"  Likes: {tweet['like_count']}, Retweets: {tweet['retweet_count']}")
        print(f"  Date: {tweet['date']}")
        print()


async def example_search_tweets():
    """Example: Search for tweets."""
    scraper = TwitterScraper()

    # Search for tweets about AI
    tweets = await scraper.search_tweets("#AI", limit=10)

    for tweet in tweets:
        print(f"@{tweet['user']}: {tweet['text'][:100]}...")
        print(f"  Likes: {tweet['like_count']}")
        print()


async def example_get_user_info():
    """Example: Get user profile information."""
    scraper = TwitterScraper()

    # Get user info
    user_info = await scraper.get_user_info("elonmusk")

    print(f"Username: @{user_info['username']}")
    print(f"Display Name: {user_info['display_name']}")
    print(f"Followers: {user_info['followers_count']}")
    print(f"Following: {user_info['following_count']}")
    print(f"Tweets: {user_info['tweets_count']}")
    print(f"Description: {user_info['description']}")


if __name__ == "__main__":
    # Run examples
    print("=== Example: Get User Tweets ===")
    asyncio.run(example_scrape_user_tweets())

    print("\n=== Example: Search Tweets ===")
    asyncio.run(example_search_tweets())

    print("\n=== Example: Get User Info ===")
    asyncio.run(example_get_user_info())
