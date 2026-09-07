from __future__ import annotations

import json
from pathlib import Path

from openai import OpenAI
from pydantic import TypeAdapter, ValidationError

from clipmind.analyzer import AIAnalyzer
from clipmind.config import Settings
from clipmind.models import SocialContentBundle, SocialPlatform, SocialPost, Transcript, ViralClip
from clipmind.utils import get_logger, read_json, slugify, write_json


SOCIAL_POSTS_ADAPTER = TypeAdapter(list[SocialPost])


SOCIAL_SYSTEM_PROMPT = """You are a senior social media strategist for premium short-form content.
Create platform-native metadata that feels human, current, and high-converting.
You understand YouTube Shorts, Facebook Reels, and TikTok distribution.
Return strict JSON only. No markdown, no commentary, no trailing commas."""


PLATFORM_RULES = {
    "youtube": """YouTube Shorts rules:
- Titles: provide 3 alternatives. They should be high-CTR, hook-driven, emotionally specific, and ideally under 60 characters. Emojis are allowed when they improve CTR.
- Put the main keyword or curiosity phrase early. Avoid clickbait that the clip cannot satisfy.
- Description: first 150 characters must work as the search/preview hook.
- Include a structured description with value summary, optional timestamps when useful, keywords, CTA, and a clear Follow Me section with saved social links.
- Include 3-5 relevant hashtags at the end.
- Never use more than 15 hashtags. Prefer 3-5. Keep links in the description, not the title.""",
    "facebook": """Facebook Reels rules:
- Titles: provide 3 alternatives focused on storytelling, relatability, and shareability.
- Title/caption should be clear, shareable, and built around a hook/problem/payoff.
- Description should usually be 150-200 characters before links/hashtags, with a soft CTA that invites comments or shares.
- Start with an engaging question when possible. Add clean social links at the bottom.
- Use 3-5 highly relevant hashtags at the end. Avoid stuffing or irrelevant trends.""",
    "tiktok": """TikTok rules:
- Titles: provide 3 extremely short, punchy, curiosity-inducing alternatives under 60 characters.
- Caption should lead with a strong hook in the first 80-100 characters.
- Use natural searchable keywords in the caption before hashtags.
- Keep description brief, high-energy, and front-load the most relevant hashtags.
- End with a comment-driving question or CTA.
- Use 3-5 relevant hashtags. Avoid lazy #fyp/#foryoupage spam unless truly relevant.
- If branded/sponsored, include a visible disclosure note.""",
}


class SocialContentGenerator:
    """Generates platform-specific titles, descriptions, and hashtags for selected clips."""

    def __init__(self, settings: Settings, client: OpenAI | None = None) -> None:
        self.settings = settings
        self._client = client
        self.logger = get_logger()

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(api_key=self.settings.openai_api_key)
        return self._client

    def social_path(self, video_path: Path) -> Path:
        return self.settings.cache_dir / f"{slugify(video_path.stem)}.social.json"

    def generate(
        self,
        *,
        transcript: Transcript,
        clips: list[ViralClip],
        video_path: Path,
        platforms: list[SocialPlatform],
        force: bool = False,
        target_clip_id: str | None = None,
    ) -> SocialContentBundle:
        output_path = self.social_path(video_path)
        if output_path.exists() and not force and target_clip_id is None:
            self.logger.info("Using cached social media metadata: %s", output_path)
            return SocialContentBundle.model_validate(read_json(output_path))

        self.settings.require_openai_key()
        existing = SocialContentBundle(source_video=str(video_path), posts=[])
        if output_path.exists():
            existing = SocialContentBundle.model_validate(read_json(output_path))

        target_clips = [clip for clip in clips if target_clip_id in (None, clip.clip_id)]
        if not target_clips:
            raise ValueError("No clips available for social metadata generation.")

        self.logger.info("Generating social media titles and descriptions")
        generated_posts = self._request_social_posts(transcript, target_clips, platforms)
        merged_posts = self._merge_posts(existing.posts, generated_posts, target_clip_id)
        bundle = SocialContentBundle(source_video=str(video_path), posts=merged_posts)
        write_json(output_path, bundle.model_dump(mode="json"))
        return bundle

    def generate_strategy(
        self,
        *,
        transcript: Transcript,
        clip: ViralClip,
        video_path: Path,
        platform: SocialPlatform,
        social_settings: dict[str, str],
        force: bool = True,
        temperature: float = 0.75,
        prompt_variation: str = "",
    ) -> SocialPost:
        output_path = self.social_path(video_path)
        existing = SocialContentBundle(source_video=str(video_path), posts=[])
        if output_path.exists():
            existing = SocialContentBundle.model_validate(read_json(output_path))
            if not force:
                for post in existing.posts:
                    if post.platform == platform and post.clip_id == clip.clip_id:
                        return post

        self.settings.require_openai_key()
        prompt = self._build_strategy_prompt(
            transcript=transcript,
            clip=clip,
            platform=platform,
            social_settings=social_settings,
            prompt_variation=prompt_variation,
        )
        completion = self.client.chat.completions.create(
            model=self.settings.openai_analysis_model,
            messages=[
                {"role": "system", "content": SOCIAL_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        message = completion.choices[0].message.content
        if not message:
            raise RuntimeError("OpenAI returned an empty social strategy response")
        posts = self.parse_posts(message)
        if not posts:
            raise RuntimeError("OpenAI did not return any social strategy posts")
        post = posts[0].model_copy(update={"platform": platform, "clip_id": clip.clip_id})
        merged_posts = self._merge_posts(existing.posts, [post], clip.clip_id)
        bundle = SocialContentBundle(source_video=str(video_path), posts=merged_posts)
        write_json(output_path, bundle.model_dump(mode="json"))
        return post

    def parse_posts(self, payload: str | list[dict] | dict) -> list[SocialPost]:
        if isinstance(payload, str):
            parsed = json.loads(AIAnalyzer._extract_json(payload))
        else:
            parsed = payload
        if isinstance(parsed, dict) and "posts" in parsed:
            parsed = parsed["posts"]
        if isinstance(parsed, dict) and "post" in parsed:
            parsed = [parsed["post"]]
        try:
            return SOCIAL_POSTS_ADAPTER.validate_python(parsed)
        except ValidationError as exc:
            raise ValueError(f"Social metadata JSON failed validation: {exc}") from exc

    def _request_social_posts(
        self,
        transcript: Transcript,
        clips: list[ViralClip],
        platforms: list[SocialPlatform],
    ) -> list[SocialPost]:
        prompt = self._build_prompt(transcript, clips, platforms)
        completion = self.client.chat.completions.create(
            model=self.settings.openai_analysis_model,
            messages=[
                {"role": "system", "content": SOCIAL_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.55,
            response_format={"type": "json_object"},
        )
        message = completion.choices[0].message.content
        if not message:
            raise RuntimeError("OpenAI returned an empty social metadata response")
        return self.parse_posts(message)

    def _build_prompt(
        self,
        transcript: Transcript,
        clips: list[ViralClip],
        platforms: list[SocialPlatform],
    ) -> str:
        platform_context = self._platform_context(platforms)
        clip_context = "\n".join(
            self._clip_prompt_block(transcript, clip)
            for clip in clips
        )
        return f"""Generate social publishing metadata for these clips.

Output format:
{{
  "posts": [
    {{
      "platform": "youtube|facebook|tiktok",
      "clip_id": "clip_1",
      "title": "catchy platform-native title",
      "description": "ready-to-publish description with platform links when provided",
      "hashtags": ["#Relevant", "#Specific"],
      "call_to_action": "short CTA",
      "compliance_notes": ["optional notes for branded/sensitive content"]
    }}
  ]
}}

Hard requirements:
- Generate one post for every requested platform for every clip.
- Use the same language and dialect as the clip when appropriate. Arabic/Egyptian Arabic is allowed and preferred when the transcript is Arabic.
- Titles must be emotionally compelling but truthful to the clip.
- Descriptions must include the platform profile link and default platform description/footer if provided below.
- Do not invent links.
- Hashtags must be relevant, concise, and platform-appropriate.
- No spam, misleading claims, fake giveaways, or engagement bait that violates platform norms.

Platform rules and account settings:
{platform_context}

Clip context:
{clip_context}"""

    def _build_strategy_prompt(
        self,
        *,
        transcript: Transcript,
        clip: ViralClip,
        platform: SocialPlatform,
        social_settings: dict[str, str],
        prompt_variation: str,
    ) -> str:
        clip_context = self._clip_prompt_block(transcript, clip)
        settings_block = self._qsettings_context(social_settings)
        variation = prompt_variation or "Create a fresh, premium, platform-native set."
        return f"""Generate one premium social media publishing strategy for {platform.upper()}.

Return strict JSON in this shape:
{{
  "post": {{
    "platform": "{platform}",
    "clip_id": "{clip.clip_id}",
    "title": "best title from the alternatives",
    "titles": ["alternative 1", "alternative 2", "alternative 3"],
    "description": "fully ready-to-publish platform description",
    "hashtags": ["#Relevant", "#Specific", "#PlatformNative"],
    "call_to_action": "CTA using the saved default CTA when appropriate",
    "compliance_notes": []
  }}
}}

Mandatory platform strategy:
{PLATFORM_RULES[platform]}

Saved creator settings from QSettings:
{settings_block}

Extra generation direction:
{variation}

Hard requirements:
- Generate exactly 3 alternative titles.
- Use the clip language naturally. If Arabic/Egyptian Arabic fits the clip, use it.
- Inject only saved links that are provided. Do not invent URLs.
- Always incorporate default_cta_text if provided.
- Blend global_hashtags with platform-specific hashtags, but keep counts within platform rules.
- Description must be ready to paste directly into {platform}.
- Avoid false claims, misleading clickbait, banned hashtag spam, or fake engagement bait.

Clip context:
{clip_context}"""

    def _platform_context(self, platforms: list[SocialPlatform]) -> str:
        contexts: list[str] = []
        for platform in platforms:
            url, default_description = self._profile_for(platform)
            contexts.append(
                f"[{platform.upper()}]\n"
                f"{PLATFORM_RULES[platform]}\n"
                f"Profile link: {url or 'not provided'}\n"
                f"Default description/footer: {default_description or 'not provided'}"
            )
        return "\n\n".join(contexts)

    def _clip_prompt_block(self, transcript: Transcript, clip: ViralClip) -> str:
        words = transcript.words_for_range(clip.start_time, clip.end_time)
        excerpt = " ".join(word.text for word in words)
        if not excerpt:
            excerpt = transcript.text[:1600]
        return (
            f"Clip ID: {clip.clip_id}\n"
            f"Time: {clip.start_time:.2f}s - {clip.end_time:.2f}s\n"
            f"Viral score: {clip.viral_score}\n"
            f"Existing hook: {clip.hook_text}\n"
            f"Why engaging: {clip.explanation}\n"
            f"Transcript excerpt: {excerpt[:2600]}"
        )

    def _profile_for(self, platform: SocialPlatform) -> tuple[str, str]:
        if platform == "youtube":
            return self.settings.youtube_channel_url, self.settings.youtube_default_description
        if platform == "facebook":
            return self.settings.facebook_page_url, self.settings.facebook_default_description
        return self.settings.tiktok_profile_url, self.settings.tiktok_default_description

    @staticmethod
    def _qsettings_context(social_settings: dict[str, str]) -> str:
        keys = [
            "youtube_channel_link",
            "facebook_page_link",
            "tiktok_profile_link",
            "instagram_profile_link",
            "default_cta_text",
            "global_hashtags",
        ]
        return "\n".join(
            f"{key}: {social_settings.get(key, '') or 'not provided'}"
            for key in keys
        )

    @staticmethod
    def _merge_posts(
        existing_posts: list[SocialPost],
        generated_posts: list[SocialPost],
        target_clip_id: str | None,
    ) -> list[SocialPost]:
        generated_keys = {(post.platform, post.clip_id) for post in generated_posts}
        merged = [
            post
            for post in existing_posts
            if (post.platform, post.clip_id) not in generated_keys
        ]
        merged.extend(generated_posts)
        return sorted(merged, key=lambda post: (post.clip_id, post.platform))
