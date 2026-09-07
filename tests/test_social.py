from clipmind.config import Settings
from clipmind.models import SocialPost
from clipmind.social import PLATFORM_RULES, SocialContentGenerator


def test_parse_social_posts_object_shape() -> None:
    generator = SocialContentGenerator(Settings(openai_api_key=""))
    posts = generator.parse_posts(
        {
            "posts": [
                {
                    "platform": "youtube",
                    "clip_id": "clip_1",
                    "title": "الخوارزمي رجع 2026!",
                    "description": "وصف جاهز للنشر",
                    "hashtags": ["#Shorts", "#الخوارزمي"],
                    "call_to_action": "اشترك وشوف الكليب كامل",
                    "compliance_notes": [],
                }
            ]
        }
    )

    assert len(posts) == 1
    assert posts[0].platform == "youtube"
    assert posts[0].hashtags == ["#Shorts", "#الخوارزمي"]
    assert posts[0].titles == ["الخوارزمي رجع 2026!"]


def test_parse_social_strategy_post_shape_with_titles() -> None:
    generator = SocialContentGenerator(Settings(openai_api_key=""))
    posts = generator.parse_posts(
        {
            "post": {
                "platform": "tiktok",
                "clip_id": "clip_2",
                "title": "الخوارزمي اتصدم!",
                "titles": ["الخوارزمي اتصدم!", "محدش كان متوقع ده", "رجع من الماضي"],
                "description": "#خوارزمي القصة دي غريبة جدًا. تابع للنهاية!",
                "hashtags": ["#TikTok", "#الخوارزمي", "#AI"],
                "call_to_action": "اكتب رأيك",
                "compliance_notes": [],
            }
        }
    )

    assert len(posts) == 1
    assert posts[0].platform == "tiktok"
    assert len(posts[0].titles) == 3
    assert posts[0].title == posts[0].titles[0]


def test_merge_posts_replaces_target_clip_posts() -> None:
    old = SocialPost(platform="tiktok", clip_id="clip_1", title="old", description="old")
    new = SocialPost(platform="tiktok", clip_id="clip_1", title="new", description="new")

    merged = SocialContentGenerator._merge_posts([old], [new], "clip_1")

    assert len(merged) == 1
    assert merged[0].title == "new"


def test_merge_posts_keeps_other_platforms_for_same_clip() -> None:
    old_tiktok = SocialPost(platform="tiktok", clip_id="clip_1", title="old", description="old")
    old_youtube = SocialPost(platform="youtube", clip_id="clip_1", title="keep", description="keep")
    new_tiktok = SocialPost(platform="tiktok", clip_id="clip_1", title="new", description="new")

    merged = SocialContentGenerator._merge_posts([old_tiktok, old_youtube], [new_tiktok], "clip_1")

    assert {(post.platform, post.title) for post in merged} == {
        ("tiktok", "new"),
        ("youtube", "keep"),
    }


def test_platform_rules_include_current_constraints() -> None:
    assert "3-5" in PLATFORM_RULES["tiktok"]
    assert "first 150 characters" in PLATFORM_RULES["youtube"]
    assert "150-200 characters" in PLATFORM_RULES["facebook"]


def test_qsettings_context_includes_requested_fields() -> None:
    context = SocialContentGenerator._qsettings_context(
        {
            "youtube_channel_link": "https://youtube.example",
            "facebook_page_link": "https://facebook.example",
            "tiktok_profile_link": "https://tiktok.example",
            "instagram_profile_link": "https://instagram.example",
            "default_cta_text": "Subscribe for more content!",
            "global_hashtags": "#ClipMind, #Shorts",
        }
    )

    assert "youtube_channel_link: https://youtube.example" in context
    assert "instagram_profile_link: https://instagram.example" in context
    assert "global_hashtags: #ClipMind, #Shorts" in context
