from clipmind.analyzer import AIAnalyzer
from clipmind.config import Settings


def test_parse_strict_json_array() -> None:
    analyzer = AIAnalyzer(Settings(openai_api_key=""))
    clips = analyzer.parse_clips(
        """
        [
          {
            "clip_id": "clip_1",
            "start_time": 10.0,
            "end_time": 45.0,
            "viral_score": 91,
            "hook_text": "This changes everything",
            "explanation": "Strong hook and payoff."
          }
        ]
        """
    )

    assert len(clips) == 1
    assert clips[0].clip_id == "clip_1"
    assert clips[0].duration == 35.0


def test_parse_json_object_fallback_shape() -> None:
    analyzer = AIAnalyzer(Settings(openai_api_key=""))
    clips = analyzer.parse_clips(
        {
            "clips": [
                {
                    "clip_id": "clip 2",
                    "start_time": 65.0,
                    "end_time": 110.0,
                    "viral_score": 88,
                    "hook_text": "Nobody talks about this",
                    "explanation": "Clear curiosity gap.",
                }
            ]
        }
    )

    assert clips[0].clip_id == "clip_2"
