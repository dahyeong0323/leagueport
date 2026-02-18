import unittest

from app.services.llm_input_transformer import build_llm_input_summary


class TestLlmInputTransformer(unittest.TestCase):
    def test_build_llm_input_summary_shape(self) -> None:
        summary = {
            "games_analyzed": 20,
            "streak_state": {"type": "W", "length": 3},
            "win_rate": 0.57,
            "avg_kda": "6.0/3.0/8.0",
            "deaths_per_game": 3.0,
            "avg_cs_per_min": 7.7,
            "main_role": "MIDDLE",
            "most_played_champ": "Ahri",
            "champion_pool_size": 6,
            "early_impact_proxy": 0.7,
            "avg_game_duration_min": 31.2,
        }

        llm_input_summary = build_llm_input_summary(summary)

        self.assertEqual(
            set(llm_input_summary.keys()),
            {
                "sample_size",
                "recent_result",
                "avg_kda",
                "avg_deaths",
                "cs_per_min",
                "main_position",
                "main_champion",
                "champion_pool_size",
                "early_game_presence",
                "avg_game_duration",
            },
        )
        self.assertEqual(llm_input_summary["sample_size"], 20)
        self.assertEqual(llm_input_summary["recent_result"], "W")
        self.assertEqual(llm_input_summary["main_position"], "MIDDLE")
        self.assertEqual(llm_input_summary["main_champion"], "Ahri")


if __name__ == "__main__":
    unittest.main()
