"""The background a trader carries into a chat.

Two properties matter most here and each has a dedicated test: the user's own
custom persona text must survive into the briefing (it is the thing chat was
missing), and building a briefing must not touch a single byte of the agent's
trading state (chat is read-only, so a run stays reproducible).
"""

import copy

import pytest

from ai_trading_society.agents.external_ai_agent import ExternalAIAgent
from ai_trading_society.agents.player_agent import PlayerAgent
from ai_trading_society.agents.roster import resolve_social_map
from ai_trading_society.agents.traits import (
    _IN_CHARACTER_CHECK,
    build_disposition,
    character_text,
    create_personality_agent,
)
from ai_trading_society.chat_context import (
    _MAX_BRIEFING_CHARS,
    agent_relations,
    build_chat_system_prompt,
    describe_peer,
)
from ai_trading_society.config import MarketConfig, StockSpec
from ai_trading_society.market_env import MarketEnv


def make_agent(name, personality="balanced", deep=True, **kwargs):
    base = ExternalAIAgent(name, cash=10000, model="claude-sonnet-5", api_key="k")
    return create_personality_agent(base, personality=personality, deep=deep, **kwargs)


def make_env(agents, deep=True):
    config = MarketConfig(
        stocks=[
            StockSpec(name="TechTitan", initial_price=150.0),
            StockSpec(name="MegaBank", initial_price=250.0),
        ],
        seed=3,
        deep_persona=deep,
    )
    env = MarketEnv(config, agents)
    env.social_map = resolve_social_map(list(env.agents.values()))
    return env


@pytest.fixture
def env():
    return make_env([
        make_agent("Ada", "aggressive"),
        make_agent("Nova", "greedy"),
        make_agent("Kit", "panicky"),
    ])


class TestCharacterText:
    def test_strips_the_trading_specific_check(self):
        disposition = build_disposition("aggressive", deep=True)
        stripped = character_text(disposition)
        assert _IN_CHARACTER_CHECK in disposition
        assert _IN_CHARACTER_CHECK not in stripped
        assert "never explain a hold and then trade" not in stripped

    def test_keeps_identity_and_preset_paragraph(self):
        stripped = character_text(build_disposition("aggressive", deep=True))
        assert "You are a real trader, not a calculator" in stripped
        # A dial sentence: proof the deep-mode character body survived.
        assert "big positions" in stripped

    def test_empty_input_is_safe(self):
        assert character_text("") == ""
        assert character_text(None) == ""  # type: ignore[arg-type]


class TestIdentityAndRelations:
    def test_identity_section_present(self, env):
        out = build_chat_system_prompt(env, "Ada")
        assert "=== WHO YOU ARE ===" in out
        assert "You are a real trader, not a calculator" in out

    def test_trading_check_never_reaches_chat(self, env):
        assert "never explain a hold and then trade" not in build_chat_system_prompt(
            env, "Ada"
        )

    def test_custom_persona_survives(self):
        # The regression test for the rebuild trap: TraitAgent keeps only the
        # assembled disposition, so rebuilding it would drop this sentence.
        line = "I once blew up a fund in Osaka."
        env = make_env([
            make_agent("Ada", "aggressive", persona=line),
            make_agent("Nova", "greedy"),
        ])
        assert line in build_chat_system_prompt(env, "Ada")

    def test_relations_are_named(self, env):
        out = build_chat_system_prompt(env, "Ada")
        relations = agent_relations(env, "Ada")
        named = [relations.get("idol"), *relations.get("friends", []),
                 *relations.get("enemies", [])]
        named = [n for n in named if n]
        assert named, "fixture roster should resolve at least one relation"
        assert "=== THE PEOPLE AROUND YOU ===" in out
        for peer_id in named:
            assert peer_id in out

    def test_style_rules_are_the_tail(self, env):
        assert build_chat_system_prompt(env, "Ada").rstrip().endswith(
            "the numbers above are the current ones."
        )

    def test_unknown_agent_raises_value_error(self, env):
        with pytest.raises(ValueError):
            build_chat_system_prompt(env, "Nobody")


class TestMoodGating:
    def test_mood_shown_in_deep_mode(self, env):
        out = build_chat_system_prompt(env, "Ada")
        assert "=== HOW YOU FEEL RIGHT NOW ===" in out
        for axis in ("Confidence", "Stress", "Frustration"):
            assert axis in out

    def test_mood_hidden_in_simple_mode(self):
        env = make_env(
            [make_agent("Ada", "aggressive", deep=False),
             make_agent("Nova", "greedy", deep=False)],
            deep=False,
        )
        out = build_chat_system_prompt(env, "Ada")
        assert "HOW YOU FEEL" not in out
        # Identity and standing still render without deep mode.
        assert "=== WHO YOU ARE ===" in out
        assert "Standing: you are" in out


class TestPeerDescription:
    def test_peer_facts(self, env):
        env._recent_actions = {
            "Nova": {
                "action": "buy", "filled": 40,
                "reasoning": "Momentum is not done yet, I am adding.",
            }
        }
        line = describe_peer(env, "Nova", "Your idol")
        assert "Nova" in line
        assert "Greedy" in line              # personality label
        assert "of 3 at" in line             # rank / return
        assert "bought 40 shares" in line    # past tense, not "buy"
        assert "Momentum is not done yet" in line

    def test_quote_is_truncated(self, env):
        env._recent_actions = {
            "Nova": {"action": "sell", "filled": 5, "reasoning": "x" * 400}
        }
        line = describe_peer(env, "Nova", "Enemy")
        assert "..." in line
        assert len(line) < 400

    def test_peer_that_never_acted_still_renders(self, env):
        env._recent_actions = {}
        line = describe_peer(env, "Nova", "Friend")
        assert "Nova" in line
        assert "Last round" not in line

    def test_hold_reads_as_inaction(self, env):
        env._recent_actions = {"Nova": {"action": "hold", "filled": 0}}
        assert "sat on their hands" in describe_peer(env, "Nova", "Friend")

    def test_unknown_peer_yields_nothing(self, env):
        assert describe_peer(env, "Ghost", "Friend") == ""


class TestGracefulDegradation:
    def test_round_zero(self, env):
        out = build_chat_system_prompt(env, "Ada")
        assert env.step_count == 0
        assert "=== WHO YOU ARE ===" in out
        assert "RECENT DECISIONS" not in out
        assert "Market since Step" not in out

    def test_memory_disabled(self):
        base = ExternalAIAgent(
            "Ada", cash=10000, model="claude-sonnet-5", api_key="k",
            enable_memory=False,
        )
        agent = create_personality_agent(base, personality="aggressive", deep=True)
        env = make_env([agent, make_agent("Nova", "greedy")])
        base._short_term_memory = ["Step 1: BUY 10 TechTitan"]
        out = build_chat_system_prompt(env, "Ada")
        assert "RECENT DECISIONS" not in out
        assert "=== WHO YOU ARE ===" in out

    def test_empty_social_map_falls_back_to_recompute(self, env):
        env.social_map = {}
        # Recomputed from the roster, so relations still resolve.
        assert agent_relations(env, "Ada") == resolve_social_map(
            list(env.agents.values())
        )["Ada"]

    def test_agent_with_no_relations_drops_the_section(self):
        env = make_env([make_agent("Solo", "balanced")])
        out = build_chat_system_prompt(env, "Solo")
        assert "THE PEOPLE AROUND YOU" not in out
        assert "=== WHO YOU ARE ===" in out

    def test_player_agent_in_roster(self):
        env = make_env([make_agent("Ada", "aggressive"), PlayerAgent("You")])
        # A briefing for the player itself must not raise on missing renderers.
        assert "=== WHO YOU ARE ===" in build_chat_system_prompt(env, "You")
        assert build_chat_system_prompt(env, "Ada")

    def test_bare_external_agent_without_persona_layer(self):
        bare = ExternalAIAgent("Ada", cash=10000, model="claude-sonnet-5", api_key="k")
        env = make_env([bare])
        out = build_chat_system_prompt(env, "Ada")
        assert "You are 'Ada'" in out


class TestReadOnly:
    def test_no_state_is_mutated(self, env):
        target = env.agents["Ada"].base_agent
        target._short_term_memory = ["Step 1: BUY 10 TechTitan"]
        target._lessons = ["Do not chase headlines."]
        target._key_events = [{"step": 1, "name": "Ban", "stock": None, "impact": -0.1}]
        target._market_history = [(1, 150.0), (2, 158.0)]
        target._position_plans = {"TechTitan": {"stop_loss": 149.0}}
        env._recent_actions = {"Nova": {"action": "buy", "filled": 4, "reasoning": "x"}}

        watched = {
            "market_history": target._market_history,
            "conversation": target._conversation_history,
            "short_term": target._short_term_memory,
            "lessons": target._lessons,
            "plans": target._position_plans,
            "key_events": target._key_events,
            "mood": env.agents["Ada"].mood,
            "recent_actions": env._recent_actions,
        }
        before = copy.deepcopy(watched)
        steps_before = env.step_count

        build_chat_system_prompt(env, "Ada")
        build_chat_system_prompt(env, "Ada")  # twice: no accumulation either

        assert watched == before
        assert env.step_count == steps_before

    def test_observation_copy_is_not_shared(self, env):
        # The builder annotates its own copy; a later observation must be clean.
        build_chat_system_prompt(env, "Ada")
        obs = env.get_observation("Ada")
        assert "persona" not in obs


class TestSizeBudget:
    def test_briefing_is_capped_and_keeps_the_essentials(self, env):
        target = env.agents["Ada"].base_agent
        target._short_term_memory = [f"Step {i}: BUY 10 TechTitan " + "x" * 200
                                     for i in range(50)]
        target._key_events = [
            {"step": i, "name": "Event " + "y" * 200, "stock": "TechTitan",
             "impact": -0.1}
            for i in range(50)
        ]
        target._lessons = ["Lesson " + "z" * 200 for _ in range(50)]
        target._market_history = [(i, 150.0 + i) for i in range(50)]

        out = build_chat_system_prompt(env, "Ada")
        # The cap applies to the body; the reply rules are appended after
        # trimming so they can never be cut.
        body = out.split("=== HOW TO REPLY ===")[0].strip()
        assert len(body) <= _MAX_BRIEFING_CHARS
        # Sections were actually dropped, not merely under the limit anyway.
        assert "Market since Step" not in out
        assert "=== WHO YOU ARE ===" in out
        assert "=== WHERE YOU STAND ===" in out
        assert "=== THE PEOPLE AROUND YOU ===" in out
        assert out.rstrip().endswith("the numbers above are the current ones.")

    def test_max_peers_is_respected(self, env):
        out = build_chat_system_prompt(env, "Ada", max_peers=1)
        section = out.split("=== THE PEOPLE AROUND YOU ===")[1]
        section = section.split("===")[0]
        assert section.count("\n- ") <= 1
