"""Tests for the expanded market events system (crypto, M&A, corporate)."""

from ai_trading_society.market_events import (
    EVENT_TEMPLATES,
    EventManager,
    EventType,
)


class TestNewEventCategories:
    """Verify the 3 new event categories and 13 new templates."""

    def test_new_event_types_exist(self):
        assert EventType.CRYPTO.value == "crypto"
        assert EventType.M_AND_A.value == "m_and_a"
        assert EventType.CORPORATE.value == "corporate"

    def test_total_template_count(self):
        # 28 original + 13 new = 41
        assert len(EVENT_TEMPLATES) == 41

    def test_crypto_templates(self):
        crypto = [t for t in EVENT_TEMPLATES if t.event_type == EventType.CRYPTO]
        names = {t.name for t in crypto}
        assert names == {
            "crypto_crash",
            "crypto_rally",
            "crypto_etf_approval",
            "crypto_exchange_collapse",
        }
        # Crash and exchange collapse should be bearish
        assert crypto[0].price_impact < 0  # crypto_crash
        assert crypto[3].price_impact < 0  # crypto_exchange_collapse

    def test_m_and_a_templates(self):
        ma = [t for t in EVENT_TEMPLATES if t.event_type == EventType.M_AND_A]
        names = {t.name for t in ma}
        assert names == {
            "acquisition_announced",
            "merger_rumor",
            "hostile_takeover_bid",
            "deal_breakup",
        }
        # deal_breakup should be bearish
        deal = [t for t in ma if t.name == "deal_breakup"][0]
        assert deal.price_impact < 0
        # acquisition_announced should have the highest bullish impact
        acq = [t for t in ma if t.name == "acquisition_announced"][0]
        assert acq.price_impact == 0.15

    def test_corporate_templates(self):
        corp = [t for t in EVENT_TEMPLATES if t.event_type == EventType.CORPORATE]
        names = {t.name for t in corp}
        assert names == {
            "stock_buyback",
            "dividend_increase",
            "stock_split",
            "spinoff_announced",
            "accounting_scandal",
        }
        # accounting_scandal should be strongly bearish
        scandal = [t for t in corp if t.name == "accounting_scandal"][0]
        assert scandal.price_impact == -0.12
        assert scandal.sentiment_shift == -0.65


class TestNewEventTriggering:
    """Verify new templates can be triggered through EventManager."""

    def test_crypto_crash_can_trigger(self):
        mgr = EventManager(templates=EVENT_TEMPLATES, event_probability_multiplier=1.0)
        crypto_crash = [t for t in mgr.templates if t.name == "crypto_crash"][0]
        event = mgr.force_trigger_event("crypto_crash", step=1)
        assert event.event_type == EventType.CRYPTO
        assert event.price_impact == crypto_crash.price_impact
        assert event.is_active()

    def test_acquisition_announced_can_trigger(self):
        mgr = EventManager(templates=EVENT_TEMPLATES, event_probability_multiplier=1.0)
        event = mgr.force_trigger_event("acquisition_announced", step=1)
        assert event.event_type == EventType.M_AND_A
        assert event.price_impact == 0.15
        assert event.is_active()

    def test_stock_buyback_can_trigger(self):
        mgr = EventManager(templates=EVENT_TEMPLATES, event_probability_multiplier=1.0)
        event = mgr.force_trigger_event("stock_buyback", step=1)
        assert event.event_type == EventType.CORPORATE
        assert event.price_impact == 0.05
        assert event.is_active()

    def test_all_new_templates_triggerable(self):
        """Every new template should be force-triggerable by name."""
        mgr = EventManager(templates=EVENT_TEMPLATES, event_probability_multiplier=1.0)
        new_names = [
            "crypto_crash", "crypto_rally", "crypto_etf_approval",
            "crypto_exchange_collapse",
            "acquisition_announced", "merger_rumor", "hostile_takeover_bid",
            "deal_breakup",
            "stock_buyback", "dividend_increase", "stock_split",
            "spinoff_announced", "accounting_scandal",
        ]
        for name in new_names:
            event = mgr.force_trigger_event(name, step=1)
            assert event.name == name
            assert event.is_active()

    def test_new_events_appear_in_observation(self):
        mgr = EventManager(templates=EVENT_TEMPLATES, event_probability_multiplier=1.0)
        mgr.force_trigger_event("crypto_crash", step=1)
        mgr.force_trigger_event("stock_buyback", step=1)
        obs = mgr.get_observation_data()
        active_names = {e["name"] for e in obs["active_events"]}
        assert "crypto_crash" in active_names
        assert "stock_buyback" in active_names
        assert obs["market_sentiment"] != 0.0


class TestEventImpactBalance:
    """Verify the new templates have reasonable impact ranges."""

    def test_all_new_impacts_within_bounds(self):
        new_types = {EventType.CRYPTO, EventType.M_AND_A, EventType.CORPORATE}
        for t in EVENT_TEMPLATES:
            if t.event_type in new_types:
                assert -0.15 <= t.price_impact <= 0.15
                assert -0.7 <= t.sentiment_shift <= 0.6
                assert 0 < t.probability <= 0.02
                assert 1 <= t.duration_steps <= 6

    def test_each_new_category_has_bullish_and_bearish(self):
        for et in (EventType.CRYPTO, EventType.M_AND_A, EventType.CORPORATE):
            templates = [t for t in EVENT_TEMPLATES if t.event_type == et]
            bullish = [t for t in templates if t.price_impact > 0]
            bearish = [t for t in templates if t.price_impact < 0]
            assert len(bullish) >= 1, f"{et.value} has no bullish event"
            assert len(bearish) >= 1, f"{et.value} has no bearish event"
