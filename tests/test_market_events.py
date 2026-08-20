"""Tests for market events: multi-trigger, lifecycle, combined effects."""


from ai_trading_society.market_events import (
    EventManager,
    EventType,
    MarketEvent,
)


class TestMultiTrigger:
    """Test that multiple events can trigger in a single step."""

    def test_returns_list_not_single(self):
        """try_trigger_event should return a list, not a single event or None."""
        em = EventManager(event_probability_multiplier=0.0)  # No events
        result = em.try_trigger_event(step=1)
        assert isinstance(result, list), "Should return a list"

    def test_empty_list_when_no_trigger(self):
        """Should return empty list when no events trigger."""
        em = EventManager(event_probability_multiplier=0.0)
        result = em.try_trigger_event(step=1)
        assert len(result) == 0

    def test_multiple_events_can_trigger(self):
        """With high probability, multiple events should trigger in one step."""
        # Create templates with probability 1.0 so they always trigger
        templates = [
            MarketEvent(
                name=f"event_{i}",
                description=f"Test event {i}",
                event_type=EventType.SOCIAL,
                price_impact=0.01,
                sentiment_shift=0.1,
                duration_steps=2,
                probability=1.0,
            )
            for i in range(5)
        ]
        em = EventManager(templates=templates, event_probability_multiplier=1.0)
        triggered = em.try_trigger_event(step=1)
        assert len(triggered) == 5, "All 5 events should trigger with probability=1.0"
        assert all(isinstance(e, MarketEvent) for e in triggered)

    def test_all_triggered_events_added_to_history(self):
        """All triggered events should be recorded in event_history."""
        templates = [
            MarketEvent(
                name=f"event_{i}",
                description=f"Test event {i}",
                event_type=EventType.EARNINGS,
                price_impact=0.01,
                duration_steps=1,
                probability=1.0,
            )
            for i in range(3)
        ]
        em = EventManager(templates=templates, event_probability_multiplier=1.0)
        em.try_trigger_event(step=1)
        assert len(em.event_history) == 3, "All 3 events should be in history"

    def test_all_triggered_events_added_to_active(self):
        """All triggered events should be added to active_events."""
        templates = [
            MarketEvent(
                name=f"event_{i}",
                description=f"Test event {i}",
                event_type=EventType.MACRO,
                duration_steps=3,
                probability=1.0,
            )
            for i in range(4)
        ]
        em = EventManager(templates=templates, event_probability_multiplier=1.0)
        em.try_trigger_event(step=1)
        assert len(em.active_events) == 4, "All 4 events should be active"


class TestEventLifecycle:
    """Test event trigger, tick, and expiration."""

    def test_event_is_active_after_trigger(self):
        """Event should be active immediately after triggering."""
        event = MarketEvent(
            name="test",
            description="Test event",
            event_type=EventType.EARNINGS,
            duration_steps=3,
        )
        event.trigger(step=1)
        assert event.is_active()
        assert event.remaining_steps == 3

    def test_event_expires_after_duration(self):
        """Event should expire after duration_steps ticks."""
        event = MarketEvent(
            name="test",
            description="Test event",
            event_type=EventType.EARNINGS,
            duration_steps=2,
        )
        event.trigger(step=1)
        assert event.is_active()
        event.tick()
        assert event.is_active()
        assert event.remaining_steps == 1
        event.tick()
        assert not event.is_active()
        assert event.remaining_steps == 0

    def test_tick_removes_expired_events(self):
        """EventManager.tick should remove expired events from active list."""
        templates = [
            MarketEvent(
                name="short_event",
                description="Short event",
                event_type=EventType.SOCIAL,
                duration_steps=1,
                probability=1.0,
            )
        ]
        em = EventManager(templates=templates, event_probability_multiplier=1.0)
        em.try_trigger_event(step=1)
        assert len(em.active_events) == 1
        em.tick()  # This should expire the event
        assert len(em.active_events) == 0


class TestCombinedEffects:
    """Test combined effects from multiple active events."""

    def test_combined_price_impact(self):
        """Combined price impact should sum from all active events."""
        templates = [
            MarketEvent(
                name="bullish",
                description="Bullish event",
                event_type=EventType.EARNINGS,
                price_impact=0.05,
                sentiment_shift=0.3,
                duration_steps=5,
                probability=1.0,
            ),
            MarketEvent(
                name="bearish",
                description="Bearish event",
                event_type=EventType.MACRO,
                price_impact=-0.03,
                sentiment_shift=-0.2,
                duration_steps=5,
                probability=1.0,
            ),
        ]
        em = EventManager(templates=templates, event_probability_multiplier=1.0)
        em.try_trigger_event(step=1)
        effects = em.get_combined_effects()
        # Both impacts should contribute (scaled by decay * 0.3)
        # decay = remaining/duration = 5/5 = 1.0 for both
        # bullish: 0.05 * 1.0 * 0.3 = 0.015
        # bearish: -0.03 * 1.0 * 0.3 = -0.009
        # combined: 0.015 - 0.009 = 0.006
        assert effects["price_impact"] > 0, "Net impact should be positive (bullish > bearish)"

    def test_sentiment_clamped_to_range(self):
        """Sentiment shift should be clamped to [-1, 1]."""
        templates = [
            MarketEvent(
                name=f"bullish_{i}",
                description="Very bullish",
                event_type=EventType.SOCIAL,
                price_impact=0.01,
                sentiment_shift=0.5,
                duration_steps=5,
                probability=1.0,
            )
            for i in range(10)
        ]
        em = EventManager(templates=templates, event_probability_multiplier=1.0)
        em.try_trigger_event(step=1)
        effects = em.get_combined_effects()
        assert effects["sentiment_shift"] <= 1.0, "Sentiment should be clamped to 1.0"


class TestObservationData:
    """Test observation data generation."""

    def test_observation_data_contains_sentiment(self):
        """get_observation_data should include market_sentiment."""
        em = EventManager(event_probability_multiplier=0.0)
        data = em.get_observation_data()
        assert "market_sentiment" in data
        assert "active_events" in data
        assert "event_price_impact" in data

    def test_observation_data_shows_active_events(self):
        """Active events should appear in observation data."""
        templates = [
            MarketEvent(
                name="test_event",
                description="A test event",
                event_type=EventType.EARNINGS,
                duration_steps=5,
                probability=1.0,
            )
        ]
        em = EventManager(templates=templates, event_probability_multiplier=1.0)
        em.try_trigger_event(step=1)
        data = em.get_observation_data()
        assert len(data["active_events"]) == 1
        assert data["active_events"][0]["name"] == "test_event"


class TestActiveEventsDetail:
    """Test the get_active_events_detail method for display."""

    def test_empty_when_no_active_events(self):
        """Should return empty list when no events are active."""
        em = EventManager(event_probability_multiplier=0.0)
        result = em.get_active_events_detail()
        assert isinstance(result, list)
        assert len(result) == 0

    def test_returns_detail_with_remaining_steps(self):
        """Should include remaining_steps and total_steps for each active event."""
        templates = [
            MarketEvent(
                name="test_event",
                description="A test event",
                event_type=EventType.EARNINGS,
                duration_steps=3,
                probability=1.0,
            )
        ]
        em = EventManager(templates=templates, event_probability_multiplier=1.0)
        em.try_trigger_event(step=1)
        detail = em.get_active_events_detail()
        assert len(detail) == 1
        assert detail[0]["name"] == "test_event"
        assert detail[0]["remaining_steps"] == 3
        assert detail[0]["total_steps"] == 3
        assert "description" in detail[0]
        assert "type" in detail[0]

    def test_remaining_steps_decreases_after_tick(self):
        """remaining_steps should decrease after tick()."""
        templates = [
            MarketEvent(
                name="test_event",
                description="A test event",
                event_type=EventType.EARNINGS,
                duration_steps=3,
                probability=1.0,
            )
        ]
        em = EventManager(templates=templates, event_probability_multiplier=1.0)
        em.try_trigger_event(step=1)
        em.tick()
        detail = em.get_active_events_detail()
        assert len(detail) == 1
        assert detail[0]["remaining_steps"] == 2

    def test_expired_events_not_in_detail(self):
        """Expired events should not appear in detail."""
        templates = [
            MarketEvent(
                name="short_event",
                description="Short event",
                event_type=EventType.SOCIAL,
                duration_steps=1,
                probability=1.0,
            )
        ]
        em = EventManager(templates=templates, event_probability_multiplier=1.0)
        em.try_trigger_event(step=1)
        em.tick()  # Expires the event
        detail = em.get_active_events_detail()
        assert len(detail) == 0
