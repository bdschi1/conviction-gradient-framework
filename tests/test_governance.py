"""Tests for IC governance layer."""

from datetime import date

import pytest

from governance.debate import DebateRecord, extract_its_from_debate_records
from governance.policies import ProcessPolicy, enforce_policy
from governance.session import SessionManager, SessionStatus

# --- Session Manager ---


class TestSessionManager:
    def test_create_session(self):
        mgr = SessionManager()
        session = mgr.create_session(
            instrument_id="AAPL",
            session_date=date(2024, 6, 1),
            participants=["analyst1", "analyst2", "analyst3"],
        )
        assert session.instrument_id == "AAPL"
        assert session.status == SessionStatus.PRE_DEBATE
        assert len(session.participants) == 3

    def test_full_lifecycle(self):
        mgr = SessionManager()
        session = mgr.create_session(
            instrument_id="AAPL",
            session_date=date(2024, 6, 1),
            participants=["a1", "a2"],
        )
        sid = session.session_id

        # Submit pre-debate probabilities
        mgr.submit_pre_probability(sid, "a1", 0.7)
        mgr.submit_pre_probability(sid, "a2", 0.6)

        # Start debate
        mgr.start_debate(sid)
        assert mgr.get_session(sid).status == SessionStatus.DEBATING

        # End debate
        mgr.end_debate(sid)
        assert mgr.get_session(sid).status == SessionStatus.POST_DEBATE

        # Submit post-debate probabilities
        mgr.submit_post_probability(sid, "a1", 0.5)
        mgr.submit_post_probability(sid, "a2", 0.55)

        # Finalize and compute ITS
        its = mgr.finalize(sid)
        # mean_post - mean_pre = 0.525 - 0.65 = -0.125
        assert its == pytest.approx(-0.125)
        assert mgr.get_session(sid).status == SessionStatus.FINALIZED

    def test_invalid_state_transition(self):
        mgr = SessionManager()
        session = mgr.create_session(
            instrument_id="AAPL",
            session_date=date(2024, 6, 1),
            participants=["a1"],
        )
        with pytest.raises(ValueError, match="not in DEBATING"):
            mgr.end_debate(session.session_id)

    def test_start_debate_without_submissions(self):
        mgr = SessionManager()
        session = mgr.create_session(
            instrument_id="AAPL",
            session_date=date(2024, 6, 1),
            participants=["a1"],
        )
        with pytest.raises(ValueError, match="No pre-debate"):
            mgr.start_debate(session.session_id)

    def test_session_not_found(self):
        mgr = SessionManager()
        with pytest.raises(KeyError):
            mgr.get_session("nonexistent")


# --- Debate Records ---


class TestDebateRecords:
    def test_extract_its(self):
        records = [
            DebateRecord(session_id="s1", pre_conviction=7.0, post_conviction=5.0),
            DebateRecord(session_id="s1", pre_conviction=6.0, post_conviction=4.0),
        ]
        its = extract_its_from_debate_records(records)
        # mean_post - mean_pre = 4.5 - 6.5 = -2.0
        assert its == pytest.approx(-2.0)

    def test_empty_records(self):
        assert extract_its_from_debate_records([]) == 0.0

    def test_missing_scores(self):
        records = [DebateRecord(session_id="s1")]
        assert extract_its_from_debate_records(records) == 0.0


# --- Policies ---


class TestPolicies:
    def test_no_violations(self):
        policy = ProcessPolicy()
        violations = enforce_policy(
            policy, "AAPL",
            fvs=0.3, fe=1.0, vol_ratio=1.2,
            has_adversarial_debate=True, participant_count=3,
        )
        assert len(violations) == 0

    def test_missing_debate(self):
        policy = ProcessPolicy(mandatory_adversarial=True)
        violations = enforce_policy(
            policy, "AAPL",
            has_adversarial_debate=False, participant_count=3,
        )
        assert any(v.policy_name == "mandatory_adversarial" for v in violations)
        assert any(v.severity == "block" for v in violations)

    def test_fvs_reset(self):
        policy = ProcessPolicy(fvs_reset_threshold=0.7)
        violations = enforce_policy(
            policy, "AAPL", fvs=0.8,
            has_adversarial_debate=True, participant_count=3,
        )
        assert any(v.policy_name == "fvs_reset" for v in violations)

    def test_vol_reset(self):
        policy = ProcessPolicy(vol_double_trigger=2.0)
        violations = enforce_policy(
            policy, "AAPL", vol_ratio=2.5,
            has_adversarial_debate=True, participant_count=3,
        )
        assert any(v.policy_name == "vol_reset" for v in violations)

    def test_fe_warning(self):
        policy = ProcessPolicy(min_fe_std_trigger=2.0)
        violations = enforce_policy(
            policy, "AAPL", fe=3.0,
            has_adversarial_debate=True, participant_count=3,
        )
        assert any(v.policy_name == "fe_review" for v in violations)

    def test_insufficient_participants(self):
        policy = ProcessPolicy(min_participants_for_its=3)
        violations = enforce_policy(
            policy, "AAPL",
            has_adversarial_debate=True, participant_count=1,
        )
        assert any(v.policy_name == "insufficient_participants" for v in violations)
