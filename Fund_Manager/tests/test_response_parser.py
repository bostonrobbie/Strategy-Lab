"""
Tests for the response_parser module.
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from response_parser import (
    ResponseParser, AgentResponse, Decision, Severity,
    parse_agent_response, is_approved, is_vetoed, get_decision_with_confidence
)


class TestDecisionExtraction:
    """Tests for decision extraction from responses."""

    def test_approved_at_start(self):
        """Test APPROVED decision at start of response."""
        response = "APPROVED: Strategy meets all risk criteria. Sharpe ratio of 1.8 is excellent."
        result = ResponseParser.parse(response)
        assert result.decision == Decision.APPROVED
        assert result.confidence > 0.8

    def test_veto_at_start(self):
        """Test VETO decision at start of response."""
        response = "VETO: Max drawdown of 35% exceeds the 25% limit."
        result = ResponseParser.parse(response)
        assert result.decision == Decision.VETOED
        assert result.confidence > 0.8

    def test_approved_in_middle(self):
        """Test APPROVED decision in middle of response."""
        response = "After careful review of the metrics, I conclude APPROVED. The strategy is sound."
        result = ResponseParser.parse(response)
        assert result.decision == Decision.APPROVED

    def test_compliant_decision(self):
        """Test COMPLIANT compliance decision."""
        response = "[COMPLIANT]: Strategy passes all regulatory checks."
        result = ResponseParser.parse(response)
        assert result.decision == Decision.COMPLIANT

    def test_violation_decision(self):
        """Test VIOLATION compliance decision."""
        response = "[VIOLATION]: Strategy shows signs of market manipulation."
        result = ResponseParser.parse(response)
        assert result.decision == Decision.VIOLATION

    def test_concern_decision(self):
        """Test CONCERN decision."""
        response = "[CONCERN]: There are some issues that need review before approval."
        result = ResponseParser.parse(response)
        assert result.decision == Decision.CONCERN

    def test_unknown_decision(self):
        """Test unknown decision when no markers present."""
        response = "The strategy looks interesting but I need more data to decide."
        result = ResponseParser.parse(response)
        assert result.decision == Decision.UNKNOWN

    def test_empty_response(self):
        """Test handling of empty response."""
        result = ResponseParser.parse("")
        assert result.decision == Decision.UNKNOWN
        assert result.confidence == 0.0

    def test_none_response(self):
        """Test handling of None response."""
        result = ResponseParser.parse(None)
        assert result.decision == Decision.UNKNOWN


class TestIssueExtraction:
    """Tests for issue extraction from code review responses."""

    def test_bug_extraction(self):
        """Test extraction of BUG issues."""
        response = """
        Code review complete.
        [BUG]: Look-ahead bias in line 42
        [BUG]: Missing null check for data
        """
        result = ResponseParser.parse(response)
        bugs = [i for i in result.issues if i["severity"] == "bug"]
        assert len(bugs) == 2
        assert "Look-ahead bias" in bugs[0]["description"]

    def test_warning_extraction(self):
        """Test extraction of WARNING issues."""
        response = "[WARNING]: Variable name 'x' is not descriptive"
        result = ResponseParser.parse(response)
        warnings = [i for i in result.issues if i["severity"] == "warning"]
        assert len(warnings) == 1

    def test_mixed_issues(self):
        """Test extraction of mixed severity issues."""
        response = """
        [BUG]: Critical issue with order logic
        [WARNING]: Potential edge case not handled
        [OPTIMIZE]: Could use vectorized operations
        [STYLE]: Consider using f-strings
        """
        result = ResponseParser.parse(response)
        assert len(result.issues) == 4

        severities = [i["severity"] for i in result.issues]
        assert "bug" in severities
        assert "warning" in severities
        assert "optimize" in severities
        assert "style" in severities


class TestRecommendationExtraction:
    """Tests for recommendation extraction."""

    def test_recommendation_extraction(self):
        """Test extraction of recommendations."""
        response = """
        [RECOMMENDATION]: Add stop-loss at 2% below entry
        [INSIGHT]: Strategy performs best in trending markets
        """
        result = ResponseParser.parse(response)
        assert len(result.recommendations) >= 2

    def test_insight_extraction(self):
        """Test extraction of insights."""
        response = "[INSIGHT]: Win rate drops significantly during low volatility periods"
        result = ResponseParser.parse(response)
        assert len(result.recommendations) == 1
        assert "Win rate" in result.recommendations[0]


class TestRequestExtraction:
    """Tests for request extraction."""

    def test_typed_request_extraction(self):
        """Test extraction of typed requests."""
        response = "[REQUEST: data - Need historical volatility data for the past 5 years]"
        result = ResponseParser.parse(response)
        assert len(result.requests) == 1
        assert result.requests[0]["type"] == "data"

    def test_data_request_extraction(self):
        """Test extraction of data requests."""
        response = "[DATA REQUEST]: Need intraday tick data for futures"
        result = ResponseParser.parse(response)
        assert len(result.requests) == 1

    def test_feature_request_extraction(self):
        """Test extraction of feature requests."""
        response = "[FEATURE]: Add regime detection indicator"
        result = ResponseParser.parse(response)
        assert len(result.requests) == 1


class TestMetricsExtraction:
    """Tests for numerical metrics extraction."""

    def test_sharpe_extraction(self):
        """Test extraction of Sharpe ratio."""
        response = "The strategy has a Sharpe Ratio: 1.85 which is excellent."
        result = ResponseParser.parse(response)
        assert "sharpe" in result.metadata
        assert result.metadata["sharpe"] == 1.85

    def test_drawdown_extraction(self):
        """Test extraction of max drawdown."""
        response = "Max Drawdown: -18.5% is within acceptable limits."
        result = ResponseParser.parse(response)
        assert "max_drawdown" in result.metadata

    def test_trade_count_extraction(self):
        """Test extraction of trade count."""
        response = "Total trades: 127 over the backtest period."
        result = ResponseParser.parse(response)
        assert "trade_count" in result.metadata
        assert result.metadata["trade_count"] == 127

    def test_multiple_metrics(self):
        """Test extraction of multiple metrics."""
        response = """
        Sharpe Ratio: 1.5
        Win Rate: 62%
        Profit Factor: 1.8
        Total Trades: 95
        Max Drawdown: -12%
        """
        result = ResponseParser.parse(response)
        assert len(result.metadata) >= 4


class TestRiskReviewParsing:
    """Tests for risk manager response parsing."""

    def test_risk_approval(self):
        """Test parsing risk manager approval."""
        response = """
        APPROVED: After thorough review of the metrics:
        - Sharpe Ratio: 1.8 (above minimum 1.0)
        - Max Drawdown: -15% (below 25% limit)
        - Trade Count: 85 (sufficient sample size)
        - Profit Factor: 1.6 (above 1.2 minimum)

        The strategy passes all risk criteria.
        """
        result = ResponseParser.parse_risk_review(response)
        assert result.decision == Decision.APPROVED
        assert result.confidence >= 0.9
        assert "sharpe" in result.metadata

    def test_risk_veto(self):
        """Test parsing risk manager veto."""
        response = """
        VETO: The strategy fails multiple risk criteria:
        - Max Drawdown: 35% exceeds the 25% limit
        - Sharpe Ratio: 0.8 is below minimum 1.0
        - Signs of overfitting detected in the magic numbers used
        """
        result = ResponseParser.parse_risk_review(response)
        assert result.decision == Decision.VETOED
        assert result.confidence >= 0.9
        assert result.metadata.get("has_overfitting_concern") is True

    def test_data_concern_detection(self):
        """Test detection of data concerns."""
        response = "VETO: There appears to be look-ahead bias in the signal calculation."
        result = ResponseParser.parse_risk_review(response)
        assert result.metadata.get("has_data_concern") is True


class TestComplianceReviewParsing:
    """Tests for compliance officer response parsing."""

    def test_compliance_compliant(self):
        """Test parsing compliant response."""
        response = "[COMPLIANT]: Strategy passes all regulatory and ethical checks."
        result = ResponseParser.parse_compliance_review(response)
        assert result.decision == Decision.COMPLIANT
        assert result.confidence >= 0.9

    def test_compliance_violation(self):
        """Test parsing violation response."""
        response = "[VIOLATION]: Strategy exhibits patterns consistent with market manipulation."
        result = ResponseParser.parse_compliance_review(response)
        assert result.decision == Decision.VIOLATION
        assert result.confidence >= 0.9

    def test_compliance_concern(self):
        """Test parsing concern response."""
        response = "[CONCERN]: Position concentration may exceed limits during volatile periods."
        result = ResponseParser.parse_compliance_review(response)
        assert result.decision == Decision.CONCERN


class TestCodeReviewParsing:
    """Tests for code reviewer response parsing."""

    def test_code_review_with_bugs(self):
        """Test code review with bugs results in veto."""
        response = """
        [BUG]: Critical issue - using future data in calculations
        [WARNING]: Variable shadowing in inner loop
        """
        result = ResponseParser.parse_code_review(response)
        assert result.decision == Decision.VETOED

    def test_code_review_warnings_only(self):
        """Test code review with only warnings results in concern."""
        response = """
        [WARNING]: Consider adding type hints
        [STYLE]: Function name could be more descriptive
        """
        result = ResponseParser.parse_code_review(response)
        assert result.decision == Decision.CONCERN

    def test_code_review_clean(self):
        """Test clean code review results in approval."""
        response = "Code looks clean and follows best practices. No issues found."
        result = ResponseParser.parse_code_review(response)
        assert result.decision == Decision.APPROVED


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_is_approved_true(self):
        """Test is_approved returns True for approval."""
        assert is_approved("APPROVED: Strategy looks good.") is True
        assert is_approved("[COMPLIANT]: Passes all checks.") is True

    def test_is_approved_false(self):
        """Test is_approved returns False for non-approval."""
        assert is_approved("VETO: Fails risk criteria.") is False
        assert is_approved("I need more information.") is False

    def test_is_vetoed_true(self):
        """Test is_vetoed returns True for veto."""
        assert is_vetoed("VETO: Max drawdown too high.") is True
        assert is_vetoed("[VIOLATION]: Market manipulation.") is True

    def test_is_vetoed_false(self):
        """Test is_vetoed returns False for non-veto."""
        assert is_vetoed("APPROVED: Looks good.") is False
        assert is_vetoed("Needs more review.") is False

    def test_get_decision_with_confidence(self):
        """Test get_decision_with_confidence returns tuple."""
        decision, confidence = get_decision_with_confidence("APPROVED: Strategy is sound.")
        assert decision == "approved"
        assert 0.0 <= confidence <= 1.0

    def test_parse_agent_response_risk(self):
        """Test parse_agent_response with risk type."""
        result = parse_agent_response("VETO: Too risky.", agent_type="risk")
        assert result.decision == Decision.VETOED
        assert result.confidence >= 0.9

    def test_parse_agent_response_compliance(self):
        """Test parse_agent_response with compliance type."""
        result = parse_agent_response("[COMPLIANT]: All good.", agent_type="compliance")
        assert result.decision == Decision.COMPLIANT

    def test_parse_agent_response_code(self):
        """Test parse_agent_response with code type."""
        result = parse_agent_response("[BUG]: Critical bug found.", agent_type="code")
        assert result.decision == Decision.VETOED


class TestAgentResponseMethods:
    """Tests for AgentResponse methods."""

    def test_is_approved_method(self):
        """Test AgentResponse.is_approved method."""
        response = AgentResponse(decision=Decision.APPROVED)
        assert response.is_approved() is True

        response = AgentResponse(decision=Decision.COMPLIANT)
        assert response.is_approved() is True

        response = AgentResponse(decision=Decision.VETOED)
        assert response.is_approved() is False

    def test_is_rejected_method(self):
        """Test AgentResponse.is_rejected method."""
        response = AgentResponse(decision=Decision.VETOED)
        assert response.is_rejected() is True

        response = AgentResponse(decision=Decision.VIOLATION)
        assert response.is_rejected() is True

        response = AgentResponse(decision=Decision.APPROVED)
        assert response.is_rejected() is False

    def test_has_concerns_method(self):
        """Test AgentResponse.has_concerns method."""
        response = AgentResponse(
            decision=Decision.CONCERN,
            issues=[{"severity": "warning", "description": "Minor issue"}]
        )
        assert response.has_concerns() is True

        response = AgentResponse(decision=Decision.APPROVED, issues=[])
        assert response.has_concerns() is False

    def test_to_dict_method(self):
        """Test AgentResponse.to_dict method."""
        response = AgentResponse(
            decision=Decision.APPROVED,
            confidence=0.95,
            reasons=["Good metrics"],
            issues=[],
            recommendations=["Consider more testing"]
        )
        result = response.to_dict()
        assert result["decision"] == "approved"
        assert result["confidence"] == 0.95
        assert "Good metrics" in result["reasons"]
