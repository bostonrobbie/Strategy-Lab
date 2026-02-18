"""
Structured response parsing for LLM agent outputs.

This module provides parsing utilities to extract structured data from
free-text LLM responses, replacing brittle string matching with robust
pattern extraction and confidence scoring.
"""
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum


class Decision(Enum):
    """Possible decision outcomes from agents."""
    APPROVED = "approved"
    VETOED = "vetoed"
    PENDING = "pending"
    COMPLIANT = "compliant"
    VIOLATION = "violation"
    CONCERN = "concern"
    UNKNOWN = "unknown"


class Severity(Enum):
    """Issue severity levels."""
    BUG = "bug"
    WARNING = "warning"
    OPTIMIZE = "optimize"
    STYLE = "style"
    INFO = "info"


@dataclass
class AgentResponse:
    """
    Structured representation of an agent's response.

    Attributes:
        decision: The primary decision (APPROVED, VETOED, etc.)
        confidence: Confidence score from 0.0 to 1.0
        reasons: List of reasons supporting the decision
        issues: List of issues found
        recommendations: List of actionable recommendations
        requests: List of resource/improvement requests
        raw_response: The original unprocessed response text
        metadata: Additional extracted metadata
    """
    decision: Decision = Decision.UNKNOWN
    confidence: float = 0.0
    reasons: List[str] = field(default_factory=list)
    issues: List[Dict[str, str]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    requests: List[Dict[str, str]] = field(default_factory=list)
    raw_response: str = ""
    metadata: Dict[str, any] = field(default_factory=dict)

    def is_approved(self) -> bool:
        """Check if the decision is an approval."""
        return self.decision in (Decision.APPROVED, Decision.COMPLIANT)

    def is_rejected(self) -> bool:
        """Check if the decision is a rejection."""
        return self.decision in (Decision.VETOED, Decision.VIOLATION)

    def has_concerns(self) -> bool:
        """Check if there are concerns or issues."""
        return len(self.issues) > 0 or self.decision == Decision.CONCERN

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "decision": self.decision.value,
            "confidence": self.confidence,
            "reasons": self.reasons,
            "issues": self.issues,
            "recommendations": self.recommendations,
            "requests": self.requests,
            "metadata": self.metadata
        }


class ResponseParser:
    """
    Parser for extracting structured data from agent responses.

    Handles various response formats from different agent types:
    - Risk Manager: VETO/APPROVED decisions
    - Compliance Officer: COMPLIANT/CONCERN/VIOLATION
    - Code Reviewer: BUG/WARNING/OPTIMIZE/STYLE issues
    - Performance Analyst: INSIGHT/RECOMMENDATION
    - Portfolio Architect: PORTFOLIO UPDATE recommendations
    """

    # Decision patterns with their mapping
    DECISION_PATTERNS = {
        # Approval patterns (order matters - more specific first)
        r'\bAPPROVED\b': Decision.APPROVED,
        r'\bAPPROVE\b': Decision.APPROVED,
        r'\b\[APPROVED\]': Decision.APPROVED,
        r'\bCOMPLIANT\b': Decision.COMPLIANT,
        r'\b\[COMPLIANT\]': Decision.COMPLIANT,

        # Rejection patterns
        r'\bVETO\b': Decision.VETOED,
        r'\bVETOED\b': Decision.VETOED,
        r'\b\[VETO\]': Decision.VETOED,
        r'\bVIOLATION\b': Decision.VIOLATION,
        r'\b\[VIOLATION\]': Decision.VIOLATION,
        r'\bREJECT\b': Decision.VETOED,
        r'\bREJECTED\b': Decision.VETOED,

        # Concern/pending patterns
        r'\bCONCERN\b': Decision.CONCERN,
        r'\b\[CONCERN\]': Decision.CONCERN,
        r'\bPENDING\b': Decision.PENDING,
        r'\bNEEDS REVIEW\b': Decision.PENDING,
    }

    # Issue severity patterns (colon is optional)
    ISSUE_PATTERNS = {
        r'\[BUG\]:?\s*(.+?)(?:\n|$)': Severity.BUG,
        r'\[WARNING\]:?\s*(.+?)(?:\n|$)': Severity.WARNING,
        r'\[OPTIMIZE\]:?\s*(.+?)(?:\n|$)': Severity.OPTIMIZE,
        r'\[STYLE\]:?\s*(.+?)(?:\n|$)': Severity.STYLE,
    }

    # Request patterns
    REQUEST_PATTERNS = [
        r'\[REQUEST:\s*([^\]]+)\s*-\s*([^\]]+)\]',
        r'\[DATA REQUEST\]:\s*(.+?)(?:\n|$)',
        r'\[FEATURE\]:\s*(.+?)(?:\n|$)',
    ]

    # Recommendation patterns
    RECOMMENDATION_PATTERNS = [
        r'\[RECOMMENDATION\]:\s*(.+?)(?:\n|$)',
        r'\[INSIGHT\]:\s*(.+?)(?:\n|$)',
        r'\[SUGGESTION\]:\s*(.+?)(?:\n|$)',
    ]

    @classmethod
    def parse(cls, response: str) -> AgentResponse:
        """
        Parse a raw agent response into a structured AgentResponse.

        Args:
            response: Raw text response from an agent

        Returns:
            AgentResponse with extracted structured data
        """
        if not response:
            return AgentResponse(
                decision=Decision.UNKNOWN,
                confidence=0.0,
                raw_response=""
            )

        result = AgentResponse(raw_response=response)

        # Extract decision
        result.decision, result.confidence = cls._extract_decision(response)

        # Extract reasons (usually follows decision keyword)
        result.reasons = cls._extract_reasons(response, result.decision)

        # Extract issues
        result.issues = cls._extract_issues(response)

        # Extract recommendations
        result.recommendations = cls._extract_recommendations(response)

        # Extract requests
        result.requests = cls._extract_requests(response)

        # Extract any metrics mentioned
        result.metadata = cls._extract_metrics(response)

        return result

    @classmethod
    def parse_risk_review(cls, response: str) -> AgentResponse:
        """
        Parse a response specifically from the Risk Manager.

        Expects format like:
        "VETO: Reason 1. Reason 2."
        or
        "APPROVED: Risk assessment details."
        """
        result = cls.parse(response)

        # Risk manager should have high confidence in explicit decisions
        if result.decision in (Decision.APPROVED, Decision.VETOED):
            result.confidence = max(result.confidence, 0.9)

        # Extract specific risk metrics
        risk_metrics = cls._extract_risk_metrics(response)
        result.metadata.update(risk_metrics)

        return result

    @classmethod
    def parse_compliance_review(cls, response: str) -> AgentResponse:
        """
        Parse a response specifically from the Compliance Officer.

        Expects format like:
        "[COMPLIANT]: Strategy passes all checks."
        "[VIOLATION]: Strategy violates guidelines - specific issue"
        """
        result = cls.parse(response)

        # Map compliance decisions
        if "COMPLIANT" in response.upper():
            result.decision = Decision.COMPLIANT
            result.confidence = 0.95
        elif "VIOLATION" in response.upper():
            result.decision = Decision.VIOLATION
            result.confidence = 0.95
        elif "CONCERN" in response.upper():
            result.decision = Decision.CONCERN
            result.confidence = 0.7

        return result

    @classmethod
    def parse_code_review(cls, response: str) -> AgentResponse:
        """
        Parse a response specifically from the Code Reviewer.

        Extracts BUG, WARNING, OPTIMIZE, STYLE issues.
        """
        result = cls.parse(response)

        # Code review: decision based on presence of bugs
        bugs = [i for i in result.issues if i.get("severity") == "bug"]
        warnings = [i for i in result.issues if i.get("severity") == "warning"]

        if bugs:
            result.decision = Decision.VETOED
            result.confidence = 0.9
        elif warnings:
            result.decision = Decision.CONCERN
            result.confidence = 0.7
        else:
            result.decision = Decision.APPROVED
            result.confidence = 0.8

        return result

    @classmethod
    def _extract_decision(cls, response: str) -> Tuple[Decision, float]:
        """Extract the primary decision and confidence score."""
        response_upper = response.upper()

        # Check patterns in order (more specific first)
        for pattern, decision in cls.DECISION_PATTERNS.items():
            if re.search(pattern, response_upper):
                # Calculate confidence based on position and emphasis
                match = re.search(pattern, response_upper)
                position_score = 1.0 - (match.start() / max(len(response), 1)) * 0.3

                # Higher confidence if at start of response
                if match.start() < 50:
                    confidence = min(0.95, position_score + 0.1)
                else:
                    confidence = position_score * 0.8

                return decision, confidence

        return Decision.UNKNOWN, 0.0

    @classmethod
    def _extract_reasons(cls, response: str, decision: Decision) -> List[str]:
        """Extract reasons supporting the decision."""
        reasons = []

        # Common reason extraction patterns
        patterns = [
            r'(?:VETO|APPROVED|REJECTED):\s*(.+?)(?:\n\n|\n-|$)',
            r'(?:because|due to|reason|issue):\s*(.+?)(?:\n|$)',
            r'^\s*[-•]\s*(.+?)$',  # Bullet points
        ]

        for pattern in patterns:
            matches = re.finditer(pattern, response, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                reason = match.group(1).strip()
                if reason and len(reason) > 10:  # Filter out short/empty
                    reasons.append(reason)

        # Deduplicate while preserving order
        seen = set()
        unique_reasons = []
        for r in reasons:
            r_lower = r.lower()
            if r_lower not in seen:
                seen.add(r_lower)
                unique_reasons.append(r)

        return unique_reasons[:5]  # Limit to top 5

    @classmethod
    def _extract_issues(cls, response: str) -> List[Dict[str, str]]:
        """Extract issues with severity levels."""
        issues = []

        for pattern, severity in cls.ISSUE_PATTERNS.items():
            matches = re.finditer(pattern, response, re.IGNORECASE)
            for match in matches:
                description = match.group(1).strip() if match.groups() else ""
                if description:
                    issues.append({
                        "severity": severity.value,
                        "description": description
                    })

        return issues

    @classmethod
    def _extract_recommendations(cls, response: str) -> List[str]:
        """Extract actionable recommendations."""
        recommendations = []

        for pattern in cls.RECOMMENDATION_PATTERNS:
            matches = re.finditer(pattern, response, re.IGNORECASE)
            for match in matches:
                rec = match.group(1).strip()
                if rec:
                    recommendations.append(rec)

        return recommendations

    @classmethod
    def _extract_requests(cls, response: str) -> List[Dict[str, str]]:
        """Extract resource/improvement requests."""
        requests = []

        for pattern in cls.REQUEST_PATTERNS:
            matches = re.finditer(pattern, response, re.IGNORECASE)
            for match in matches:
                groups = match.groups()
                if len(groups) >= 2:
                    requests.append({
                        "type": groups[0].strip(),
                        "description": groups[1].strip()
                    })
                elif len(groups) == 1:
                    requests.append({
                        "type": "general",
                        "description": groups[0].strip()
                    })

        return requests

    @classmethod
    def _extract_metrics(cls, response: str) -> Dict[str, float]:
        """Extract any numerical metrics mentioned."""
        metrics = {}

        # Common metric patterns
        metric_patterns = [
            (r'[Ss]harpe\s*(?:[Rr]atio)?\s*[:=]?\s*([-\d.]+)', 'sharpe'),
            (r'[Dd]rawdown\s*[:=]?\s*([-\d.]+)%?', 'max_drawdown'),
            (r'[Rr]eturn\s*[:=]?\s*([-\d.]+)%?', 'total_return'),
            (r'[Ww]in\s*[Rr]ate\s*[:=]?\s*([-\d.]+)%?', 'win_rate'),
            (r'[Pp]rofit\s*[Ff]actor\s*[:=]?\s*([-\d.]+)', 'profit_factor'),
            (r'[Tt]rade[s]?\s*[:=]?\s*(\d+)', 'trade_count'),
        ]

        for pattern, name in metric_patterns:
            match = re.search(pattern, response)
            if match:
                try:
                    value = float(match.group(1))
                    metrics[name] = value
                except ValueError:
                    pass

        return metrics

    @classmethod
    def _extract_risk_metrics(cls, response: str) -> Dict[str, any]:
        """Extract risk-specific metrics and flags."""
        metrics = cls._extract_metrics(response)

        # Add risk-specific flags
        metrics['has_overfitting_concern'] = bool(
            re.search(r'overfit|curve.?fit|magic.?number', response, re.IGNORECASE)
        )
        metrics['has_data_concern'] = bool(
            re.search(r'look.?ahead|data.?snoop|future.?data', response, re.IGNORECASE)
        )
        metrics['has_sample_concern'] = bool(
            re.search(r'insufficient|small.?sample|few.?trade', response, re.IGNORECASE)
        )

        return metrics


def parse_agent_response(response: str, agent_type: str = "general") -> AgentResponse:
    """
    Convenience function to parse an agent response based on agent type.

    Args:
        response: Raw text response from the agent
        agent_type: Type of agent ("risk", "compliance", "code", "general")

    Returns:
        AgentResponse with structured data
    """
    parsers = {
        "risk": ResponseParser.parse_risk_review,
        "compliance": ResponseParser.parse_compliance_review,
        "code": ResponseParser.parse_code_review,
        "general": ResponseParser.parse,
    }

    parser = parsers.get(agent_type, ResponseParser.parse)
    return parser(response)


def is_approved(response: str) -> bool:
    """
    Quick check if a response indicates approval.

    More robust than simple string matching.

    Args:
        response: Raw agent response text

    Returns:
        True if the response indicates approval
    """
    parsed = ResponseParser.parse(response)
    return parsed.is_approved()


def is_vetoed(response: str) -> bool:
    """
    Quick check if a response indicates rejection/veto.

    Args:
        response: Raw agent response text

    Returns:
        True if the response indicates veto/rejection
    """
    parsed = ResponseParser.parse(response)
    return parsed.is_rejected()


def get_decision_with_confidence(response: str) -> Tuple[str, float]:
    """
    Get the decision and confidence score from a response.

    Args:
        response: Raw agent response text

    Returns:
        Tuple of (decision_string, confidence_float)
    """
    parsed = ResponseParser.parse(response)
    return parsed.decision.value, parsed.confidence
