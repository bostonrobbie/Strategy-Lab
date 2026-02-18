import ollama
from datetime import datetime
import json
import os
import traceback
from circuit_breaker import ollama_breaker, ollama_rate_limiter, CircuitOpenError

# Error and Request Log
ERROR_LOG_FILE = os.path.join(os.path.dirname(__file__), "reports", "AGENT_ERRORS.json")
REQUEST_LOG_FILE = os.path.join(os.path.dirname(__file__), "reports", "AGENT_REQUESTS.json")

class AgentLogger:
    """Centralized logging for agent errors and improvement requests."""

    @staticmethod
    def log_error(agent_name, error_type, message, context=None):
        """Log an error encountered by an agent."""
        os.makedirs(os.path.dirname(ERROR_LOG_FILE), exist_ok=True)

        errors = []
        if os.path.exists(ERROR_LOG_FILE):
            try:
                with open(ERROR_LOG_FILE, 'r') as f:
                    errors = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                # Log file corrupted or unreadable - start fresh
                errors = []

        entry = {
            "timestamp": datetime.now().isoformat(),
            "agent": agent_name,
            "error_type": error_type,
            "message": message,
            "context": context or {},
            "resolved": False
        }
        errors.append(entry)

        # Keep last 100 errors
        errors = errors[-100:]

        with open(ERROR_LOG_FILE, 'w') as f:
            json.dump(errors, f, indent=2)

    @staticmethod
    def log_request(agent_name, request_type, description, priority="medium"):
        """Log an improvement request from an agent."""
        os.makedirs(os.path.dirname(REQUEST_LOG_FILE), exist_ok=True)

        requests = []
        if os.path.exists(REQUEST_LOG_FILE):
            try:
                with open(REQUEST_LOG_FILE, 'r') as f:
                    requests = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                # Log file corrupted or unreadable - start fresh
                requests = []

        entry = {
            "timestamp": datetime.now().isoformat(),
            "agent": agent_name,
            "request_type": request_type,
            "description": description,
            "priority": priority,
            "status": "pending"
        }
        requests.append(entry)

        # Keep last 50 requests
        requests = requests[-50:]

        with open(REQUEST_LOG_FILE, 'w') as f:
            json.dump(requests, f, indent=2)

    @staticmethod
    def get_recent_errors(limit=10):
        if os.path.exists(ERROR_LOG_FILE):
            try:
                with open(ERROR_LOG_FILE, 'r') as f:
                    errors = json.load(f)
                return errors[-limit:]
            except (json.JSONDecodeError, IOError):
                # Log file corrupted or unreadable
                pass
            except Exception as e:
                # Unexpected error - log but don't crash
                print(f"[AgentLogger] Error reading error log: {e}")
        return []

    @staticmethod
    def get_pending_requests():
        if os.path.exists(REQUEST_LOG_FILE):
            try:
                with open(REQUEST_LOG_FILE, 'r') as f:
                    requests = json.load(f)
                return [r for r in requests if r.get("status") == "pending"]
            except (json.JSONDecodeError, IOError):
                # Log file corrupted or unreadable
                pass
            except Exception as e:
                # Unexpected error - log but don't crash
                print(f"[AgentLogger] Error reading request log: {e}")
        return []


class Agent:
    def __init__(self, name, role, system_prompt, model="llama3"):
        self.name = name
        self.role = role
        self.system_prompt = system_prompt
        self.model = model
        self.logger = AgentLogger()

    def chat(self, user_input):
        """
        Send a message to the agent and get a response.

        Uses rate limiting and circuit breaker for protection.

        Args:
            user_input: The prompt/question to send to the agent

        Returns:
            The agent's response text, or error message if failed
        """
        messages = [
            {'role': 'system', 'content': self.system_prompt},
            {'role': 'user', 'content': user_input}
        ]

        try:
            # Apply rate limiting (will block if needed)
            ollama_rate_limiter.wait_if_needed()

            # Check circuit breaker
            if not ollama_breaker.allow_request():
                error_msg = f"[{self.name}] Circuit breaker OPEN - API temporarily unavailable"
                self.logger.log_error(self.name, "circuit_open", error_msg)
                return error_msg

            # Make the API call
            response = ollama.chat(model=self.model, messages=messages)
            ollama_breaker.record_success()
            return response['message']['content']

        except CircuitOpenError as e:
            error_msg = f"[{self.name}] {str(e)}"
            return error_msg

        except Exception as e:
            ollama_breaker.record_failure()
            error_msg = f"[{self.name} Error]: {str(e)}"
            self.logger.log_error(
                self.name,
                "chat_error",
                str(e),
                {
                    "input_snippet": user_input[:200],
                    "traceback": traceback.format_exc()[-500:]
                }
            )
            return error_msg

    def request_improvement(self, request_type, description, priority="medium"):
        """Agent can request resources, data, or capabilities."""
        self.logger.log_request(self.name, request_type, description, priority)
        return f"[{self.name}] Request logged: {request_type}"


# ============================================
# CORE TEAM - The Original Four
# ============================================

class FundManagerAgent(Agent):
    """Lead Quantitative Researcher - Innovation & Alpha Generation"""
    def __init__(self):
        prompt = (
            "You are the Lead Quantitative Researcher and Chief Strategy Officer. "
            "Your primary responsibilities:\n"
            "1. INNOVATE: Develop novel trading strategies that capture Alpha\n"
            "2. ITERATE: Rapidly test and refine ideas based on backtest results\n"
            "3. COLLABORATE: Work with the team to improve strategies\n"
            "4. DOCUMENT: Clearly explain your trading logic and reasoning\n\n"
            "You are creative but data-driven. Every idea must be testable.\n"
            "Adhere to 'The Love Equation' (Positive Sum Games).\n"
            "When you need resources or data, explicitly state what you need.\n"
            "Format improvement requests as: [REQUEST: type - description]"
        )
        super().__init__("Fund Manager", "Chief Strategy Officer", prompt)


class RiskManagerAgent(Agent):
    """Dr. No - Chief Risk Officer & Capital Protection"""
    def __init__(self):
        prompt = (
            "You are 'Dr. No', the Chief Risk Officer. "
            "Your PRIMARY mission is to PROTECT CAPITAL at all costs.\n\n"
            "VETO CRITERIA (Non-negotiable):\n"
            "- Max Drawdown > 25%\n"
            "- Sharpe Ratio < 1.0\n"
            "- Trade Count < 30 (insufficient sample)\n"
            "- Profit Factor < 1.2\n"
            "- Signs of overfitting (magic numbers, curve-fitting)\n\n"
            "APPROVAL CRITERIA:\n"
            "- All risk metrics within bounds\n"
            "- Logic is sound and generalizable\n"
            "- Proper risk management (stops, position sizing)\n\n"
            "Response format:\n"
            "- If VETO: Start with 'VETO:' followed by specific reasons\n"
            "- If APPROVE: Start with 'APPROVED:' followed by risk assessment\n\n"
            "When you encounter issues, log them as: [ERROR: type - description]\n"
            "When you need something, log as: [REQUEST: type - description]"
        )
        super().__init__("Risk Manager", "Dr. No - CRO", prompt)


class SentimentScoutAgent(Agent):
    """Market Intelligence & Regime Detection"""
    def __init__(self):
        prompt = (
            "You are the Sentiment Scout and Market Intelligence Officer.\n\n"
            "Your responsibilities:\n"
            "1. REGIME DETECTION: Identify current market regime (trending, ranging, volatile)\n"
            "2. SENTIMENT ANALYSIS: Gauge market sentiment and risk appetite\n"
            "3. MACRO AWARENESS: Consider macroeconomic factors\n"
            "4. STRATEGY ALIGNMENT: Recommend which strategy types suit current conditions\n\n"
            "Current Simulated Environment: HIGH VOLATILITY, INFLATIONARY REGIME\n\n"
            "Provide actionable advice:\n"
            "- What strategy types work in this environment?\n"
            "- Risk adjustments needed?\n"
            "- Position sizing recommendations?\n\n"
            "When you need market data or tools, request them explicitly."
        )
        super().__init__("Sentiment Scout", "Market Intelligence", prompt)


class ChiefPsychologistAgent(Agent):
    """Team Health & Process Optimization"""
    def __init__(self):
        prompt = (
            "You are the Chief Psychologist and Process Guardian.\n\n"
            "Your responsibilities:\n"
            "1. MONITOR: Watch for loops, hallucinations, or 'tilt' in the AI team\n"
            "2. INTERVENE: Step in when agents are stuck or arguing unproductively\n"
            "3. OPTIMIZE: Suggest process improvements for better collaboration\n"
            "4. PRINCIPLES: Ensure alignment with Zero Human values (Truth, Love, Excellence)\n\n"
            "Watch for:\n"
            "- Repetitive failed attempts (> 3 same errors)\n"
            "- Conflicting agent recommendations\n"
            "- Over-optimization or analysis paralysis\n"
            "- Deviation from core principles\n\n"
            "Interventions:\n"
            "- [PAUSE]: Suggest team takes a different approach\n"
            "- [RESET]: Clear current line of thinking, start fresh\n"
            "- [ESCALATE]: Flag issue for human review"
        )
        super().__init__("Chief Psychologist", "Process Guardian", prompt)


# ============================================
# NEW TEAM MEMBERS - Expanded Capabilities
# ============================================

class PortfolioArchitectAgent(Agent):
    """Portfolio Construction & Diversification"""
    def __init__(self):
        prompt = (
            "You are the Portfolio Architect, responsible for strategy allocation.\n\n"
            "Your responsibilities:\n"
            "1. PORTFOLIO CONSTRUCTION: Combine strategies for optimal risk-adjusted returns\n"
            "2. CORRELATION ANALYSIS: Ensure strategies are uncorrelated\n"
            "3. WEIGHT OPTIMIZATION: Allocate capital across strategies\n"
            "4. REBALANCING: Recommend when to adjust allocations\n\n"
            "Portfolio Guidelines:\n"
            "- Maximum 30% allocation to any single strategy\n"
            "- Target portfolio Sharpe > 1.5\n"
            "- Maximum portfolio drawdown < 20%\n"
            "- Aim for 3-7 uncorrelated strategies\n\n"
            "When analyzing strategies for inclusion, consider:\n"
            "- Correlation with existing portfolio strategies\n"
            "- Regime performance (does it complement others?)\n"
            "- Risk contribution to overall portfolio\n\n"
            "Output format for portfolio recommendations:\n"
            "[PORTFOLIO UPDATE]\n"
            "- Strategy: [name] | Weight: [%] | Reason: [explanation]"
        )
        super().__init__("Portfolio Architect", "Portfolio Manager", prompt)


class CodeReviewerAgent(Agent):
    """Code Quality & Implementation Verification"""
    def __init__(self):
        prompt = (
            "You are the Code Reviewer and Implementation Specialist.\n\n"
            "Your responsibilities:\n"
            "1. CODE REVIEW: Check strategy code for bugs and issues\n"
            "2. PERFORMANCE: Identify slow or inefficient code\n"
            "3. BEST PRACTICES: Ensure code follows Python best practices\n"
            "4. REPRODUCIBILITY: Verify backtests are reproducible\n\n"
            "Check for:\n"
            "- Look-ahead bias (using future data)\n"
            "- Data snooping (fitting to historical data)\n"
            "- Missing edge cases (market closes, gaps, splits)\n"
            "- Proper order of operations (path-dependent issues)\n"
            "- Memory leaks or inefficiencies\n\n"
            "Report issues as:\n"
            "[BUG]: Critical issue that breaks functionality\n"
            "[WARNING]: Potential issue that may cause problems\n"
            "[OPTIMIZE]: Suggestion for better performance\n"
            "[STYLE]: Code style improvement"
        )
        super().__init__("Code Reviewer", "Implementation Specialist", prompt)


class PerformanceAnalystAgent(Agent):
    """Deep Performance Analysis & Attribution"""
    def __init__(self):
        prompt = (
            "You are the Performance Analyst, responsible for deep-dive analysis.\n\n"
            "Your responsibilities:\n"
            "1. ATTRIBUTION: Break down returns by factor, time, and market condition\n"
            "2. DRAWDOWN ANALYSIS: Understand when and why drawdowns occur\n"
            "3. TRADE ANALYSIS: Study individual trade patterns\n"
            "4. BENCHMARKING: Compare to relevant benchmarks\n\n"
            "Key metrics to analyze:\n"
            "- Win rate and average win/loss ratio\n"
            "- Maximum consecutive losses\n"
            "- Time in drawdown\n"
            "- Performance by hour/day/month\n"
            "- Slippage and commission impact\n\n"
            "Provide insights on:\n"
            "- What market conditions favor this strategy?\n"
            "- When should the strategy be turned off?\n"
            "- What improvements would have the biggest impact?\n\n"
            "Format recommendations as:\n"
            "[INSIGHT]: Key finding\n"
            "[RECOMMENDATION]: Suggested improvement"
        )
        super().__init__("Performance Analyst", "Quantitative Analyst", prompt)


class DataEngineerAgent(Agent):
    """Data Quality & Pipeline Management"""
    def __init__(self):
        prompt = (
            "You are the Data Engineer, responsible for data quality and pipelines.\n\n"
            "Your responsibilities:\n"
            "1. DATA QUALITY: Ensure data is clean, accurate, and complete\n"
            "2. PIPELINE MANAGEMENT: Maintain data ingestion and processing\n"
            "3. FEATURE ENGINEERING: Create and validate new features\n"
            "4. DATA REQUESTS: Source new data when strategies need it\n\n"
            "Data Quality Checks:\n"
            "- Missing values and gaps\n"
            "- Outliers and anomalies\n"
            "- Timestamp alignment\n"
            "- Corporate actions (splits, dividends)\n\n"
            "When strategies fail due to data issues:\n"
            "[DATA ERROR]: Description of the data problem\n"
            "[DATA REQUEST]: New data source needed\n"
            "[FEATURE]: New feature engineering suggestion\n\n"
            "Always validate data before strategies use it."
        )
        super().__init__("Data Engineer", "Data Pipeline Manager", prompt)


class ComplianceOfficerAgent(Agent):
    """Regulatory Compliance & Ethical Trading"""
    def __init__(self):
        prompt = (
            "You are the Compliance Officer, ensuring ethical and legal trading.\n\n"
            "Your responsibilities:\n"
            "1. REGULATORY: Ensure strategies comply with trading regulations\n"
            "2. ETHICAL: Verify strategies align with ethical principles\n"
            "3. DOCUMENTATION: Maintain audit trails and records\n"
            "4. LIMITS: Enforce position and exposure limits\n\n"
            "Check strategies for:\n"
            "- Market manipulation patterns\n"
            "- Front-running or information abuse\n"
            "- Excessive concentration\n"
            "- Proper record-keeping\n\n"
            "Compliance Response:\n"
            "[COMPLIANT]: Strategy passes all checks\n"
            "[CONCERN]: Potential issue needs review\n"
            "[VIOLATION]: Strategy violates guidelines - BLOCK\n\n"
            "All trading must be positive-sum and ethical."
        )
        super().__init__("Compliance Officer", "Ethics & Compliance", prompt)


# ============================================
# TEAM COLLABORATION UTILITIES
# ============================================

class TeamCollaboration:
    """Facilitates multi-agent collaboration and debate."""

    def __init__(self):
        self.fund_manager = FundManagerAgent()
        self.risk_manager = RiskManagerAgent()
        self.sentiment_scout = SentimentScoutAgent()
        self.psychologist = ChiefPsychologistAgent()
        self.portfolio_architect = PortfolioArchitectAgent()
        self.code_reviewer = CodeReviewerAgent()
        self.performance_analyst = PerformanceAnalystAgent()
        self.data_engineer = DataEngineerAgent()
        self.compliance_officer = ComplianceOfficerAgent()

        self.all_agents = {
            "fund_manager": self.fund_manager,
            "risk_manager": self.risk_manager,
            "sentiment_scout": self.sentiment_scout,
            "psychologist": self.psychologist,
            "portfolio_architect": self.portfolio_architect,
            "code_reviewer": self.code_reviewer,
            "performance_analyst": self.performance_analyst,
            "data_engineer": self.data_engineer,
            "compliance_officer": self.compliance_officer
        }

    def round_table(self, topic, context=""):
        """
        All agents discuss a topic and provide their perspective.
        Returns a consolidated view with all opinions.
        """
        responses = {}

        prompt = f"TOPIC: {topic}\n\nCONTEXT:\n{context}\n\nProvide your professional assessment from your role's perspective. Be concise but thorough."

        for name, agent in self.all_agents.items():
            try:
                responses[name] = agent.chat(prompt)
            except Exception as e:
                responses[name] = f"[ERROR]: {str(e)}"
                AgentLogger.log_error(agent.name, "round_table_error", str(e))

        return responses

    def strategy_review(self, strategy_code, backtest_results):
        """
        Full strategy review by relevant team members.
        """
        reviews = {}

        # Code Review
        code_prompt = f"Review this strategy code for issues:\n\n{strategy_code[-2000:]}"
        reviews["code_review"] = self.code_reviewer.chat(code_prompt)

        # Risk Review
        risk_prompt = f"Review these backtest results:\n\n{backtest_results}\n\nApprove or VETO?"
        reviews["risk_review"] = self.risk_manager.chat(risk_prompt)

        # Performance Analysis
        perf_prompt = f"Analyze these backtest results:\n\n{backtest_results}"
        reviews["performance_analysis"] = self.performance_analyst.chat(perf_prompt)

        # Compliance Check
        comp_prompt = f"Check this strategy for compliance:\n\n{strategy_code[-1000:]}"
        reviews["compliance_check"] = self.compliance_officer.chat(comp_prompt)

        return reviews

    def portfolio_decision(self, strategy_name, strategy_metrics, current_portfolio):
        """
        Decide whether to add a strategy to the portfolio.
        """
        prompt = (
            f"New Strategy: {strategy_name}\n"
            f"Metrics: {json.dumps(strategy_metrics, indent=2)}\n\n"
            f"Current Portfolio:\n{json.dumps(current_portfolio, indent=2)}\n\n"
            "Should we add this strategy? What weight? How does it affect portfolio risk?"
        )

        return self.portfolio_architect.chat(prompt)

    def get_all_improvement_requests(self):
        """Collect improvement requests from all agents."""
        return AgentLogger.get_pending_requests()

    def get_all_errors(self, limit=20):
        """Get recent errors from all agents."""
        return AgentLogger.get_recent_errors(limit)
