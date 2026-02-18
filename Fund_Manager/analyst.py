import os
import subprocess
import ollama
import json
import glob
import time
import re
import sys
import traceback
from datetime import datetime
from agents import (
    FundManagerAgent, RiskManagerAgent, SentimentScoutAgent, ChiefPsychologistAgent,
    PortfolioArchitectAgent, CodeReviewerAgent, PerformanceAnalystAgent,
    DataEngineerAgent, ComplianceOfficerAgent, TeamCollaboration, AgentLogger
)
from memory import GlobalMemory
from leaderboard import Leaderboard
from portfolio import Portfolio, get_portfolio
from code_validator import validate_and_fix_code, get_validation_prompt_suffix
from utils import safe_get, extract_nested_metrics, truncate_string
from constants import SYSTEM, FILES, LOGGING, RISK, SANDBOX_DIR, REPORTS_DIR
from response_parser import ResponseParser, parse_agent_response, is_approved, is_vetoed
import shutil

# Configuration
OLLAMA_MODEL = "llama3"  # Ensure this model is pulled: `ollama pull llama3`
RESEARCH_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(RESEARCH_DIR, "Fund_Manager", "reports")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)


class BoardLogger:
    """Persistent logger for 'Board Member' review."""
    def __init__(self, log_dir=OUTPUT_DIR):
        self.log_file = os.path.join(log_dir, "BOARD_LOG.md")
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w") as f:
                f.write("# Board Member Log\n\n")
                f.write("| Timestamp | Type | Message |\n")
                f.write("|-----------|------|---------|\n")

    def log(self, category, message):
        """Categories: [IMPROVEMENT], [FRICTION], [REQUEST], [PORTFOLIO], [TEAM]"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"| {timestamp} | **{category}** | {message} |\n"
        with open(self.log_file, "a") as f:
            f.write(entry)
        print(f"[{category}] {message}")
        sys.stdout.flush()


class DialogLogger:
    """Logs full LLM conversation (Dialog) for the Board - NEWEST FIRST."""
    def __init__(self, log_dir=OUTPUT_DIR):
        self.log_file = os.path.join(log_dir, "DIALOG_LOG.md")
        self.entries = []
        self._load_existing()

    def _load_existing(self):
        """Load existing entries to maintain order."""
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'r') as f:
                    content = f.read()
                # Parse existing entries (simplified - just keep raw content for now)
                self.entries = []
            except:
                self.entries = []

    def log(self, agent_name, prompt, response):
        """Log dialog with newest entries at the top."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        entry = {
            "timestamp": timestamp,
            "agent": agent_name,
            "prompt": prompt[:500],
            "response": response
        }
        self.entries.insert(0, entry)  # Insert at beginning (newest first)

        # Keep last 100 entries
        self.entries = self.entries[:100]

        # Rewrite file with newest first
        self._save()

    def _save(self):
        with open(self.log_file, "w") as f:
            f.write("# Agent Dialog Log (Newest First)\n\n")
            f.write(f"*Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
            f.write("---\n\n")

            for entry in self.entries:
                f.write(f"## [{entry['timestamp']}] {entry['agent']}\n\n")
                f.write(f"**Prompt:**\n```\n{entry['prompt']}...\n```\n\n")
                f.write(f"**Response:**\n{entry['response']}\n\n")
                f.write("---\n\n")


class FundManager:
    def __init__(self, model_name=OLLAMA_MODEL):
        self.model = model_name
        self.board_logger = BoardLogger()
        self.dialog_logger = DialogLogger()

        # The Multi-Agent Team (Core)
        self.fund_manager = FundManagerAgent()
        self.risk_manager = RiskManagerAgent()
        self.sentiment_scout = SentimentScoutAgent()
        self.psychologist = ChiefPsychologistAgent()

        # Extended Team
        self.portfolio_architect = PortfolioArchitectAgent()
        self.code_reviewer = CodeReviewerAgent()
        self.performance_analyst = PerformanceAnalystAgent()
        self.data_engineer = DataEngineerAgent()
        self.compliance_officer = ComplianceOfficerAgent()

        # Team Collaboration
        self.team = TeamCollaboration()

        # The Brain
        self.memory = GlobalMemory(model=model_name)
        self.leaderboard = Leaderboard()
        self.portfolio = get_portfolio()

    def generate_response(self, prompt, agent=None):
        """Generates a response from a specific agent."""
        active_agent = agent if agent else self.fund_manager
        agent_name = active_agent.name

        # Actual Chat
        response = active_agent.chat(prompt)

        # Log the interaction (newest first)
        self.dialog_logger.log(agent_name, prompt[:200], response)

        return response

    def list_available_strategies(self):
        """Lists all python research scripts in the Quant_Lab directory."""
        scripts = glob.glob(os.path.join(RESEARCH_DIR, "research_*.py"))
        return [os.path.basename(s) for s in scripts]

    def run_research_task(self, script_name):
        """Executes a specific research script and captures the output."""
        script_path = os.path.join(RESEARCH_DIR, script_name)
        if not os.path.exists(script_path):
            AgentLogger.log_error("System", "file_not_found", f"Script {script_name} not found")
            return f"Error: Script {script_name} not found."

        print(f"Running research task: {script_name}...")
        sys.stdout.flush()

        try:
            result = subprocess.run(
                ["python", script_path],
                capture_output=True,
                text=True,
                cwd=RESEARCH_DIR,
                timeout=300  # 5 minute timeout
            )
            # Separate stdout and stderr - only use stdout for parsing
            output = result.stdout

            # Log stderr separately if present (don't mix with output)
            if result.stderr and result.stderr.strip():
                stderr_preview = result.stderr[:500] if len(result.stderr) > 500 else result.stderr
                self.board_logger.log("[WARNING]", f"Script stderr: {stderr_preview[:200]}")
                AgentLogger.log_error("System", "script_stderr", stderr_preview, {"script": script_name})

            return output
        except subprocess.TimeoutExpired as e:
            error_msg = f"Script timed out after 300 seconds"
            AgentLogger.log_error("System", "timeout", error_msg, {"script": script_name})
            self.board_logger.log("[FRICTION]", f"Failed to run research task {script_name}: {error_msg}")
            return f"Failed to run script: {error_msg}"
        except Exception as e:
            AgentLogger.log_error("System", "execution_error", str(e), {"script": script_name})
            self.board_logger.log("[FRICTION]", f"Failed to run research task {script_name}: {e}")
            return f"Failed to run script: {str(e)}"

    def analyze_results(self, research_output, script_name):
        """Sends the research output to the LLM for analysis."""
        prompt = (
            f"Here are the results from the research script '{script_name}':\n\n"
            f"```\n{research_output[-4000:]}\n```\n"
            f"(Note: Output may be truncated)\n\n"
            f"Please provide a hierarchical report:\n"
            f"1. Executive Summary: What did this test prove?\n"
            f"2. Key Metrics: Analyze the PnL, Sharpe, or other metrics found in the logs.\n"
            f"3. Recommendation: Should we Deploy, Refine, or Discard this strategy?\n"
            f"4. Board Requests: Do you need any specific tools or data to improve this?\n"
            f"5. Errors Encountered: List any errors or issues you noticed.\n"
        )
        response = self.generate_response(prompt)

        # Extract requests for the Board Log
        try:
            if "Board Requests:" in response:
                requests_text = response.split("Board Requests:")[1].split("\n\n")[0].strip()
                if len(requests_text) > 10:
                    self.board_logger.log("[REQUEST]", f"From Analysis of {script_name}: {requests_text[:100]}...")
                    # Also log to agent request system
                    AgentLogger.log_request("Fund Manager", "analysis_request", requests_text[:200])
        except (IndexError, AttributeError) as e:
            # Failed to parse board requests - not critical
            pass
        except Exception as e:
            AgentLogger.log_error("System", "request_parse_error", str(e), {"script": script_name})

        return response

    def parse_metrics_from_output(self, output):
        """
        Extract metrics from backtest output using multiple strategies.

        Strategy 1: Try to parse JSON output (most reliable)
        Strategy 2: Use regex patterns with multiple variations
        Strategy 3: Use MetricsExtractor for alternate key names

        Returns:
            Dictionary of extracted metrics with canonical key names
        """
        metrics = {}

        # Strategy 1: Try JSON parsing first (most reliable)
        json_metrics = self._try_json_extraction(output)
        if json_metrics:
            metrics.update(json_metrics)

        # Strategy 2: Regex patterns with multiple variations
        # Each key has a list of patterns to try (order matters - more specific first)
        pattern_groups = {
            'sharpe': [
                r'Sharpe\s*Ratio\s*[:=]\s*(-?\d+\.?\d*)',
                r'Sharpe\s*[:=]\s*(-?\d+\.?\d*)',
                r'SR\s*[:=]\s*(-?\d+\.?\d*)',
                r'"sharpe":\s*(-?\d+\.?\d*)',
            ],
            'return_pct': [
                r'Total\s*Return\s*[:=]\s*(-?\d+\.?\d*)%?',
                r'Return\s*[:=]\s*(-?\d+\.?\d*)%?',
                r'CAGR\s*[:=]\s*(-?\d+\.?\d*)%?',
                r'"return_pct":\s*(-?\d+\.?\d*)',
                r'"total_return":\s*(-?\d+\.?\d*)',
            ],
            'max_dd': [
                r'Max\s*Drawdown\s*[:=]\s*(-?\d+\.?\d*)%?',
                r'Max\s*DD\s*[:=]\s*(-?\d+\.?\d*)%?',
                r'MDD\s*[:=]\s*(-?\d+\.?\d*)%?',
                r'Maximum\s*Drawdown\s*[:=]\s*(-?\d+\.?\d*)%?',
                r'"max_dd":\s*(-?\d+\.?\d*)',
                r'"max_drawdown":\s*(-?\d+\.?\d*)',
            ],
            'profit_factor': [
                r'Profit\s*Factor\s*[:=]\s*(\d+\.?\d*)',
                r'PF\s*[:=]\s*(\d+\.?\d*)',
                r'"profit_factor":\s*(\d+\.?\d*)',
            ],
            'win_rate': [
                r'Win\s*Rate\s*[:=]\s*(\d+\.?\d*)%?',
                r'Win%\s*[:=]\s*(\d+\.?\d*)',
                r'Winning\s*%\s*[:=]\s*(\d+\.?\d*)',
                r'"win_rate":\s*(\d+\.?\d*)',
            ],
            'trade_count': [
                r'Trade\s*Count\s*[:=]\s*(\d+)',
                r'Total\s*Trades\s*[:=]\s*(\d+)',
                r'Trades\s*[:=]\s*(\d+)',
                r'Num\s*Trades\s*[:=]\s*(\d+)',
                r'"trade_count":\s*(\d+)',
                r'"trades":\s*(\d+)',
            ],
            'sortino': [
                r'Sortino\s*Ratio\s*[:=]\s*(-?\d+\.?\d*)',
                r'Sortino\s*[:=]\s*(-?\d+\.?\d*)',
                r'"sortino":\s*(-?\d+\.?\d*)',
            ],
            'calmar': [
                r'Calmar\s*Ratio\s*[:=]\s*(-?\d+\.?\d*)',
                r'Calmar\s*[:=]\s*(-?\d+\.?\d*)',
                r'"calmar":\s*(-?\d+\.?\d*)',
            ],
            'avg_win': [
                r'Avg\s*Win\s*[:=]\s*\$?(-?\d+\.?\d*)',
                r'Average\s*Win\s*[:=]\s*\$?(-?\d+\.?\d*)',
            ],
            'avg_loss': [
                r'Avg\s*Loss\s*[:=]\s*\$?(-?\d+\.?\d*)',
                r'Average\s*Loss\s*[:=]\s*\$?(-?\d+\.?\d*)',
            ],
            'expectancy': [
                r'Expectancy\s*[:=]\s*\$?(-?\d+\.?\d*)',
                r'"expectancy":\s*(-?\d+\.?\d*)',
            ],
        }

        for key, patterns in pattern_groups.items():
            if key in metrics:
                continue  # Skip if already found via JSON

            for pattern in patterns:
                match = re.search(pattern, output, re.IGNORECASE)
                if match:
                    try:
                        metrics[key] = float(match.group(1))
                        break  # Found a match, move to next metric
                    except (ValueError, TypeError):
                        continue  # Try next pattern

        return metrics

    def _try_json_extraction(self, output):
        """
        Try to extract metrics from JSON in the output.

        Looks for JSON blocks or inline JSON metrics.

        Returns:
            Dictionary of metrics if found, empty dict otherwise
        """
        metrics = {}

        # Try to find JSON block
        json_patterns = [
            r'\{[^{}]*"sharpe"[^{}]*\}',
            r'```json\s*(\{[^`]+\})\s*```',
            r'METRICS:\s*(\{[^}]+\})',
        ]

        for pattern in json_patterns:
            match = re.search(pattern, output, re.IGNORECASE | re.DOTALL)
            if match:
                try:
                    json_str = match.group(1) if match.lastindex else match.group(0)
                    data = json.loads(json_str)

                    # Map common JSON keys to canonical names
                    key_mapping = {
                        'sharpe_ratio': 'sharpe',
                        'total_return': 'return_pct',
                        'max_drawdown': 'max_dd',
                        'trades': 'trade_count',
                        'num_trades': 'trade_count',
                    }

                    for json_key, value in data.items():
                        canonical = key_mapping.get(json_key.lower(), json_key.lower())
                        if isinstance(value, (int, float)):
                            metrics[canonical] = float(value)

                    if metrics:
                        return metrics
                except (json.JSONDecodeError, ValueError, TypeError):
                    continue

        return metrics

    def evolve_strategy(self, script_name, iterations=1):
        """Iteratively improves a strategy via Multi-Agent Debate and Memory."""
        print(f"\n{'='*60}")
        print(f"Starting evolutionary cycle for {script_name}...")
        print(f"{'='*60}\n")
        sys.stdout.flush()

        source_path = os.path.join(RESEARCH_DIR, script_name)
        with open(source_path, 'r') as f:
            original_code = f.read()

        sandbox_dir = os.path.join(os.path.dirname(__file__), "Strategy_Sandbox")
        os.makedirs(sandbox_dir, exist_ok=True)

        current_code = original_code

        for i in range(iterations):
            print(f"\n--- Iteration {i+1}/{iterations} ---")
            sys.stdout.flush()

            # --- STEP A: SENTIMENT SCOUT ---
            print("  [1/8] Consulting Sentiment Scout...")
            sys.stdout.flush()
            scout_advice = self.generate_response(
                "Current simulated environment: HIGH VOLATILITY, INFLATIONARY. "
                "What strategic adjustments should we make? "
                "If you encounter any issues or need data, log them as [REQUEST: type - description]",
                agent=self.sentiment_scout
            )

            # --- STEP B: DATA ENGINEER CHECK ---
            print("  [2/8] Data Engineer checking data quality...")
            sys.stdout.flush()
            data_check = self.generate_response(
                f"Review the data requirements for this strategy. "
                f"Current strategy uses: NQ 5-minute data from 2015-2024. "
                f"Any data quality concerns? Log issues as [DATA ERROR: description] or [DATA REQUEST: description]",
                agent=self.data_engineer
            )

            # --- STEP C: MEMORY RECALL & LEADERBOARD ---
            print("  [3/8] Consulting Memory Bank...")
            sys.stdout.flush()
            past_lessons = self.memory.query_memory(f"Improve {script_name} with {scout_advice[:50]}")
            memory_context = ""
            if past_lessons:
                memory_context = "PAST LESSONS:\n" + "\n".join([f"- {p['summary']} (Outcome: {p['outcome']})" for p in past_lessons])

            champion = self.leaderboard.get_champion()
            champion_context = ""
            if champion:
                champion_context = f"CURRENT CHAMPION: {champion['name']} (Sharpe: {champion['sharpe']})\nCHAMPION LOGIC: {champion['logic_summary']}\n(Try to beat this!)"

            # Get portfolio context (safely access nested dict)
            portfolio_summary = self.portfolio.get_portfolio_summary()
            metrics = extract_nested_metrics(portfolio_summary)
            num_strategies = metrics['active_count']
            portfolio_sharpe = metrics['sharpe']
            portfolio_context = f"CURRENT PORTFOLIO: {num_strategies} strategies, Portfolio Sharpe: {portfolio_sharpe}"

            # --- STEP D: IDEATION (Fund Manager) ---
            print("  [4/8] Fund Manager ideating improvements...")
            sys.stdout.flush()
            prompt = (
                f"Analyze this strategy:\n\n{current_code[-2000:]}\n\n"
                f"Sentiment Scout Advice: {scout_advice[:200]}\n"
                f"Data Engineer Notes: {data_check[:200]}\n"
                f"{memory_context}\n"
                f"{champion_context}\n"
                f"{portfolio_context}\n"
                f"Suggest ONE concrete code improvement. Return ONLY the suggested change description."
            )
            idea = self.generate_response(prompt, agent=self.fund_manager)
            print(f"  Idea: {idea[:100]}...")
            sys.stdout.flush()

            # --- STEP E: CODE GENERATION (Fund Manager) ---
            print("  [5/8] Generating improved code...")
            sys.stdout.flush()

            # Enhanced prompt with explicit import requirements
            validation_suffix = get_validation_prompt_suffix()
            code_prompt = (
                f"Here is the strategy code:\n\n{current_code}\n\n"
                f"APPLY THIS CHANGE: {idea}\n"
                f"{validation_suffix}"
            )
            new_code = self.generate_response(code_prompt, agent=self.fund_manager)

            # Validate and auto-fix the generated code
            is_valid, new_code, validation_issues = validate_and_fix_code(new_code)

            if validation_issues:
                self.board_logger.log("[IMPROVEMENT]", f"Code auto-fixed: {', '.join(validation_issues[:3])}")

            if not is_valid:
                print(f"  [WARNING] Code validation failed: {validation_issues}")
                sys.stdout.flush()
                self.board_logger.log("[FRICTION]", f"Code generation failed validation: {validation_issues[0] if validation_issues else 'Unknown'}")
                AgentLogger.log_error("Fund Manager", "code_validation", f"Generated code failed: {validation_issues}")

                # Retry with error feedback (one attempt)
                retry_prompt = (
                    f"The previous code had these issues: {validation_issues}\n"
                    f"Please regenerate the strategy code fixing these problems.\n"
                    f"Original code:\n{current_code[-1500:]}\n"
                    f"Requested change: {idea}\n"
                    f"{validation_suffix}"
                )
                new_code = self.generate_response(retry_prompt, agent=self.fund_manager)
                is_valid, new_code, retry_issues = validate_and_fix_code(new_code)

                if not is_valid:
                    print(f"  [ERROR] Code still invalid after retry. Skipping this iteration.")
                    sys.stdout.flush()
                    continue

            # --- STEP F: CODE REVIEW ---
            print("  [6/8] Code Reviewer checking for issues...")
            sys.stdout.flush()
            code_review = self.generate_response(
                f"Review this strategy code for bugs, look-ahead bias, or issues:\n\n{new_code[-2000:]}\n\n"
                f"Report issues as [BUG], [WARNING], or [OPTIMIZE].",
                agent=self.code_reviewer
            )

            # Check for critical bugs
            if "[BUG]" in code_review.upper():
                self.board_logger.log("[FRICTION]", f"Code Reviewer found bugs - skipping candidate")
                AgentLogger.log_error("Code Reviewer", "code_bug", code_review[:200])
                continue

            # Save Candidate
            candidate_name = f"candidate_{script_name.replace('.py', '')}_v{int(time.time())}.py"
            candidate_path = os.path.join(sandbox_dir, candidate_name)
            with open(candidate_path, 'w') as f:
                f.write(new_code)

            print(f"  [7/8] Testing candidate: {candidate_name}...")
            sys.stdout.flush()

            # --- STEP G: VERIFICATION (Backtest) ---
            relative_candidate_path = os.path.join("Fund_Manager", "Strategy_Sandbox", candidate_name)
            result_output = self.run_research_task(relative_candidate_path)

            # Parse metrics
            metrics = self.parse_metrics_from_output(result_output)

            # --- STEP H: MULTI-AGENT REVIEW ---
            print("  [8/8] Team review in progress...")
            sys.stdout.flush()

            if "Traceback" in result_output or "Error" in result_output:
                outcome = "FAILURE"
                self.board_logger.log("[FRICTION]", f"Candidate {candidate_name} crashed.")
                AgentLogger.log_error("System", "backtest_crash", result_output[-500:], {"candidate": candidate_name})
            else:
                # Risk Manager Review
                risk_review = self.generate_response(
                    f"Review these backtest results:\n{result_output[-2000:]}\n\n"
                    f"Approve or VETO based on MaxDD < 25%, Sharpe > 1, Profit Factor > 1.2. "
                    f"Start response with VETO: or APPROVED:",
                    agent=self.risk_manager
                )

                # Compliance Check
                compliance_review = self.generate_response(
                    f"Check this strategy for compliance and ethical concerns:\n{new_code[-1000:]}\n\n"
                    f"Respond with [COMPLIANT], [CONCERN], or [VIOLATION].",
                    agent=self.compliance_officer
                )

                # Performance Analysis
                perf_analysis = self.generate_response(
                    f"Analyze these backtest results for insights:\n{result_output[-1500:]}\n\n"
                    f"Provide [INSIGHT] and [RECOMMENDATION] tags.",
                    agent=self.performance_analyst
                )

                # Parse structured responses
                risk_parsed = parse_agent_response(risk_review, agent_type="risk")
                compliance_parsed = parse_agent_response(compliance_review, agent_type="compliance")
                perf_parsed = parse_agent_response(perf_analysis, agent_type="general")

                # Log confidence scores for transparency
                self.board_logger.log("[TEAM]",
                    f"Risk: {risk_parsed.decision.value} ({risk_parsed.confidence:.0%}), "
                    f"Compliance: {compliance_parsed.decision.value} ({compliance_parsed.confidence:.0%})"
                )

                # Decision: Approved only if risk approves AND compliance doesn't reject
                if risk_parsed.is_approved() and not compliance_parsed.is_rejected():
                    outcome = "SUCCESS"
                    self.board_logger.log("[IMPROVEMENT]", f"Team Approved: {candidate_name}")
                    self.board_logger.log("[TEAM]", f"Performance: {perf_analysis[:100]}...")

                    # Log any recommendations from performance analysis
                    if perf_parsed.recommendations:
                        for rec in perf_parsed.recommendations[:3]:
                            self.board_logger.log("[TEAM]", f"Perf Insight: {rec[:80]}")

                    # Update Leaderboard
                    if metrics:
                        self.leaderboard.update(candidate_name, {
                            "Sharpe Ratio": metrics.get("sharpe", 0),
                            "Total Return": f"{metrics.get('return_pct', 0)}%"
                        }, idea[:200])

                    # Portfolio Decision
                    portfolio_decision = self.generate_response(
                        f"New Strategy: {candidate_name}\n"
                        f"Metrics: {json.dumps(metrics, indent=2)}\n"
                        f"Current Portfolio: {json.dumps(self.portfolio.get_portfolio_summary(), indent=2)}\n\n"
                        f"Should we add this to the portfolio? What weight?",
                        agent=self.portfolio_architect
                    )

                    # Add to portfolio if recommended
                    if "add" in portfolio_decision.lower() or "include" in portfolio_decision.lower():
                        self.portfolio.add_strategy(
                            name=candidate_name,
                            metrics=metrics,
                            code_path=candidate_path,
                            logic_summary=idea[:300]
                        )
                        self.board_logger.log("[PORTFOLIO]", f"Added {candidate_name} to portfolio")

                else:
                    outcome = "FAILURE"
                    if risk_parsed.is_rejected():
                        self.board_logger.log("[FRICTION]", f"Risk Manager VETOED: {candidate_name}")
                        # Log specific reasons if available
                        for reason in risk_parsed.reasons[:2]:
                            self.board_logger.log("[FRICTION]", f"  Reason: {reason[:100]}")
                    if compliance_parsed.is_rejected():
                        self.board_logger.log("[FRICTION]", f"Compliance VIOLATION: {candidate_name}")
                        for issue in compliance_parsed.issues[:2]:
                            self.board_logger.log("[FRICTION]", f"  Issue: {issue.get('description', '')[:100]}")

            # --- STEP I: SAVE TO MEMORY ---
            self.memory.add_experience(
                summary=f"Idea: {idea[:100]} | Context: {scout_advice[:50]}",
                outcome=outcome,
                metrics={"raw_output_snippet": result_output[-100:], **metrics}
            )

            # --- STEP J: PSYCHOLOGIST CHECK ---
            psych_review = self.generate_response(
                f"Review recent activity. We've had {i+1} iterations. "
                f"Outcome: {outcome}. Are we stuck in a loop? Aligned with principles?",
                agent=self.psychologist
            )

            if "[PAUSE]" in psych_review or "[RESET]" in psych_review:
                self.board_logger.log("[TEAM]", f"Psychologist intervention: {psych_review[:100]}")
                break

    def cleanup_sandbox(self, max_files: int = None, max_age_days: int = None):
        """
        Clean up old strategy candidates from the sandbox directory.

        Archives candidates that are:
        - Older than max_age_days
        - Beyond the max_files limit

        Protects strategies that are in the portfolio or on the leaderboard.

        Args:
            max_files: Maximum files to keep (default from constants)
            max_age_days: Maximum age in days before archiving (default from constants)
        """
        from datetime import timedelta

        max_files = max_files or FILES.MAX_SANDBOX_FILES
        max_age_days = max_age_days or FILES.SANDBOX_RETENTION_DAYS

        sandbox_dir = os.path.join(os.path.dirname(__file__), "Strategy_Sandbox")
        if not os.path.exists(sandbox_dir):
            return

        # Get all Python files in sandbox
        files = glob.glob(os.path.join(sandbox_dir, "*.py"))

        if len(files) <= max_files:
            return

        # Get file info with modification times
        file_info = []
        for f in files:
            try:
                stat = os.stat(f)
                file_info.append({
                    'path': f,
                    'name': os.path.basename(f),
                    'mtime': datetime.fromtimestamp(stat.st_mtime),
                    'size': stat.st_size
                })
            except OSError:
                continue

        # Get strategies to protect (in portfolio or leaderboard)
        protected = set()

        # Protect portfolio strategies
        for s in self.portfolio.get_all_strategies():
            protected.add(s.get('name', ''))
            if s.get('code_path'):
                protected.add(os.path.basename(s['code_path']))

        # Protect leaderboard strategies
        for s in self.leaderboard.leaders:
            protected.add(s.get('name', ''))

        # Sort by modification time (oldest first)
        file_info.sort(key=lambda x: x['mtime'])

        # Create archive directory
        archive_dir = os.path.join(sandbox_dir, "_archive")
        os.makedirs(archive_dir, exist_ok=True)

        archived = 0
        cutoff_date = datetime.now() - timedelta(days=max_age_days)

        for info in file_info:
            # Stop if we're within the file limit
            if len(files) - archived <= max_files:
                break

            # Don't archive protected strategies
            if info['name'] in protected:
                continue

            # Archive if old enough
            if info['mtime'] < cutoff_date:
                try:
                    archive_path = os.path.join(archive_dir, info['name'])
                    shutil.move(info['path'], archive_path)
                    archived += 1
                except (OSError, shutil.Error) as e:
                    AgentLogger.log_error("System", "cleanup_error", str(e), {"file": info['name']})

        if archived > 0:
            self.board_logger.log("[CLEANUP]", f"Archived {archived} old strategy candidates")
            print(f"[CLEANUP] Archived {archived} old strategy candidates")
            sys.stdout.flush()

    def generate_portfolio_report(self):
        """Generate a comprehensive portfolio report."""
        print("Generating Portfolio Report...")
        sys.stdout.flush()

        summary = self.portfolio.get_portfolio_summary()
        allocation = self.portfolio.get_allocation_breakdown()

        report_path = os.path.join(OUTPUT_DIR, "PORTFOLIO_REPORT.md")

        with open(report_path, "w") as f:
            f.write("# Portfolio Report\n\n")
            f.write(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")

            f.write("## Portfolio Overview\n\n")
            f.write(f"- **Total Strategies**: {summary['total_strategies']}\n")
            f.write(f"- **Active**: {summary['active_count']}\n")
            f.write(f"- **Paused**: {summary['paused_count']}\n")
            f.write(f"- **Deprecated**: {summary['deprecated_count']}\n")
            f.write(f"- **Total Capital**: ${summary['total_capital']:,.2f}\n\n")

            f.write("## Portfolio Metrics\n\n")
            pm = summary.get('portfolio_metrics', {})
            f.write(f"- **Portfolio Sharpe**: {pm.get('sharpe', 0):.2f}\n")
            f.write(f"- **Portfolio Return**: {pm.get('total_return', 0):.2f}%\n")
            f.write(f"- **Max Drawdown**: {pm.get('max_drawdown', 0):.2f}%\n\n")

            f.write("## Strategy Allocations\n\n")
            f.write("| Strategy | Weight | Allocated | Sharpe | Status |\n")
            f.write("|----------|--------|-----------|--------|--------|\n")
            for a in allocation:
                f.write(f"| {a['name'][:30]} | {a['weight_pct']}% | ${a['allocated_capital']:,.2f} | {a['sharpe']:.2f} | {a['status']} |\n")

            f.write("\n## All Strategies\n\n")
            for s in summary['strategies']:
                f.write(f"### {s['name']}\n")
                f.write(f"- Weight: {s['weight']}%\n")
                f.write(f"- Sharpe: {s['sharpe']:.2f}\n")
                f.write(f"- Return: {s['return_pct']:.2f}%\n")
                f.write(f"- Status: {s['status']}\n\n")

        print(f"Portfolio Report: {report_path}")
        return report_path

    def generate_master_report(self):
        """Aggregates all daily briefings into a Master Report."""
        print("Generating Master Report...")
        sys.stdout.flush()

        reports = glob.glob(os.path.join(OUTPUT_DIR, "Daily_Briefing_*.md"))
        reports.sort(reverse=True)

        master_path = os.path.join(OUTPUT_DIR, "MASTER_REPORT.md")

        # Get team requests and errors
        pending_requests = AgentLogger.get_pending_requests()
        recent_errors = AgentLogger.get_recent_errors(10)

        with open(master_path, "w") as master:
            master.write("# Fund Manager Master Report\n")
            master.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Portfolio Summary
            master.write("## Portfolio Status\n\n")
            summary = self.portfolio.get_portfolio_summary()
            pm = extract_nested_metrics(summary)
            master.write(f"- Active Strategies: {pm['active_count']}\n")
            master.write(f"- Portfolio Sharpe: {pm['sharpe']:.2f}\n")
            master.write(f"- Portfolio Return: {pm['total_return']:.2f}%\n\n")

            # Recent Activities
            master.write("## Recent Activities\n\n")
            for rep in reports[:5]:
                master.write(f"- [{os.path.basename(rep)}]({os.path.basename(rep)})\n")

            # Team Requests
            master.write("\n## Team Improvement Requests\n\n")
            if pending_requests:
                for req in pending_requests[:10]:
                    master.write(f"- **[{req['agent']}]** {req['request_type']}: {req['description'][:100]}\n")
            else:
                master.write("No pending requests.\n")

            # Recent Errors
            master.write("\n## Recent Errors\n\n")
            if recent_errors:
                for err in recent_errors[:5]:
                    master.write(f"- **[{err['agent']}]** {err['error_type']}: {err['message'][:100]}\n")
            else:
                master.write("No recent errors.\n")

            # Strategy Evolution
            master.write("\n## Strategy Evolution Highlights\n\n")
            sandbox_files = glob.glob(os.path.join(os.path.dirname(__file__), "Strategy_Sandbox", "*.py"))
            master.write(f"Total Candidates Generated: {len(sandbox_files)}\n\n")

            # Latest Briefing
            master.write("## Latest Briefing Details\n\n")
            if reports:
                with open(reports[0], 'r') as last_rep:
                    master.write(last_rep.read())

        print(f"Master Report Updated: {master_path}")
        return master_path

    def start_continuous_cycle(self, interval_seconds=None):
        """Runs the fund manager in an infinite loop."""
        interval_seconds = interval_seconds or SYSTEM.CYCLE_INTERVAL_SECONDS

        print(f"{'='*60}")
        print(f"Starting Continuous Autonomous Cycle")
        print(f"Interval: {interval_seconds}s | Team Size: 9 Agents")
        print(f"{'='*60}")
        sys.stdout.flush()

        error_count = 0
        max_errors = SYSTEM.MAX_ERROR_COUNT
        cycle_count = 0

        while True:
            try:
                cycle_count += 1
                print(f"\n[CYCLE {cycle_count}] Starting...")
                sys.stdout.flush()

                self.daily_briefing()
                self.generate_portfolio_report()
                self.generate_master_report()

                # Periodic cleanup (every 10 cycles)
                if cycle_count % 10 == 0:
                    self.cleanup_sandbox()

                error_count = 0
                print(f"\n[CYCLE {cycle_count}] Complete. Sleeping for {interval_seconds} seconds...")
                sys.stdout.flush()
                time.sleep(interval_seconds)
            except KeyboardInterrupt:
                print("\nStopping loop...")
                sys.stdout.flush()
                break
            except Exception as e:
                error_count += 1
                print(f"\n[ERROR] Cycle error ({error_count}/{max_errors}): {e}")
                traceback.print_exc()
                sys.stdout.flush()

                AgentLogger.log_error("System", "cycle_error", str(e))

                if error_count >= max_errors:
                    print(f"[WARNING] Too many errors. Pausing for {SYSTEM.ERROR_PAUSE_SECONDS}s...")
                    sys.stdout.flush()
                    time.sleep(SYSTEM.ERROR_PAUSE_SECONDS)
                    error_count = 0
                else:
                    print(f"[INFO] Retrying in 60 seconds...")
                    sys.stdout.flush()
                    time.sleep(60)

    def daily_briefing(self):
        """Runs a standard suite of checks and generates a daily report."""
        print("\n" + "="*60)
        print("Starting Daily Fund Verification...")
        print("="*60 + "\n")
        sys.stdout.flush()

        # 1. Run Standard Review
        target_script = "research_5m_strategies.py"
        raw_output = self.run_research_task(target_script)
        analysis = self.analyze_results(raw_output, target_script)

        # 2. Run Evolution (The Flywheel)
        print("\n--- Initiating Autonomous Evolution ---")
        sys.stdout.flush()
        self.evolve_strategy(target_script, iterations=1)

        # 3. Collect Team Insights
        print("\n--- Collecting Team Insights ---")
        sys.stdout.flush()

        # Each agent provides end-of-day summary
        team_summaries = {}
        for agent_name, agent in [
            ("Sentiment Scout", self.sentiment_scout),
            ("Data Engineer", self.data_engineer),
            ("Performance Analyst", self.performance_analyst)
        ]:
            try:
                summary = self.generate_response(
                    "Provide a brief end-of-day summary. Any requests or concerns?",
                    agent=agent
                )
                team_summaries[agent_name] = summary[:300]
            except Exception as e:
                # Log but don't fail the briefing if one agent fails
                AgentLogger.log_error(agent_name, "summary_error", str(e))
                team_summaries[agent_name] = f"[Error getting summary: {str(e)[:50]}]"

        # Save Report
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        report_filename = f"Daily_Briefing_{timestamp}.md"
        report_path = os.path.join(OUTPUT_DIR, report_filename)

        with open(report_path, "w") as f:
            f.write(f"# Daily Fund Briefing: {timestamp}\n\n")
            f.write(f"## Strategy Focus: {target_script}\n\n")
            f.write("**Hierarchical Report**\n\n")
            f.write(analysis)

            f.write("\n\n## Team Summaries\n\n")
            for name, summary in team_summaries.items():
                f.write(f"### {name}\n{summary}\n\n")

            f.write("\n## Evolution Status\n")
            f.write("Autonomous strategy evolution cycle executed. Check 'Strategy_Sandbox' for candidates.\n")

            # Portfolio snapshot
            f.write("\n## Portfolio Snapshot\n")
            summary = self.portfolio.get_portfolio_summary()
            pm = extract_nested_metrics(summary)
            f.write(f"- Active Strategies: {pm['active_count']}\n")
            f.write(f"- Portfolio Sharpe: {pm['sharpe']:.2f}\n")

        print(f"\nReport generated: {report_path}")
        sys.stdout.flush()
        return report_path


if __name__ == "__main__":
    manager = FundManager()

    print("="*60)
    print("Quant Lab Fund Manager Initialized")
    print("Team: 9 Specialized Agents")
    print("="*60)

    if len(sys.argv) > 1 and sys.argv[1] == "--loop":
        manager.start_continuous_cycle()
    elif len(sys.argv) > 1 and sys.argv[1] == "--report":
        manager.generate_master_report()
    elif len(sys.argv) > 1 and sys.argv[1] == "--portfolio":
        manager.generate_portfolio_report()
    else:
        print("\nUsage: python analyst.py [--loop | --report | --portfolio]")
        print("\nRunning single briefing cycle...\n")
        manager.daily_briefing()
        manager.generate_portfolio_report()
