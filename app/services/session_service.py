"""Session lifecycle and dependency wiring."""

from __future__ import annotations

from dataclasses import dataclass
import shlex
import shutil
import subprocess
from uuid import uuid4

from app.config import Config
from app.llm.interfaces import SummaryProvider
from app.llm.model_manager import ModelManager
from app.llm.ollama_client import OllamaClient
from app.nodes.summarizer import RuleBasedSummarizer
from app.schemas import FacilitatorDecision
from app.services.langfuse_service import LangfuseService
from app.services.search_service import DefaultSearchDigester, SearchService
from app.state import DiscussionState
from app.utils.markdown_logger import MarkdownLogger


@dataclass(slots=True)
class SessionServices:
    """Container for runtime services shared across nodes."""

    config: Config
    ollama_client: OllamaClient
    model_manager: ModelManager
    langfuse: LangfuseService
    markdown_logger: MarkdownLogger
    search_service: SearchService
    summarizer: SummaryProvider


def initialize_session(
    *,
    config: Config,
    topic: str,
    initial_compact_summary: str = "",
    input_sources: list[str] | None = None,
    session_id: str | None = None,
    preload_models: bool = True,
) -> tuple[SessionServices, DiscussionState]:
    """Create service container and initial state for a new session."""
    sid = session_id or uuid4().hex

    ollama_client = OllamaClient(
        base_url=config.ollama_base_url,
        timeout_seconds=config.ollama_timeout_seconds,
    )
    model_manager = ModelManager(client=ollama_client, keep_alive=config.model_keep_alive)

    langfuse = LangfuseService(
        enabled=config.langfuse_enabled,
        host=config.langfuse_host,
        public_key=config.langfuse_public_key,
        secret_key=config.langfuse_secret_key,
    )
    langfuse.startup_check()
    langfuse.start_trace(session_id=sid, topic=topic)

    markdown_logger = MarkdownLogger(
        output_dir=config.markdown_log_dir_path,
        result_dir=config.output_dir_path,
    )
    markdown_path = markdown_logger.create_session_file(session_id=sid, topic=topic)

    search_service = SearchService(
        command_template=config.search_command_template,
        timeout_seconds=config.search_timeout_seconds,
        digester=DefaultSearchDigester(),
        backend=config.search_backend,
        max_results=config.search_max_results,
    )

    search_enabled = True
    search_status_message: str | None = None
    try:
        if config.search_backend == "api":
            try:
                import ddgs  # noqa: F401
            except Exception:
                import duckduckgo_search  # noqa: F401  # pragma: no cover - compatibility path
        else:
            probe_command = shlex.split(config.search_command_template.format(query="probe"))
            probe_binary = probe_command[0] if probe_command else ""
            if not probe_binary:
                search_enabled = False
                search_status_message = "Search is disabled: SEARCH_COMMAND_TEMPLATE produced an empty command."
            elif shutil.which(probe_binary) is None:
                search_enabled = False
                search_status_message = (
                    f"Search is disabled: CLI binary '{probe_binary}' was not found in PATH. "
                    "Install the CLI or update SEARCH_COMMAND_TEMPLATE."
                )
            elif probe_binary == "ddgs":
                # Validate ddgs subcommand semantics early to avoid runtime retries on config errors.
                ddgs_probe = subprocess.run(
                    [probe_binary, "text", "--help"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False,
                )
                if ddgs_probe.returncode != 0:
                    error_excerpt = " ".join(ddgs_probe.stderr.strip().split()) or "ddgs text --help failed"
                    search_enabled = False
                    search_status_message = f"Search is disabled: ddgs CLI probe failed. stderr={error_excerpt}"
    except Exception as exc:
        search_enabled = False
        if config.search_backend == "api":
            search_status_message = f"Search is disabled: failed to initialize SEARCH_BACKEND=api ({exc})."
        else:
            search_status_message = f"Search is disabled: failed to parse SEARCH_COMMAND_TEMPLATE ({exc})."

    summarizer = RuleBasedSummarizer(recent_turns=config.summary_recent_turns)

    services = SessionServices(
        config=config,
        ollama_client=ollama_client,
        model_manager=model_manager,
        langfuse=langfuse,
        markdown_logger=markdown_logger,
        search_service=search_service,
        summarizer=summarizer,
    )

    if preload_models:
        models = [
            config.facilitator_model,
            config.debater_a_model,
            config.debater_b_model,
            config.validator_model,
        ]
        preload_results = model_manager.preload_models(
            models=models,
            warmup=config.warmup_models_on_preload,
        )
        failed = {model: status for model, status in preload_results.items() if status != "success"}
        if failed:
            missing = [model for model, status in failed.items() if status == "not_found"]
            other_failures = {model: status for model, status in failed.items() if status.startswith("error:")}

            lines = ["Failed to preload one or more Ollama models."]
            if missing:
                lines.append("Missing models were detected. Pull them with:")
                for model in missing:
                    lines.append(f"  ollama pull {model}")
            if other_failures:
                lines.append("Other preload errors:")
                for model, status in other_failures.items():
                    lines.append(f"  {model}: {status}")
                lines.append(f"Check OLLAMA_BASE_URL={config.ollama_base_url} and ensure Ollama is running.")

            raise RuntimeError("\n".join(lines))

    initial_state: DiscussionState = {
        "topic": topic,
        "transcript": [],
        "search_results": [],
        "search_enabled": search_enabled,
        "search_status_message": search_status_message,
        "validation_log": [],
        "compact_summary": initial_compact_summary,
        "turn_count": 0,
        "max_turns": config.max_turns,
        "next_action": "speak_a",
        "last_decision": FacilitatorDecision(action="speak_a", reason="initial action"),
        "final_summary": None,
        "markdown_path": str(markdown_path),
        "result_markdown_path": None,
        "input_sources": input_sources or [],
        "session_id": sid,
        "last_error": None,
        "continuation_mode": False,
        "continuation_turn_count": 0,
        "continuation_max_turns": config.continuation_rounds,
    }
    return services, initial_state
