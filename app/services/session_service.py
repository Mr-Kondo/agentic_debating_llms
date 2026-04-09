"""Session lifecycle and dependency wiring."""

from __future__ import annotations

from dataclasses import dataclass
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
    langfuse.start_trace(session_id=sid, topic=topic)

    markdown_logger = MarkdownLogger(output_dir=config.markdown_log_dir_path)
    markdown_path = markdown_logger.create_session_file(session_id=sid, topic=topic)

    search_service = SearchService(
        command_template=config.search_command_template,
        timeout_seconds=config.search_timeout_seconds,
        digester=DefaultSearchDigester(),
    )

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
        models = [config.facilitator_model, config.debater_a_model, config.debater_b_model]
        model_manager.preload_models(models=models, warmup=config.warmup_models_on_preload)

    initial_state: DiscussionState = {
        "topic": topic,
        "transcript": [],
        "search_results": [],
        "compact_summary": "",
        "turn_count": 0,
        "max_turns": config.max_turns,
        "next_action": "speak_a",
        "last_decision": FacilitatorDecision(action="speak_a", reason="initial action"),
        "final_summary": None,
        "markdown_path": str(markdown_path),
        "session_id": sid,
        "last_error": None,
    }
    return services, initial_state
