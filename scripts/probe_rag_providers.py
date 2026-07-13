"""Run redacted, real provider probes from one strict local RAG config."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from pydantic import ValidationError


PROJECT_ROOT_FROM_SCRIPT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT_FROM_SCRIPT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FROM_SCRIPT))

from src.rag.parent_child.provider_probe import (  # noqa: E402
    LlmProbeConfig,
    ProviderProbeError,
    run_provider_probe,
)


def _embedding_probe_batch_size(raw_value: str) -> int:
    """Parse an explicitly requested probe batch without applying a default."""

    try:
        value = int(raw_value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "embedding probe batch size must be an integer"
        ) from exc
    if value < 2:
        raise argparse.ArgumentTypeError(
            "embedding probe batch size must be at least two"
        )
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--index-config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--embedding-probe-batch-size", type=_embedding_probe_batch_size
    )
    parser.add_argument("--probe-llm", action="store_true")
    parser.add_argument("--llm-provider")
    parser.add_argument("--llm-protocol")
    parser.add_argument("--llm-model")
    parser.add_argument("--llm-base-url")
    parser.add_argument("--llm-endpoint-path")
    parser.add_argument("--llm-api-key-env")
    parser.add_argument("--llm-timeout-seconds", type=float)
    return parser


def _llm_config_from_args(args: argparse.Namespace) -> LlmProbeConfig | None:
    """Build chat probe config only when every explicit LLM field is present."""

    values = (
        args.llm_provider,
        args.llm_protocol,
        args.llm_model,
        args.llm_base_url,
        args.llm_endpoint_path,
        args.llm_api_key_env,
        args.llm_timeout_seconds,
    )
    if not args.probe_llm:
        if any(value is not None for value in values):
            raise ProviderProbeError("LLM options require --probe-llm")
        return None
    if any(value is None for value in values):
        return None
    try:
        return LlmProbeConfig(
            provider=args.llm_provider,
            protocol=args.llm_protocol,
            model=args.llm_model,
            base_url=args.llm_base_url,
            endpoint_path=args.llm_endpoint_path,
            api_key_env=args.llm_api_key_env,
            timeout_seconds=args.llm_timeout_seconds,
        )
    except ValidationError:
        return None


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = run_provider_probe(
            project_root=args.project_root,
            index_config_path=args.index_config,
            run_id=args.run_id,
            output_directory=args.output_dir,
            embedding_probe_batch_size=args.embedding_probe_batch_size,
            probe_llm_enabled=args.probe_llm,
            llm_config=_llm_config_from_args(args),
        )
    except (OSError, ProviderProbeError, ValueError, ValidationError) as exc:
        # Deliberately omit exception text: it can contain provider-originated data.
        print(f"Provider probe failed before report writing: {type(exc).__name__}")
        return 2
    print(f"Provider probe completed: success={str(report.success).lower()}")
    return 0 if report.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
