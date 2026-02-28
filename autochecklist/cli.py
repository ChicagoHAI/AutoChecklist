"""Command-line interface for autochecklist."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


def _load_jsonl(
    path: str, input_key: str = "input", target_key: str = "target",
) -> List[Dict[str, Any]]:
    """Load JSONL file, remapping keys to standard input/target."""
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            item = {"input": obj.get(input_key, ""), "target": obj.get(target_key, "")}
            for k, v in obj.items():
                if k not in (input_key, target_key):
                    item[k] = v
            data.append(item)
    return data


def _maybe_load_config(args: argparse.Namespace) -> None:
    """If --config is specified, load and register the pipeline config."""
    config_path = getattr(args, "config", None)
    if config_path is not None:
        from autochecklist.registry import load_pipeline_config
        name = load_pipeline_config(config_path)
        # Use the loaded pipeline name if --pipeline wasn't set
        if args.pipeline is None:
            args.pipeline = name


def cmd_run(args: argparse.Namespace) -> None:
    """Run full pipeline: generate checklists + score targets."""
    from autochecklist import pipeline

    _maybe_load_config(args)
    data = _load_jsonl(args.data, args.input_key, args.target_key)

    scorer_kwargs = {}
    if args.scorer_prompt:
        scorer_kwargs["custom_prompt"] = Path(args.scorer_prompt)

    pipe = pipeline(
        task=args.pipeline,
        generator_model=args.generator_model,
        scorer_model=args.scorer_model,
        scorer=args.scorer,
        provider=args.provider,
        base_url=args.base_url,
        api_key=args.api_key,
        api_format=args.api_format,
        custom_prompt=Path(args.generator_prompt) if args.generator_prompt else None,
        scorer_kwargs=scorer_kwargs or None,
    )

    result = pipe.run_batch(
        data=data,
        output_path=args.output,
        show_progress=True,
        overwrite=args.overwrite,
    )

    print(f"\n{'='*40}", file=sys.stderr)
    print(f"Examples:          {len(result.scores)}", file=sys.stderr)
    print(f"Mean score:        {result.mean_score:.4f}", file=sys.stderr)
    print(f"Macro pass rate:   {result.macro_pass_rate:.4f}", file=sys.stderr)
    print(f"Micro pass rate:   {result.micro_pass_rate:.4f}", file=sys.stderr)
    print(f"{'='*40}", file=sys.stderr)

    if args.output:
        print(f"Results written to: {args.output}", file=sys.stderr)


def cmd_generate(args: argparse.Namespace) -> None:
    """Generate checklists only (no scoring)."""
    from autochecklist import pipeline

    _maybe_load_config(args)
    data = _load_jsonl(args.data, args.input_key, target_key="target")

    pipe = pipeline(
        task=args.pipeline,
        generator_model=args.generator_model,
        provider=args.provider,
        base_url=args.base_url,
        api_key=args.api_key,
        api_format=args.api_format,
        custom_prompt=Path(args.generator_prompt) if args.generator_prompt else None,
    )

    checklists = pipe.generate_batch(
        data=data,
        output_path=args.output,
        show_progress=True,
        overwrite=args.overwrite,
    )

    print(f"\nGenerated {len(checklists)} checklists", file=sys.stderr)
    if args.output:
        print(f"Written to: {args.output}", file=sys.stderr)


def cmd_score(args: argparse.Namespace) -> None:
    """Score targets against a pre-existing checklist."""
    from autochecklist import Checklist, get_scorer
    from autochecklist.pipeline import BatchResult

    checklist = Checklist.load(args.checklist)
    data = _load_jsonl(args.data, args.input_key, args.target_key)

    scorer_cls = get_scorer(args.scorer or "batch")
    scorer_kwargs: dict[str, Any] = {}
    if args.scorer_model:
        scorer_kwargs["model"] = args.scorer_model
    if args.provider:
        scorer_kwargs["provider"] = args.provider
    if args.base_url:
        scorer_kwargs["base_url"] = args.base_url
    if args.api_key:
        scorer_kwargs["api_key"] = args.api_key
    if args.api_format:
        scorer_kwargs["api_format"] = args.api_format
    if args.scorer_prompt:
        scorer_kwargs["custom_prompt"] = Path(args.scorer_prompt)

    scorer = scorer_cls(**scorer_kwargs)

    targets = [d["target"] for d in data]
    inputs = [d["input"] for d in data]
    scores = scorer.score_batch(checklist, targets, inputs)

    result = BatchResult(scores=scores, data=data, checklist=checklist)

    print(f"\n{'='*40}", file=sys.stderr)
    print(f"Examples:          {len(result.scores)}", file=sys.stderr)
    print(f"Mean score:        {result.mean_score:.4f}", file=sys.stderr)
    print(f"Macro pass rate:   {result.macro_pass_rate:.4f}", file=sys.stderr)
    print(f"Micro pass rate:   {result.micro_pass_rate:.4f}", file=sys.stderr)
    print(f"{'='*40}", file=sys.stderr)

    if args.output:
        result.to_jsonl(args.output)
        print(f"Results written to: {args.output}", file=sys.stderr)


def cmd_list(args: argparse.Namespace) -> None:
    """List available components."""
    from autochecklist import (
        list_generators_with_info,
        list_refiners_with_info,
        list_scorers_with_info,
    )

    component = args.component

    if component == "generators":
        items = list_generators_with_info()
        print(f"{'Name':<25} {'Level':<10} {'Scorer':<12} Description")
        print("-" * 80)
        for g in items:
            scorer = g.get("default_scorer") or "-"
            print(
                f"{g['name']:<25} {g['level']:<10} "
                f"{scorer:<12} {g.get('description', '')}"
            )
    elif component == "scorers":
        items = list_scorers_with_info()
        print(f"{'Name':<20} {'Method':<15} Description")
        print("-" * 60)
        for s in items:
            print(f"{s['name']:<20} {s.get('method', '-'):<15} {s.get('description', '')}")
    elif component == "refiners":
        items = list_refiners_with_info()
        print(f"{'Name':<20} Description")
        print("-" * 50)
        for r in items:
            print(f"{r['name']:<20} {r.get('description', '')}")


def _add_provider_flags(parser: argparse.ArgumentParser) -> None:
    """Add provider/model flags shared across subcommands."""
    parser.add_argument(
        "--provider", default=None,
        choices=["openrouter", "openai", "vllm"],
        help="LLM provider (default: openrouter)",
    )
    parser.add_argument("--base-url", default=None, help="Custom base URL for provider")
    parser.add_argument("--api-key", default=None, help="API key (default: from env)")
    parser.add_argument(
        "--api-format", default=None,
        choices=["chat", "responses"],
        help="API format",
    )


def _add_io_flags(parser: argparse.ArgumentParser, include_target: bool = True) -> None:
    """Add input/output flags."""
    parser.add_argument("--data", required=True, help="Path to input JSONL file")
    parser.add_argument("-o", "--output", default=None, help="Output JSONL path (enables resume)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output instead of resuming")
    parser.add_argument("--input-key", default="input", help="JSONL key for input field (default: input)")
    if include_target:
        parser.add_argument("--target-key", default="target", help="JSONL key for target field (default: target)")


def main(argv: list[str] | None = None) -> None:
    from dotenv import load_dotenv
    load_dotenv()

    from autochecklist import __version__

    parser = argparse.ArgumentParser(
        prog="autochecklist",
        description="AutoChecklist: checklist-based evaluation with LLMs from the command line",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- run ---
    run_parser = subparsers.add_parser("run", help="Generate checklists and score targets")
    run_parser.add_argument("--pipeline", default=None, help="Pipeline name (e.g. tick, rocketeval, rlcf_direct)")
    run_parser.add_argument("--config", default=None, help="Path to pipeline config JSON file")
    run_parser.add_argument("--generator-model", default=None, help="Model for generator (e.g. openai/gpt-4o-mini)")
    run_parser.add_argument("--scorer-model", default=None, help="Model for scorer")
    run_parser.add_argument("--scorer", default=None, help="Override scorer (batch, item, weighted, normalized)")
    run_parser.add_argument("--generator-prompt", default=None, help="Path to custom generator prompt (.md)")
    run_parser.add_argument("--scorer-prompt", default=None, help="Path to custom scorer prompt (.md)")
    _add_provider_flags(run_parser)
    _add_io_flags(run_parser)
    run_parser.set_defaults(func=cmd_run)

    # --- generate ---
    gen_parser = subparsers.add_parser("generate", help="Generate checklists only (no scoring)")
    gen_parser.add_argument("--pipeline", default=None, help="Pipeline name (e.g. tick, rocketeval)")
    gen_parser.add_argument("--config", default=None, help="Path to pipeline config JSON file")
    gen_parser.add_argument("--generator-model", default=None, help="Model for generator")
    gen_parser.add_argument("--generator-prompt", default=None, help="Path to custom generator prompt (.md)")
    _add_provider_flags(gen_parser)
    _add_io_flags(gen_parser, include_target=False)
    gen_parser.set_defaults(func=cmd_generate)

    # --- score ---
    score_parser = subparsers.add_parser("score", help="Score targets against existing checklist")
    score_parser.add_argument("--checklist", required=True, help="Path to checklist JSON file")
    score_parser.add_argument("--scorer", default=None, help="Scorer to use (batch, item, weighted, normalized)")
    score_parser.add_argument("--scorer-model", default=None, help="Model for scorer")
    score_parser.add_argument("--scorer-prompt", default=None, help="Path to custom scorer prompt (.md)")
    _add_provider_flags(score_parser)
    _add_io_flags(score_parser)
    score_parser.set_defaults(func=cmd_score)

    # --- list ---
    list_parser = subparsers.add_parser("list", help="List available components")
    list_parser.add_argument(
        "--component", default="generators",
        choices=["generators", "scorers", "refiners"],
        help="Component type to list (default: generators)",
    )
    list_parser.set_defaults(func=cmd_list)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
