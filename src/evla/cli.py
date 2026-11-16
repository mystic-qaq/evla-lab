"""``evla`` command line interface.

Every stage of the pipeline is one subcommand, which is what makes an experiment
reproducible: the CLI is the only supported entry point, and every command writes
a JSON artefact under ``results/``. Nothing important lives in a notebook.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml


# ---------------------------------------------------------------------------- utils
def _load_yaml(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _dump_json(obj, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    return path


def _fmt_report(report: dict) -> str:
    keys = ["success_rate", "successes", "episodes", "control_hz", "latency_p50_ms",
            "latency_p95_ms", "deadline_miss_rate", "inference_calls_per_step"]
    return "  ".join(f"{k}={report[k]}" for k in keys if k in report)


# ---------------------------------------------------------------------------- doctor
def cmd_doctor(args: argparse.Namespace) -> int:
    info = {"python": sys.version.split()[0]}
    for mod in ("numpy", "torch", "lerobot", "transformers", "gym_pusht", "imageio", "matplotlib", "peft"):
        try:
            module = __import__(mod)
            info[mod] = getattr(module, "__version__", "installed")
        except Exception:
            info[mod] = None
    try:
        import torch

        info["cuda_available"] = torch.cuda.is_available()
        info["device_count"] = torch.cuda.device_count()
        if torch.cuda.is_available():
            info["device_name"] = torch.cuda.get_device_name(0)
            info["cuda_version"] = torch.version.cuda
    except Exception:
        info["cuda_available"] = False
    print(json.dumps(info, indent=2))
    return 0


# ---------------------------------------------------------------------------- toy flow
def cmd_toy_collect(args: argparse.Namespace) -> int:
    from evla.data.demos import collect_toy_demos, save_demos

    demos = collect_toy_demos(n_episodes=args.episodes, seed=args.seed, drift=args.drift)
    out = Path(args.out)
    save_demos(out, demos)
    print(f"collected {len(demos['actions'])} transitions from {args.episodes} episodes -> {out}")
    return 0


def _toy_train(args: argparse.Namespace) -> tuple[object, dict]:
    import torch

    from evla.data.demos import build_chunks, normalization_stats
    from evla.models.flow_chunk import FlowChunkPolicy

    data = np.load(args.demos)
    states, actions, ep_ids = data["states"], data["actions"], data["episode_ids"]
    s_mean, s_std = normalization_stats(states)
    a_mean, a_std = normalization_stats(actions)
    # Chunks are cut in *raw* units: the policy normalises internally, so passing
    # pre-normalised targets would apply the statistics twice (a silent 1e-2 scale bug).
    xs, ys = build_chunks(states, actions, ep_ids, args.chunk_size)

    device = args.device
    policy = FlowChunkPolicy(
        state_dim=states.shape[1],
        action_dim=actions.shape[1],
        chunk_size=args.chunk_size,
        num_inference_steps=args.sampling_steps,
        device=device,
        state_mean=s_mean, state_std=s_std, action_mean=a_mean, action_std=a_std,
    )
    opt = torch.optim.AdamW(policy.net.parameters(), lr=args.lr, weight_decay=1e-4)
    n = len(xs)
    rng = np.random.default_rng(0)
    history = []
    policy.net.train()
    for step in range(1, args.steps + 1):
        idx = rng.integers(0, n, size=args.batch_size)
        lr = args.lr * (0.5 * (1 + np.cos(np.pi * step / args.steps)))
        for group in opt.param_groups:
            group["lr"] = lr
        loss = policy.loss(xs[idx], ys[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.net.parameters(), 1.0)
        opt.step()
        if step % max(1, args.steps // 10) == 0 or step == 1:
            history.append({"step": step, "loss": round(float(loss.item()), 5), "lr": round(lr, 6)})
            print(f"  step {step:5d}/{args.steps} loss={loss.item():.5f}", flush=True)
    policy.net.eval()
    return policy, {"history": history, "n_transitions": int(n), "chunk_size": args.chunk_size}


def cmd_toy_train(args: argparse.Namespace) -> int:
    policy, meta = _toy_train(args)
    out = Path(args.out)
    policy.save(out, extra=meta)
    print(f"saved policy -> {out.with_suffix('.npz')}")

    if args.eval_episodes <= 0:
        return 0
    from evla.eval.runner import evaluate
    from evla.optimize.profiler import InstrumentedPolicy

    result = evaluate(
        lambda: __import__("evla.envs.registry", fromlist=["make_env"]).make_env("toy_reach"),
        InstrumentedPolicy(policy, warmup=1),
        label=f"toy_flow_h{policy.chunk_size}",
        n_episodes=args.eval_episodes,
        mode="open_loop",
        progress=False,
    )
    print(f"eval: success {result.rate_estimate} | control {result.control_hz:.1f} Hz")
    if args.report_dir:
        result.save(Path(args.report_dir) / f"{result.label}.json")
    return 0


def cmd_toy_eval(args: argparse.Namespace) -> int:
    from evla.envs.registry import make_env
    from evla.eval.report import write_report
    from evla.eval.runner import evaluate
    from evla.models.flow_chunk import FlowChunkPolicy
    from evla.optimize.profiler import InstrumentedPolicy

    policy = FlowChunkPolicy.load(args.policy, device="cpu")
    out_dir = Path(args.out)
    results = []
    for mode, exec_steps in (("receding", 1), ("open_loop", policy.chunk_size), ("temporal_ens", 8)):
        wrapped = InstrumentedPolicy(policy, warmup=1)
        res = evaluate(
            lambda: make_env("toy_reach", drift=args.drift),
            wrapped,
            label=f"{mode}-{exec_steps}",
            n_episodes=args.episodes,
            mode=mode,
            execute_steps=exec_steps,
            progress=False,
        )
        res.save(out_dir / f"{res.label}.json")
        results.append(res)
        print(f"  {res.label:16s} {res.rate_estimate}  {res.control_hz:6.1f} Hz")
    print(write_report(results, out_dir, title="Toy chunking study").read_text(encoding="utf-8")[:400])
    return 0


# ---------------------------------------------------------------------------- finetune
def cmd_finetune(args: argparse.Namespace) -> int:
    """Wrap ``lerobot-train`` with the exact flags this repository expects.

    Using the upstream trainer instead of re-implementing it is a deliberate
    choice: the value added here is the evaluation / optimization / deployment
    stack, not a copy of someone's training loop. The generated command is printed
    so the run is reproducible by hand.
    """
    cfg = _load_yaml(args.config)
    policy = cfg.get("policy", {})
    dataset = cfg.get("dataset", {})
    train = cfg.get("train", {})
    cmd = [
        args.trainer,
        f"--policy.path={policy['base_checkpoint']}",
        f"--policy.repo_id={policy.get('repo_id', 'local/evla')}",
        "--policy.push_to_hub=false",
        f"--policy.device={policy.get('device', 'cuda')}",
        f"--policy.empty_cameras={policy.get('empty_cameras', 2)}",
        f"--policy.chunk_size={policy.get('chunk_size', 50)}",
        f"--policy.n_action_steps={policy.get('n_action_steps', 50)}",
        f"--dataset.repo_id={dataset['repo_id']}",
        f"--dataset.root={dataset['root']}",
        f"--dataset.video_backend={dataset.get('video_backend', 'pyav')}",
        f"--rename_map={json.dumps(dataset.get('rename_map', {'observation.image': 'observation.images.camera1'}))}",
        f"--batch_size={train.get('batch_size', 32)}",
        f"--steps={train.get('steps', 20000)}",
        f"--save_freq={train.get('save_freq', 4000)}",
        f"--log_freq={train.get('log_freq', 100)}",
        f"--num_workers={train.get('num_workers', 6)}",
        f"--output_dir={train.get('output_dir', 'outputs/smolvla_pusht')}",
        f"--job_name={train.get('job_name', 'smolvla_pusht')}",
        "--wandb.enable=false",
    ]
    printable = " \\\n  ".join(cmd)
    print(printable)
    if args.dry_run:
        return 0
    return subprocess.call(cmd)


# ---------------------------------------------------------------------------- bench
def _make_observation(env_name: str, seed: int = 0):
    from evla.envs.registry import make_env

    env = make_env(env_name)
    obs = env.reset(seed=seed)
    spec = env.spec()
    env.close()
    return obs, spec


def cmd_bench(args: argparse.Namespace) -> int:
    from evla.models.registry import load_policy
    from evla.optimize.apply import OptimizationSpec, apply_optimizations
    from evla.optimize.profiler import InstrumentedPolicy

    cfg = _load_yaml(args.config) if args.config else {}
    policy_cfg = dict(cfg.get("policy", {}))
    if args.checkpoint:
        policy_cfg["checkpoint"] = args.checkpoint
    policy = load_policy(policy_cfg)
    spec = OptimizationSpec.from_dict({**cfg.get("optimize", {}), **(cfg.get("sweep_one") or {})})
    policy, applied = apply_optimizations(policy, spec)
    obs, env_spec = _make_observation(cfg.get("env", {}).get("name", "pusht")
                                     if isinstance(cfg.get("env"), dict) else cfg.get("env", "pusht"))
    profiled = InstrumentedPolicy(policy, warmup=args.warmup)
    for _ in range(args.iterations):
        profiled.predict_chunk(obs)
    summary = profiled.stats.summary()
    if hasattr(policy, "peak_memory_mb"):
        summary["peak_memory_mb"] = policy.peak_memory_mb()  # type: ignore[attr-defined]
    out = {
        "label": spec.key(),
        "policy": profiled.describe(),
        "env": env_spec,
        "optimizations": applied.as_dict(),
        "latency": summary,
        "budget_hz_at_p95": round(1000.0 / max(1e-6, summary.get("p95_ms", 1.0)), 2),
    }
    print(json.dumps(out, indent=2))
    if args.out:
        _dump_json(out, args.out)
    return 0


# ---------------------------------------------------------------------------- eval
def _build_policy(policy_cfg: dict, optimize_cfg: dict | None, profile: bool = True):
    from evla.models.registry import load_policy
    from evla.optimize.apply import OptimizationSpec, apply_optimizations
    from evla.optimize.profiler import InstrumentedPolicy

    policy = load_policy(dict(policy_cfg))
    applied = None
    if optimize_cfg:
        policy, applied = apply_optimizations(policy, OptimizationSpec.from_dict(optimize_cfg))
    return (InstrumentedPolicy(policy, warmup=3) if profile else policy), applied


def cmd_eval(args: argparse.Namespace) -> int:
    from evla.envs.registry import make_env
    from evla.eval.runner import evaluate

    cfg = _load_yaml(args.config) if args.config else {}
    policy_cfg = dict(cfg.get("policy", {}))
    if args.checkpoint:
        policy_cfg["checkpoint"] = args.checkpoint
    eval_cfg = dict(cfg.get("eval", {}))
    env_cfg = cfg.get("env", "pusht")
    policy, applied = _build_policy(policy_cfg, cfg.get("optimize"), profile=True)

    n = int(eval_cfg.get("episodes", 20))
    seed0 = int(eval_cfg.get("seed0", 0))
    label = args.label or eval_cfg.get("label") or _auto_label(str(policy_cfg.get("checkpoint", "policy")))
    video = eval_cfg.get("video")
    video_path = None
    if video:
        video_path = Path(video) if not isinstance(video, bool) else Path("results/videos") / f"{label}.mp4"
    result = evaluate(
        lambda: make_env(env_cfg),
        policy,
        label=label,
        n_episodes=n,
        seeds=list(range(seed0, seed0 + n)),
        mode=eval_cfg.get("mode", "open_loop"),
        execute_steps=eval_cfg.get("execute_steps"),
        decay=float(eval_cfg.get("ens_decay", 0.5)),
        video_path=video_path,
        progress=not args.quiet,
    )
    payload = result.as_dict()
    if applied is not None:
        payload["optimizations"] = applied.as_dict()
    out = Path(args.out) if args.out else Path("results") / f"{_safe(label)}.json"
    _dump_json(payload, out)
    print(f"\n{_safe(label)}: success {result.rate_estimate} | control {result.control_hz:.1f} Hz "
          f"| p50 {result.latency.get('p50_ms', '-')} ms -> {out}")
    return 0


def _auto_label(checkpoint: str) -> str:
    """Turn ``outputs/run/checkpoints/020000/pretrained_model`` into ``run-020000``."""
    parts = Path(checkpoint).parts
    if len(parts) >= 3 and parts[-1] == "pretrained_model" and parts[-3] == "checkpoints":
        return f"{parts[-4]}-{parts[-2]}"
    if len(parts) >= 2 and parts[-1] == "pretrained_model":
        return parts[-2]
    return Path(checkpoint).name or "policy"


def _release_gpu() -> None:
    """Drop the previous policy from the cache between sweep entries.

    Each config instantiates a fresh 450M model; without this a long sweep slowly
    fragments the caching allocator and eventually OOMs on the last entry.
    """
    try:
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "-" for c in str(name))


# ---------------------------------------------------------------------------- sweep
def cmd_sweep(args: argparse.Namespace) -> int:
    """Run a matrix of (optimization x execution) configs and report them together."""
    from evla.envs.registry import make_env
    from evla.eval.leaderboard import build_leaderboard, to_markdown
    from evla.eval.report import write_report
    from evla.eval.runner import evaluate

    cfg = _load_yaml(args.config)
    base_policy = dict(cfg.get("policy", {}))
    base_eval = dict(cfg.get("eval", {}))
    env_cfg = cfg.get("env", "pusht")
    sweep = cfg.get("sweep", [])
    if not sweep:
        raise SystemExit("config has no `sweep` list")
    out_dir = Path(args.out or cfg.get("out", "results/sweep"))
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for i, entry in enumerate(sweep):
        entry = dict(entry)
        label = entry.pop("label", f"cfg{i}")
        optimize = entry.pop("optimize", None)
        policy_cfg = {**base_policy, **{k: entry.pop(k) for k in list(entry) if k in
                                        ("checkpoint", "precision", "num_flow_steps", "device", "type", "image_key")}}
        eval_cfg = {**base_eval, **entry}
        policy, applied = _build_policy(policy_cfg, optimize, profile=True)
        n = int(eval_cfg.get("episodes", 20))
        seed0 = int(eval_cfg.get("seed0", 0))
        video = eval_cfg.get("video")
        video_path = (out_dir / "videos" / f"{_safe(label)}.mp4") if video else None
        print(f"\n=== {label} ===", flush=True)
        res = evaluate(
            lambda: make_env(env_cfg),
            policy,
            label=label,
            n_episodes=n,
            seeds=list(range(seed0, seed0 + n)),
            mode=eval_cfg.get("mode", "open_loop"),
            execute_steps=eval_cfg.get("execute_steps"),
            decay=float(eval_cfg.get("ens_decay", 0.5)),
            video_path=video_path,
            progress=True,
        )
        payload = res.as_dict()
        if applied is not None:
            payload["optimizations"] = applied.as_dict()
        _dump_json(payload, out_dir / f"{_safe(label)}.json")
        results.append((res, payload))
        print(f"--- {label}: success {res.rate_estimate} | {res.control_hz:.1f} Hz", flush=True)
        _release_gpu()

    only = [r for r, _ in results]
    report = write_report(only, out_dir, title=cfg.get("title", "Sweep report"))
    board = build_leaderboard(only, reference=cfg.get("reference"))
    (out_dir / "LEADERBOARD.md").write_text(to_markdown(board), encoding="utf-8")
    (out_dir / "leaderboard.json").write_text(json.dumps(board, indent=2), encoding="utf-8")
    print(f"\nreport -> {report}\nleaderboard -> {out_dir / 'LEADERBOARD.md'}")
    return 0


# ---------------------------------------------------------------------------- report
def cmd_report(args: argparse.Namespace) -> int:
    from evla.eval.leaderboard import build_leaderboard, to_markdown
    from evla.eval.report import load_results, write_report

    results = load_results([args.results])
    if not results:
        raise SystemExit(f"no result JSONs under {args.results}")
    out = Path(args.out or args.results)
    path = write_report(results, out, title=args.title)
    board = build_leaderboard(results, reference=args.reference)
    (out / "LEADERBOARD.md").write_text(to_markdown(board), encoding="utf-8")
    print(path.read_text(encoding="utf-8"))
    return 0


def cmd_leaderboard(args: argparse.Namespace) -> int:
    from evla.eval.leaderboard import build_leaderboard, to_markdown
    from evla.eval.report import load_results

    board = build_leaderboard(load_results([args.results]), reference=args.reference)
    text = to_markdown(board)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    return 0


# ---------------------------------------------------------------------------- deploy
def cmd_serve(args: argparse.Namespace) -> int:
    from evla.deploy.server import PolicyServer
    from evla.models.registry import load_policy

    cfg = _load_yaml(args.config) if args.config else {}
    policy_cfg = dict(cfg.get("policy", {}))
    if args.checkpoint:
        policy_cfg["checkpoint"] = args.checkpoint
    if args.precision:
        policy_cfg["precision"] = args.precision
    policy = load_policy(policy_cfg)
    if hasattr(policy, "warmup"):
        policy.warmup(3)  # type: ignore[attr-defined]
    server = PolicyServer(policy, host=args.host, port=args.port)
    print(json.dumps(policy.describe(), indent=2))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()
    return 0


def cmd_deploy(args: argparse.Namespace) -> int:
    """Run the full runtime against sim (or a real SO-101 when ``--backend so101``)."""
    from evla.deploy.hal.sim import SimBackend
    from evla.deploy.runtime import DeployRuntime, RuntimeConfig
    from evla.deploy.safety import SafetyConfig
    from evla.deploy.server import RemotePolicy
    from evla.envs.registry import make_env
    from evla.models.registry import load_policy

    cfg = _load_yaml(args.config) if args.config else {}
    rt_cfg = RuntimeConfig.from_dict({**cfg.get("runtime", {}), **({"episodes": args.episodes} if args.episodes else {})})
    safety_cfg = SafetyConfig.from_dict(cfg.get("safety"))

    if args.backend == "so101":
        from evla.deploy.hal.so101 import SO101Backend

        hw = cfg.get("hardware", {})
        backend = SO101Backend(**hw)
    else:
        backend = SimBackend(lambda: make_env(cfg.get("env", "pusht")), seed=rt_cfg.seed)
    backend.connect()

    if args.remote:
        policy = RemotePolicy(args.remote_host, args.remote_port)
    else:
        policy = load_policy(dict(cfg.get("policy", {})))
    if hasattr(policy, "warmup"):
        policy.warmup(3)  # type: ignore[attr-defined]

    runtime = DeployRuntime(backend, policy, rt_cfg, safety_config=safety_cfg, video_path=args.video)
    try:
        report = runtime.run()
    finally:
        backend.disconnect()
    print(json.dumps(report.as_dict(), indent=2))
    if args.out:
        _dump_json(report.as_dict(), args.out)
    return 0


# ---------------------------------------------------------------------------- quickstart
def cmd_quickstart(args: argparse.Namespace) -> int:
    """CPU-only end-to-end demo: collect -> train -> evaluate -> report.

    This exists so a reviewer (or CI) can verify the entire pipeline in under a
    minute without a GPU or a robot.
    """
    tmp = Path(args.out)
    tmp.mkdir(parents=True, exist_ok=True)
    demos = tmp / "toy_demos.npz"
    if not demos.exists():
        cmd_toy_collect(argparse.Namespace(episodes=args.episodes, seed=0, drift=0.012, out=str(demos)))
    ckpt = tmp / "toy_flow.npz"
    if not ckpt.exists():
        cmd_toy_train(argparse.Namespace(
            demos=str(demos), out=str(tmp / "toy_flow"), steps=args.steps, batch_size=128,
            lr=3e-3, chunk_size=8, sampling_steps=4, device="cpu", eval_episodes=0, report_dir=None,
        ))
    return cmd_toy_eval(argparse.Namespace(
        policy=str(ckpt), out=str(tmp / "eval"), episodes=args.eval_episodes, drift=args.drift,
    ))


# ---------------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="evla", description="End-to-end VLA fine-tuning / optimization / deployment lab")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("doctor", help="print environment diagnostics")
    d.set_defaults(func=cmd_doctor)

    q = sub.add_parser("quickstart", help="CPU-only end-to-end toy pipeline")
    q.add_argument("--out", default="outputs/quickstart")
    q.add_argument("--episodes", type=int, default=120)
    q.add_argument("--steps", type=int, default=4000)
    q.add_argument("--eval-episodes", type=int, default=20, dest="eval_episodes")
    q.add_argument("--drift", type=float, default=0.012)
    q.set_defaults(func=cmd_quickstart)

    c = sub.add_parser("toy-collect", help="collect scripted demonstrations on the toy env")
    c.add_argument("--episodes", type=int, default=160)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--drift", type=float, default=0.012)
    c.add_argument("--out", default="outputs/toy_demos.npz")
    c.set_defaults(func=cmd_toy_collect)

    t = sub.add_parser("toy-train", help="train the flow-matching chunk policy on toy demos")
    t.add_argument("--demos", default="outputs/toy_demos.npz")
    t.add_argument("--out", default="outputs/toy_flow")
    t.add_argument("--steps", type=int, default=4000)
    t.add_argument("--batch-size", type=int, default=64, dest="batch_size")
    t.add_argument("--lr", type=float, default=3e-3)
    t.add_argument("--chunk-size", type=int, default=8, dest="chunk_size")
    t.add_argument("--sampling-steps", type=int, default=8, dest="sampling_steps")
    t.add_argument("--device", default="cpu")
    t.add_argument("--eval-episodes", type=int, default=20, dest="eval_episodes")
    t.add_argument("--report-dir", default=None, dest="report_dir")
    t.set_defaults(func=cmd_toy_train)

    te = sub.add_parser("toy-eval", help="compare execution modes on the toy env")
    te.add_argument("--policy", default="outputs/toy_flow.npz")
    te.add_argument("--episodes", type=int, default=30)
    te.add_argument("--drift", type=float, default=0.012,
                    help="target speed; raise it to stress test reactivity vs chunk length")
    te.add_argument("--out", default="outputs/toy_eval")
    te.set_defaults(func=cmd_toy_eval)

    f = sub.add_parser("finetune", help="launch the SmolVLA fine-tune through lerobot-train")
    f.add_argument("-c", "--config", required=True)
    f.add_argument("--trainer", default="lerobot-train")
    f.add_argument("--dry-run", action="store_true", dest="dry_run")
    f.set_defaults(func=cmd_finetune)

    b = sub.add_parser("bench", help="latency microbenchmark for one optimization config")
    b.add_argument("-c", "--config")
    b.add_argument("--checkpoint")
    b.add_argument("--iterations", type=int, default=30)
    b.add_argument("--warmup", type=int, default=5)
    b.add_argument("--out")
    b.set_defaults(func=cmd_bench)

    e = sub.add_parser("eval", help="evaluate one policy configuration")
    e.add_argument("-c", "--config")
    e.add_argument("--checkpoint")
    e.add_argument("--label", default=None)
    e.add_argument("--out")
    e.add_argument("--quiet", action="store_true")
    e.set_defaults(func=cmd_eval)

    s = sub.add_parser("sweep", help="run a config matrix and write report + leaderboard")
    s.add_argument("-c", "--config", required=True)
    s.add_argument("--out")
    s.set_defaults(func=cmd_sweep)

    r = sub.add_parser("report", help="rebuild the markdown report from result JSONs")
    r.add_argument("--results", default="results")
    r.add_argument("--out")
    r.add_argument("--title", default="Evaluation report")
    r.add_argument("--reference", default=None)
    r.set_defaults(func=cmd_report)

    lb = sub.add_parser("leaderboard", help="paired leaderboard from result JSONs")
    lb.add_argument("--results", default="results")
    lb.add_argument("--reference", default=None)
    lb.add_argument("--out")
    lb.set_defaults(func=cmd_leaderboard)

    sv = sub.add_parser("serve", help="serve a policy over TCP")
    sv.add_argument("-c", "--config")
    sv.add_argument("--checkpoint")
    sv.add_argument("--precision", default=None)
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=5599)
    sv.set_defaults(func=cmd_serve)

    dp = sub.add_parser("deploy", help="run the fixed-rate runtime against sim or SO-101")
    dp.add_argument("-c", "--config")
    dp.add_argument("--backend", default="sim", choices=["sim", "so101"])
    dp.add_argument("--episodes", type=int, default=None)
    dp.add_argument("--video", default=None)
    dp.add_argument("--remote", action="store_true")
    dp.add_argument("--remote-host", default="127.0.0.1", dest="remote_host")
    dp.add_argument("--remote-port", type=int, default=5599, dest="remote_port")
    dp.add_argument("--out", default=None)
    dp.set_defaults(func=cmd_deploy)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
