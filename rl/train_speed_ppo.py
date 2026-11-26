"""Train the camera car's speed governor with PPO in the stand-in town (rl/speed_env.py).

Runs in WSL with the detector's PyTorch environment (CPU is enough; the network is tiny):

  ~/.venvs/idd-detect/bin/python rl/train_speed_ppo.py --output runs/rl_speed_<stamp>

Every --eval-every iterations the greedy policy drives a fixed set of held-out episodes (the town's
place-to-place routes, seeds never used in training) next to the rule driver; the checkpoint kept as
best is the fastest one with no more collisions, near misses or rule breaks than the rule driver there.
The kept policy is exported for numpy (integration/india_rl_speed.SpeedPolicy) as policy.npz.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "rl"))
sys.path.insert(0, str(ROOT / "integration"))
from india_rl_speed import ACTIONS, OBS_SIZE  # noqa: E402
from speed_env import Routes, TownSpeedEnv, run_episodes  # noqa: E402

torch.set_num_threads(int(os.environ.get("RL_THREADS", "4")))


class RunningNorm:
    def __init__(self, size):
        self.mean, self.var, self.count = np.zeros(size), np.ones(size), 1e-4

    def update(self, x):
        b_mean, b_var, b_count = x.mean(0), x.var(0), len(x)
        delta = b_mean - self.mean
        total = self.count + b_count
        self.mean = self.mean + delta * b_count / total
        self.var = (self.var * self.count + b_var * b_count + delta ** 2 * self.count * b_count / total) / total
        self.count = total

    @property
    def std(self):
        return np.sqrt(self.var) + 1e-6

    def __call__(self, x):
        return np.clip((x - self.mean) / self.std, -10., 10.)


def mlp(sizes, out_gain=1.):
    layers = []
    for i in range(len(sizes) - 1):
        lin = nn.Linear(sizes[i], sizes[i + 1])
        nn.init.orthogonal_(lin.weight, math.sqrt(2) if i < len(sizes) - 2 else out_gain)
        nn.init.zeros_(lin.bias)
        layers.append(lin)
        if i < len(sizes) - 2:
            layers.append(nn.Tanh())
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    def __init__(self, obs, actions, hidden=128):
        super().__init__()
        self.pi = mlp([obs, hidden, hidden, actions], out_gain=.01)
        self.v = mlp([obs, hidden, hidden, 1], out_gain=1.)

    def forward(self, x):
        return self.pi(x), self.v(x).squeeze(-1)


def export(model: ActorCritic, norm: RunningNorm, path: Path, meta: dict):
    weights = {}
    linears = [m for m in model.pi if isinstance(m, nn.Linear)]
    for i, lin in enumerate(linears):
        weights[f"w{i}"] = lin.weight.detach().cpu().numpy().T.astype(np.float32)
        weights[f"b{i}"] = lin.bias.detach().cpu().numpy().astype(np.float32)
    np.savez(path, obs_mean=norm.mean.astype(np.float32), obs_std=norm.std.astype(np.float32),
             meta=np.array(json.dumps(meta)), **weights)


def summarise(st):
    n = len(st["time"])
    return {"episodes": n, "time_s": float(st["time"].mean()), "arrived": float(st["arrived"].mean()),
            "collisions": int(st["collisions"].sum()), "near_miss_s": float(st["near_miss_s"].sum()),
            "person_near_miss_s": float(st["person_near_miss_s"].sum()), "hard_brake_s": float(st["hard_brake_s"].mean()),
            "emergency_s": float(st["emergency_s"].mean()), "speeding_s": float(st["speeding_s"].mean()),
            "violations": int(st["violations"].sum()), "timeouts": int(st["timeout"].sum()),
            "over_limit_s": float(st["over_limit_s"].mean()), "max_over_m_s": float(st["max_over"].max()),
            "overshoot": int(st["overshoot"].sum()), "jerk2": float(st["jerk2"].mean()),
            "min_gap_m": float(np.min(st["min_gap"])), "min_person_clear_m": float(np.min(st["min_person_clear"])),
            "return": float(st["return"].mean())}


def eval_set(routes, n, seed, start_m=500.):
    ids = np.flatnonzero(routes.split == 1)
    rng = np.random.default_rng(seed)
    eps = []
    for i in range(n):
        k = int(ids[i % len(ids)])
        span = start_m if start_m else rng.uniform(150., 650.)
        eps.append((k, max(0., float(routes.length[k]) - span), 10_000_000 + seed * 10_000 + i))
    return eps


def evaluate(routes, policy_fn, mode, episodes, wait_median, limit_hold=False, entry_ease=False):
    env = TownSpeedEnv(len(episodes), routes, split="all", wait_median=wait_median, limit_hold=limit_hold, entry_ease=entry_ease)
    env.set_fixed_episodes(episodes)
    return summarise(run_episodes(env, mode, policy_fn))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--envs", type=int, default=512)
    ap.add_argument("--steps", type=int, default=128, help="decisions per env per rollout")
    ap.add_argument("--iterations", type=int, default=300)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=.995)
    ap.add_argument("--lam", type=float, default=.95)
    ap.add_argument("--clip", type=float, default=.2)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatch", type=int, default=8192)
    ap.add_argument("--entropy", type=float, default=.01)
    ap.add_argument("--entropy-final", type=float, default=.001)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--wait-median", type=float, default=8.)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--eval-episodes", type=int, default=330)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--init", type=Path, help="continue from this checkpoint.pt")
    ap.add_argument("--people-scale", type=float, default=1.)
    ap.add_argument("--vehicle-scale", type=float, default=1.)
    ap.add_argument("--phantom-scale", type=float, default=1.)
    ap.add_argument("--limit-hold", action="store_true", help="the car raises its set-point to hold a posted limit")
    ap.add_argument("--entry-ease", action="store_true", help="the car reaches the junction speed at the junction's entry")
    ap.add_argument("--stress", type=float, default=0., help="per-episode extra jaywalkers / missed detections (0-1)")
    ap.add_argument("--w-hard", type=float, default=.03, help="reward weight on braking harder than 2.5 m/s^2")
    ap.add_argument("--w-emergency", type=float, default=.05, help="reward weight per 0.05 s of emergency braking")
    args = ap.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    routes = Routes()
    env = TownSpeedEnv(args.envs, routes, split="train", seed=args.seed, wait_median=args.wait_median,
                       people_scale=args.people_scale, vehicle_scale=args.vehicle_scale, phantom_scale=args.phantom_scale,
                       limit_hold=args.limit_hold, entry_ease=args.entry_ease, stress=args.stress,
                       w_hard=args.w_hard, w_emergency=args.w_emergency)
    model = ActorCritic(OBS_SIZE, len(ACTIONS), args.hidden)
    norm = RunningNorm(OBS_SIZE)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)
    start_iter = 0
    if args.init:
        ck = torch.load(args.init, weights_only=False)
        model.load_state_dict(ck["model"])
        norm.mean, norm.var, norm.count = ck["norm"]
    config = vars(args) | {"output": str(out), "init": str(args.init) if args.init else None, "actions": ACTIONS.tolist(),
                           "obs_size": OBS_SIZE, "routes": int(len(routes.length))}
    (out / "config.json").write_text(json.dumps(config, indent=1))
    log = (out / "train_log.jsonl").open("a")
    episodes = eval_set(routes, args.eval_episodes, seed=7)
    baseline = {m: evaluate(routes, None, m, episodes, 18.) for m in ("rule", "legal", "shield")}
    if args.limit_hold or args.entry_ease:
        baseline["shield_hold"] = evaluate(routes, None, "shield", episodes, 18., limit_hold=args.limit_hold, entry_ease=args.entry_ease)
    (out / "baselines.json").write_text(json.dumps(baseline, indent=1))
    print("baselines", json.dumps({m: {k: round(v, 2) if isinstance(v, float) else v for k, v in b.items()} for m, b in baseline.items()}), flush=True)
    rule = baseline["rule"]
    best = None
    obs = env.reset()
    T, N = args.steps, args.envs
    ep_returns, ep_ret = [], np.zeros(N)
    t0 = time.time()
    for it in range(start_iter, args.iterations):
        frac = it / max(1, args.iterations - 1)
        for g in opt.param_groups:
            g["lr"] = args.lr * (1. - .9 * frac)
        ent_coef = args.entropy + (args.entropy_final - args.entropy) * frac
        buf_obs = np.zeros((T, N, OBS_SIZE), np.float32)
        buf_act = np.zeros((T, N), np.int64)
        buf_logp = np.zeros((T, N), np.float32)
        buf_val = np.zeros((T + 1, N), np.float32)
        buf_rew = np.zeros((T, N), np.float32)
        buf_done = np.zeros((T, N), np.float32)
        buf_trunc_val = np.zeros((T, N), np.float32)
        for t in range(T):
            norm.update(obs)
            x = torch.as_tensor(norm(obs), dtype=torch.float32)
            with torch.no_grad():
                logits, value = model(x)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
            buf_obs[t], buf_act[t], buf_logp[t], buf_val[t] = x.numpy(), action.numpy(), dist.log_prob(action).numpy(), value.numpy()
            timeout_before = env.stats["timeout"].copy()
            reward, ended = env.step(action.numpy(), mode="rl")
            trunc = env.stats["timeout"] > timeout_before
            buf_rew[t] = reward
            buf_done[t] = ended & ~trunc
            if trunc.any():
                # a timeout is not the end of the task: bootstrap from where it stopped
                with torch.no_grad():
                    _, v_end = model(torch.as_tensor(norm(env.observe()), dtype=torch.float32))
                buf_trunc_val[t] = np.where(trunc, v_end.numpy(), 0.)
                buf_done[t] = np.where(trunc, 1., buf_done[t])
            ep_ret += reward
            for i in np.flatnonzero(ended):
                ep_returns.append(ep_ret[i]); ep_ret[i] = 0.
            obs = env.reset_done()
        with torch.no_grad():
            _, last_v = model(torch.as_tensor(norm(obs), dtype=torch.float32))
        buf_val[T] = last_v.numpy()
        adv = np.zeros((T, N), np.float32)
        gae = np.zeros(N, np.float32)
        for t in range(T - 1, -1, -1):
            nonterminal = 1. - buf_done[t]
            next_v = buf_val[t + 1] * nonterminal + buf_trunc_val[t]
            delta = buf_rew[t] + args.gamma * next_v - buf_val[t]
            gae = delta + args.gamma * args.lam * nonterminal * gae
            adv[t] = gae
        ret = adv + buf_val[:T]
        b_obs = torch.as_tensor(buf_obs.reshape(-1, OBS_SIZE))
        b_act = torch.as_tensor(buf_act.reshape(-1))
        b_logp = torch.as_tensor(buf_logp.reshape(-1))
        b_adv = torch.as_tensor(adv.reshape(-1))
        b_ret = torch.as_tensor(ret.reshape(-1))
        b_val = torch.as_tensor(buf_val[:T].reshape(-1))
        total = len(b_obs)
        clipfracs, kl_sum, pl_sum, vl_sum, ent_sum, nb = [], 0., 0., 0., 0., 0
        for _ in range(args.epochs):
            perm = torch.randperm(total)
            for s in range(0, total, args.minibatch):
                mb = perm[s:s + args.minibatch]
                logits, value = model(b_obs[mb])
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(b_act[mb])
                ratio = torch.exp(logp - b_logp[mb])
                a = b_adv[mb]
                a = (a - a.mean()) / (a.std() + 1e-8)
                pl = -torch.min(ratio * a, torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * a).mean()
                v_clipped = b_val[mb] + torch.clamp(value - b_val[mb], -args.clip * 10, args.clip * 10)
                vl = .5 * torch.max((value - b_ret[mb]) ** 2, (v_clipped - b_ret[mb]) ** 2).mean()
                ent = dist.entropy().mean()
                loss = pl + .5 * vl - ent_coef * ent
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), .5)
                opt.step()
                with torch.no_grad():
                    kl_sum += float(((ratio - 1) - torch.log(ratio)).mean())
                    clipfracs.append(float(((ratio - 1).abs() > args.clip).float().mean()))
                pl_sum += float(pl); vl_sum += float(vl); ent_sum += float(ent); nb += 1
        recent = ep_returns[-500:]
        st = env.stats
        row = {"iter": it, "samples": (it + 1) * T * N, "wall_s": round(time.time() - t0, 1),
               "ep_return": float(np.mean(recent)) if recent else None, "episodes": len(ep_returns),
               "policy_loss": pl_sum / nb, "value_loss": vl_sum / nb, "entropy": ent_sum / nb, "kl": kl_sum / nb,
               "clipfrac": float(np.mean(clipfracs)), "lr": opt.param_groups[0]["lr"],
               "mean_frac": float(ACTIONS[buf_act].mean())}
        if (it + 1) % args.eval_every == 0 or it == args.iterations - 1:
            model.eval()

            def greedy(o):
                with torch.no_grad():
                    lg, _ = model(torch.as_tensor(norm(o), dtype=torch.float32))
                return lg.argmax(-1).numpy()
            ev = evaluate(routes, greedy, "rl", episodes, 18., limit_hold=args.limit_hold, entry_ease=args.entry_ease)
            model.train()
            safe = (ev["collisions"] <= rule["collisions"] and ev["near_miss_s"] <= rule["near_miss_s"] + 1e-9
                    and ev["person_near_miss_s"] <= rule["person_near_miss_s"] + 1e-9 and ev["violations"] == 0
                    and ev["speeding_s"] <= .05 and ev["over_limit_s"] <= .01 and ev["timeouts"] <= rule["timeouts"])
            row["eval"] = ev
            row["eval_safe"] = safe
            torch.save({"model": model.state_dict(), "norm": (norm.mean, norm.var, norm.count), "iter": it}, out / "last.pt")
            if safe and (best is None or ev["time_s"] < best["time_s"] - 1e-6):
                best = dict(ev, iter=it)
                torch.save({"model": model.state_dict(), "norm": (norm.mean, norm.var, norm.count), "iter": it}, out / "best.pt")
                meta = {"iteration": it, "samples": row["samples"], "eval": ev, "baselines": baseline, "config": config}
                export(model, norm, out / "policy.npz", meta)
                (out / "best.json").write_text(json.dumps(best, indent=1))
            print(f"[{it}] {row['samples']/1e6:.1f}M  ret {row['ep_return'] or 0.:.2f}  frac {row['mean_frac']:.3f}  ent {row['entropy']:.3f} | "
                  f"eval time {ev['time_s']:.1f} (rule {rule['time_s']:.1f} legal {baseline['legal']['time_s']:.1f} "
                  f"shield {baseline['shield']['time_s']:.1f}) coll {ev['collisions']} nearV {ev['near_miss_s']:.1f} "
                  f"nearP {ev['person_near_miss_s']:.1f} hard {ev['hard_brake_s']:.2f} emerg {ev['emergency_s']:.2f} "
                  f"speed {ev['speeding_s']:.2f} safe {safe}  {row['wall_s']:.0f}s", flush=True)
        else:
            print(f"[{it}] {row['samples']/1e6:.1f}M  ret {row['ep_return'] if row['ep_return'] is None else round(row['ep_return'], 2)}  "
                  f"frac {row['mean_frac']:.3f}  ent {row['entropy']:.3f} kl {row['kl']:.4f}  {row['wall_s']:.0f}s", flush=True)
        log.write(json.dumps(row) + "\n"); log.flush()
    (out / "done.json").write_text(json.dumps({"best": best, "baselines": baseline}, indent=1))


if __name__ == "__main__":
    main()
