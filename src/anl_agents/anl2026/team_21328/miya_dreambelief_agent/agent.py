from __future__ import annotations

import contextlib
import base64
import json
import importlib
import io
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
from negmas.outcomes import Outcome
from negmas.preferences import LambdaMultiFun
from negmas.sao import ResponseType, SAOResponse, SAOState

from .safe import MiyaDreamBeliefSafeNegotiator as FallbackNegotiator


MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
FIXED_MAX_ISSUES = 6
FIXED_MAX_VALUES = 5
FIXED_N_PROPOSALS = FIXED_MAX_VALUES ** FIXED_MAX_ISSUES
TARGET_ACTION_KEYS = (
    "target_z_e",
    "target_z_delta_down",
    "target_z_delta_up",
    "target_z_umax",
    "target_rho",
    "target_delta_down",
    "target_delta_up",
    "target_u_max",
)
POLICY_WEIGHT_REGEX = (
    r"^(enc|dyn|pol|fusion|osi|task_belief|action_enc)/"
)
_FLOAT_RE = re.compile(
    r"^[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?$")


class MiyaDreamBeliefNegotiator(FallbackNegotiator):
    """Dreamer-backed ANL negotiator with safe heuristic fallback.

    Expected submission layout:

    model/
      config.yaml
      ckpt_best/
        <checkpoint-folder>/
          agent.pkl
          step.pkl
          done

    The Dreamer stack is loaded lazily.  If any dependency, checkpoint, or
    domain-size requirement is missing, this class behaves exactly like the
    fallback negotiator from ``mynegotiator.py``.
    """

    _FALLBACK_ON_DREAMER_ERROR = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._dreamer = None
        self._dreamer_error: str | None = None
        self._dreamer_carry = None
        self._last_obs = None
        self._last_action = np.int32(0)
        self._fixed_proposals_cache = self._build_fixed_proposals()
        self._proposal_mask_cache_key = None
        self._proposal_mask_cache = None

    def on_preferences_changed(self, changes):
        super().on_preferences_changed(changes)
        self._proposal_mask_cache_key = None
        self._proposal_mask_cache = None
        self._reset_dreamer_state()

    def on_negotiation_start(self, state: SAOState) -> None:
        super().on_negotiation_start(state)
        self._proposal_mask_cache_key = None
        self._proposal_mask_cache = None
        self._reset_dreamer_state()

    def __call__(self, state: SAOState, dest: str | None = None) -> SAOResponse:
        dreamer = self._dreamer_or_none()
        if dreamer is None:
            return self._fallback_or_end(state, dest)
        try:
            if state.current_offer is not None:
                self.update_opponent_model(state)
            obs = self._make_obs(state)
            with self._silence_output():
                carry, act, out = dreamer["agent"].policy(
                    self._dreamer_carry, self._batch_obs(obs), mode="eval_osi")
            self._dreamer_carry = carry
            self._update_private_opponent_ufun_from_osi(out)
            action_id = int(np.asarray(act["action"])[0])
            self._last_action = np.int32(action_id)
            self._last_obs = obs
            if action_id >= FIXED_N_PROPOSALS and state.current_offer is not None:
                return SAOResponse(ResponseType.ACCEPT_OFFER, state.current_offer)
            offer = self._action_to_offer(action_id)
            if offer is None:
                return self._fallback_or_end(state, dest)
            self._last_self_offer = offer
            self._record_self_offer(offer)
            return SAOResponse(ResponseType.REJECT_OFFER, offer)
        except Exception as exc:
            self._dreamer_error = f"{type(exc).__name__}: {exc}"
            return self._fallback_or_end(state, dest)

    def _fallback_or_end(self, state: SAOState, dest: str | None = None) -> SAOResponse:
        if self._FALLBACK_ON_DREAMER_ERROR:
            return super().__call__(state, dest)
        return SAOResponse(ResponseType.END_NEGOTIATION, None)

    def _update_private_opponent_ufun_from_osi(self, out) -> None:
        private_utils = out.get("private_opp_utils") if isinstance(out, dict) else None
        if private_utils is not None:
            utils = np.asarray(private_utils, dtype=np.float32)
            if utils.ndim >= 2:
                utils = utils.reshape((-1, utils.shape[-1]))[0]
            if utils.shape[-1] >= FIXED_N_PROPOSALS:
                self._score_private_opp_utils = [
                    float(np.clip(x, 0.0, 1.0))
                    for x in utils[:FIXED_N_PROPOSALS]]

        belief = out.get("opp_belief") if isinstance(out, dict) else None
        if belief is None:
            return
        flat = np.asarray(belief)
        if flat.ndim == 0:
            return
        flat = flat.reshape((-1, flat.shape[-1]))[0]
        expected = 2 * FIXED_MAX_ISSUES + 2 * FIXED_MAX_ISSUES * FIXED_MAX_VALUES + 2
        if flat.shape[-1] < expected:
            return

        pos = 0
        weights_mean = flat[pos:pos + FIXED_MAX_ISSUES].astype(np.float32)
        pos += FIXED_MAX_ISSUES
        weights_std = flat[pos:pos + FIXED_MAX_ISSUES].astype(np.float32)
        pos += FIXED_MAX_ISSUES
        size = FIXED_MAX_ISSUES * FIXED_MAX_VALUES
        values_mean = flat[pos:pos + size].reshape(
            FIXED_MAX_ISSUES, FIXED_MAX_VALUES).astype(np.float32)
        pos += size
        values_std = flat[pos:pos + size].reshape(
            FIXED_MAX_ISSUES, FIXED_MAX_VALUES).astype(np.float32)

        issue_values = self._issue_values
        if not issue_values:
            self._initialize()
            issue_values = self._issue_values
        if not issue_values:
            return

        active = np.zeros((FIXED_MAX_ISSUES,), dtype=np.float32)
        active[:min(len(issue_values), FIXED_MAX_ISSUES)] = 1.0
        weights_mean = np.where(active > 0, weights_mean, 0.0)
        weights_std = np.where(active > 0, weights_std, 1.0)
        n = min(len(issue_values), FIXED_MAX_ISSUES)
        self._score_osi_weights_mean = [
            float(x) for x in weights_mean[:n]]
        self._score_osi_weights_std = [
            float(x) for x in weights_std[:n]]
        self._score_osi_values_mean = [
            [float(x) for x in values_mean[i, :min(len(issue_values[i]), FIXED_MAX_VALUES)]]
            for i in range(n)
        ]
        self._score_osi_values_std = [
            [float(x) for x in values_std[i, :min(len(issue_values[i]), FIXED_MAX_VALUES)]]
            for i in range(n)
        ]

    @property
    def dreamer_available(self) -> bool:
        return self._dreamer_or_none() is not None

    @property
    def dreamer_error(self) -> str | None:
        self._dreamer_or_none()
        return self._dreamer_error

    def _reset_dreamer_state(self) -> None:
        self._dreamer_carry = None
        self._last_obs = None
        self._last_action = np.int32(0)
        dreamer = self._dreamer_or_none()
        if dreamer is not None:
            self._dreamer_carry = dreamer["agent"].init_policy(1)

    def _dreamer_or_none(self):
        if self._dreamer is not None:
            return self._dreamer
        if self._dreamer_error:
            return None
        try:
            self._dreamer = self._load_dreamer()
            return self._dreamer
        except Exception as exc:
            self._dreamer_error = f"{type(exc).__name__}: {exc}"
            return None

    def _load_dreamer(self):
        root = self._find_dreamer_root()
        if root is not None and str(root) not in sys.path:
            sys.path.insert(0, str(root))
            sys.path.insert(1, str(root.parent))

        import elements
        import yaml

        config_path = MODEL_DIR / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(config_path)
        config_data = self._coerce_config_scalars(yaml.safe_load(
            config_path.read_text()))
        config = elements.Config(config_data)

        with self._silence_output():
            obs_space = self._obs_space(elements)
            act_space = {
                "action": elements.Space(np.int32, (), 0, FIXED_N_PROPOSALS + 1),
                "dreamer_opp_weights": elements.Space(
                    np.float32, (FIXED_MAX_ISSUES,), 0.0, 1.0),
                "dreamer_opp_values": elements.Space(
                    np.float32, (FIXED_MAX_ISSUES, FIXED_MAX_VALUES), 0.0, 1.0),
            }
            act_space.update({
                key: elements.Space(np.float32, (), -np.inf, np.inf)
                for key in TARGET_ACTION_KEYS
            })
            module = importlib.import_module(f"dreamerv3.{config.agent_module}")
            agent = module.Agent(obs_space, act_space, elements.Config(
                **config.agent,
                logdir=str(MODEL_DIR),
                seed=int(getattr(config, "seed", 0)),
                jax=config.jax,
                batch_size=config.batch_size,
                batch_length=config.batch_length,
                replay_context=config.replay_context,
                report_length=config.report_length,
                replica=getattr(config, "replica", 0),
                replicas=getattr(config, "replicas", 1),
            ))

            ckpt = self._load_agent_weights(agent, elements)
        return {"agent": agent, "config": config, "checkpoint": ckpt}

    @staticmethod
    @contextlib.contextmanager
    def _silence_output():
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            yield

    def _coerce_config_scalars(self, value):
        if isinstance(value, dict):
            return {k: self._coerce_config_scalars(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._coerce_config_scalars(v) for v in value]
        if isinstance(value, str):
            text = value.strip()
            lower = text.lower()
            if lower in ("true", "false"):
                return lower == "true"
            if lower == "null":
                return None
            if _FLOAT_RE.match(text):
                number = float(text)
                if (
                    "e" not in lower
                    and "." not in text
                    and number.is_integer()
                ):
                    return int(number)
                return number
        return value

    def _find_dreamer_root(self) -> Path | None:
        candidates = [
            Path(__file__).resolve().parent / "dreamerv3",
            Path(__file__).resolve().parent / "dreamerv3-negotiation",
            MODEL_DIR / "dreamerv3",
        ]
        for path in candidates:
            if (path / "dreamerv3").exists() and (path / "embodied").exists():
                return path
        return None

    def _checkpoint_path(self) -> Path:
        direct = MODEL_DIR / "checkpoint"
        if (direct / "agent.pkl").exists():
            return direct
        ckpt_best = MODEL_DIR / "ckpt_best"
        if ckpt_best.exists():
            children = sorted(p for p in ckpt_best.iterdir() if p.is_dir())
            children = [p for p in children if (p / "agent.pkl").exists()]
            if children:
                return children[-1]
        raise FileNotFoundError(f"No Dreamer checkpoint found under {MODEL_DIR}")

    def _load_agent_weights(self, agent, elements):
        embedded = self._embedded_agent_weights()
        if embedded is not None:
            params, counters = embedded
            params = self._cast_params_for_agent(agent, params)
            agent.load(
                {"params": params, "counters": counters},
                regex=POLICY_WEIGHT_REGEX)
            return "embedded:weights.py"

        npz_path = MODEL_DIR / "agent_params.npz"
        meta_path = MODEL_DIR / "agent_params.json"
        if npz_path.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text())
            keys = meta["keys"]
            arrays = np.load(npz_path, allow_pickle=False)
            params = {key: arrays[f"arr_{i}"] for i, key in enumerate(keys)}
            params = self._cast_params_for_agent(agent, params)
            counters = meta.get(
                "counters", {"updates": 0, "batches": 0, "actions": 0})
            agent.load(
                {"params": params, "counters": counters},
                regex=POLICY_WEIGHT_REGEX)
            return npz_path

        ckpt = self._checkpoint_path()
        cp = elements.Checkpoint()
        cp.agent = agent
        cp.load(ckpt, keys=["agent"])
        return ckpt

    @staticmethod
    def _cast_params_for_agent(agent, params):
        casted = {}
        for key, value in params.items():
            target = agent.params.get(key)
            array = np.asarray(value)
            if target is not None:
                dtype = getattr(target, "dtype", None)
                if dtype is not None and array.dtype != dtype:
                    array = array.astype(dtype)
            casted[key] = array
        return casted

    def _embedded_agent_weights(self):
        split = self._embedded_split_agent_weights()
        if split is not None:
            return split
        try:
            from . import weights
        except Exception:
            return None
        payload = "".join(weights.PARAMS_B64)
        raw = base64.b64decode(payload.encode("ascii"))
        arrays = np.load(io.BytesIO(raw), allow_pickle=False)
        params = {
            key: arrays[f"arr_{i}"]
            for i, key in enumerate(weights.KEYS)
        }
        counters = getattr(
            weights, "COUNTERS", {"updates": 0, "batches": 0, "actions": 0})
        return params, counters

    def _embedded_split_agent_weights(self):
        try:
            from . import weights_index
        except Exception:
            return None
        chunks = []
        for name in weights_index.PARTS:
            module = importlib.import_module(f".{name}", __package__)
            chunks.append(module.DATA)
        raw = base64.b64decode("".join(chunks).encode("ascii"))
        arrays = np.load(io.BytesIO(raw), allow_pickle=False)
        params = {
            key: arrays[f"arr_{i}"]
            for i, key in enumerate(weights_index.KEYS)
        }
        counters = getattr(
            weights_index, "COUNTERS",
            {"updates": 0, "batches": 0, "actions": 0})
        return params, counters

    def _obs_space(self, elements):
        return {
            "relative_time": elements.Space(np.float32, (), 0.0, 1.0),
            "last_opp_offer": elements.Space(
                np.int32, (FIXED_MAX_ISSUES,), 0, FIXED_MAX_VALUES),
            "last_self_offer": elements.Space(
                np.int32, (FIXED_MAX_ISSUES,), 0, FIXED_MAX_VALUES),
            "last_opp_valid": elements.Space(bool),
            "last_self_valid": elements.Space(bool),
            "issue_mask": elements.Space(bool, (FIXED_MAX_ISSUES,)),
            "value_mask": elements.Space(
                bool, (FIXED_MAX_ISSUES, FIXED_MAX_VALUES)),
            "opp_offer_counts": elements.Space(
                np.float32, (FIXED_MAX_ISSUES, FIXED_MAX_VALUES), 0.0),
            "self_offer_counts": elements.Space(
                np.float32, (FIXED_MAX_ISSUES, FIXED_MAX_VALUES), 0.0),
            "self_weights": elements.Space(
                np.float32, (FIXED_MAX_ISSUES,), 0.0, 1.0),
            "self_values": elements.Space(
                np.float32, (FIXED_MAX_ISSUES, FIXED_MAX_VALUES), 0.0, 1.0),
            "self_reserved": elements.Space(np.float32, (), 0.0, 1.0),
            "true_opp_weights": elements.Space(
                np.float32, (FIXED_MAX_ISSUES,), 0.0, 1.0),
            "true_opp_values": elements.Space(
                np.float32, (FIXED_MAX_ISSUES, FIXED_MAX_VALUES), 0.0, 1.0),
            "true_opp_reserved": elements.Space(np.float32, (), 0.0, 1.0),
            "opponent_type_id": elements.Space(np.int32, (), 0, 4),
            "proposal_mask": elements.Space(bool, (FIXED_N_PROPOSALS,)),
            "reward": elements.Space(np.float32),
            "is_first": elements.Space(bool),
            "is_last": elements.Space(bool),
            "is_terminal": elements.Space(bool),
        }

    def _make_obs(self, state: SAOState) -> dict[str, np.ndarray]:
        issue_values = self._issue_values
        if not issue_values:
            self._initialize()
            issue_values = self._issue_values
        if len(issue_values) > FIXED_MAX_ISSUES:
            raise ValueError("Dreamer fixed policy supports at most 6 issues")
        if any(len(values) > FIXED_MAX_VALUES for values in issue_values):
            raise ValueError("Dreamer fixed policy supports at most 5 values per issue")

        issue_mask = np.zeros((FIXED_MAX_ISSUES,), bool)
        value_mask = np.zeros((FIXED_MAX_ISSUES, FIXED_MAX_VALUES), bool)
        for i, values in enumerate(issue_values):
            issue_mask[i] = True
            value_mask[i, : len(values)] = True

        weights, values = self._approx_self_table(issue_values)
        last_opp = self._offer_to_indices(state.current_offer)
        last_self = self._offer_to_indices(self._last_self_offer)
        opp_counts = self._counts_matrix(self._opp_counts, issue_values)
        self_counts = self._counts_matrix(self._self_counts, issue_values)
        return {
            "relative_time": np.float32(state.relative_time),
            "last_opp_offer": last_opp,
            "last_self_offer": last_self,
            "last_opp_valid": np.bool_(state.current_offer is not None),
            "last_self_valid": np.bool_(self._last_self_offer is not None),
            "issue_mask": issue_mask,
            "value_mask": value_mask,
            "opp_offer_counts": opp_counts,
            "self_offer_counts": self_counts,
            "self_weights": weights,
            "self_values": values,
            "self_reserved": np.float32(getattr(self.ufun, "reserved_value", 0.0) or 0.0),
            "true_opp_weights": np.zeros((FIXED_MAX_ISSUES,), np.float32),
            "true_opp_values": np.zeros(
                (FIXED_MAX_ISSUES, FIXED_MAX_VALUES), np.float32),
            "true_opp_reserved": np.float32(0.0),
            "opponent_type_id": np.int32(0),
            "proposal_mask": self._proposal_mask(issue_values),
            "reward": np.float32(0.0),
            "is_first": np.bool_(self._last_obs is None),
            "is_last": np.bool_(False),
            "is_terminal": np.bool_(False),
        }

    def _counts_matrix(self, counters, issue_values):
        counts = np.zeros((FIXED_MAX_ISSUES, FIXED_MAX_VALUES), np.float32)
        for i, values in enumerate(issue_values[:FIXED_MAX_ISSUES]):
            width = min(len(values), FIXED_MAX_VALUES)
            counts[i, :width] = 1.0
            if i >= len(counters):
                continue
            for j, value in enumerate(values[:width]):
                counts[i, j] += float(counters[i][value])
        return counts

    def _approx_self_table(self, issue_values):
        weights = np.zeros((FIXED_MAX_ISSUES,), np.float32)
        values = np.zeros((FIXED_MAX_ISSUES, FIXED_MAX_VALUES), np.float32)
        if not issue_values:
            return weights, values
        weights[: len(issue_values)] = 1.0 / len(issue_values)
        baseline = [vals[0] for vals in issue_values]
        for i, vals in enumerate(issue_values):
            raw = []
            for value in vals:
                outcome = list(baseline)
                outcome[i] = value
                raw.append(float(self.ufun(tuple(outcome))))
            lo, hi = min(raw), max(raw)
            scale = hi - lo
            if scale <= 1e-9:
                values[i, : len(vals)] = 0.5
            else:
                values[i, : len(vals)] = [(x - lo) / scale for x in raw]
        return weights, values

    def _proposal_mask(self, issue_values):
        key = tuple(len(values) for values in issue_values)
        if key == self._proposal_mask_cache_key and self._proposal_mask_cache is not None:
            return self._proposal_mask_cache
        proposals = self._fixed_proposals()
        valid = np.ones((FIXED_N_PROPOSALS,), bool)
        for i in range(FIXED_MAX_ISSUES):
            if i < len(issue_values):
                valid &= proposals[:, i] < len(issue_values[i])
            else:
                valid &= proposals[:, i] == 0
        self._proposal_mask_cache_key = key
        self._proposal_mask_cache = valid
        return valid

    def _action_to_offer(self, action_id: int) -> Outcome | None:
        if action_id >= FIXED_N_PROPOSALS:
            return None
        indices = self._fixed_proposals()[action_id]
        return self._indices_to_offer(indices)

    def _offer_to_indices(self, offer: Outcome | None) -> np.ndarray:
        indices = np.zeros((FIXED_MAX_ISSUES,), np.int32)
        if offer is None:
            return indices
        values = self._as_tuple(offer)
        for i, value in enumerate(values[:FIXED_MAX_ISSUES]):
            try:
                indices[i] = self._issue_values[i].index(value)
            except (ValueError, IndexError):
                indices[i] = 0
        return indices

    def _indices_to_offer(self, indices: np.ndarray) -> Outcome:
        offer = []
        for i, values in enumerate(self._issue_values[:FIXED_MAX_ISSUES]):
            idx = int(indices[i])
            idx = min(max(idx, 0), len(values) - 1)
            offer.append(values[idx])
        return tuple(offer)

    def _batch_obs(self, obs):
        return {key: np.expand_dims(value, 0) for key, value in obs.items()}

    def _fixed_proposals(self):
        if self._fixed_proposals_cache is None:
            self._fixed_proposals_cache = self._build_fixed_proposals()
        return self._fixed_proposals_cache

    @staticmethod
    def _build_fixed_proposals():
        return np.array(
            np.meshgrid(
                *[np.arange(FIXED_MAX_VALUES, dtype=np.int32)
                  for _ in range(FIXED_MAX_ISSUES)],
                indexing="ij",
            )
        ).reshape(FIXED_MAX_ISSUES, -1).T
