"""
v3 experiment: protocol-as-mechanism in LLM bargaining, with protocol-neutral
outcome measurement (post-hoc elicitation) and repeated episodes.

Design notes
------------
* Outcome measurement no longer depends on per-protocol regex parsing of the
  conversation. After every episode each agent is asked, with an identical
  prompt, whether an agreement was reached and at what split. An agreement
  counts only if BOTH agents say YES and report the same A-share. The legacy
  parser is still applied to free-text transcripts, so its bias can be measured.
* Each "trial" is a pair of agents playing K repeated episodes. Only the
  outcome (not the transcript) is carried across episodes, which isolates
  learning-from-outcome from context-length effects.
* All token counts come from the API `usage` field, not from character counts.

Usage
-----
  python v3/run_v3.py --dry-run
  python v3/run_v3.py --models flash --trials 30 --episodes 5 --workers 10
  python v3/run_v3.py --models flash pro --protocols sealed_bid alternating_offers --trials 20
"""

import argparse
import csv
import json
import os
import random
import re
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent))
from legacy_parser import legacy_free_text_agreement  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BASE_URL = "https://api.deepseek.com"

MODELS = {
    # key: provider-aware spec
    # NOTE (2026-10-23): DeepSeek endpoints default to a reasoning mode that can consume
    # the whole max_tokens budget on hidden reasoning and return empty content, so it is
    # pinned to reasoning_effort="none". Qwen3 / GLM-4.5 default to a thinking mode that
    # is disabled through the chat-template flag enable_thinking=False.
    "flash": dict(id="deepseek-flash", label="DeepSeek-V4.1-Flash", max_tokens=700,
                  base_url="https://api.deepseek.com", key_env="DEEPSEEK_API_KEY",
                  extra_body={"reasoning_effort": "none"}),
    "pro": dict(id="deepseek-v4-pro", label="DeepSeek-V4-Pro", max_tokens=700,
                base_url="https://api.deepseek.com", key_env="DEEPSEEK_API_KEY",
                extra_body={"reasoning_effort": "none"}),
    "qwen32b": dict(id="Qwen/Qwen3-32B", label="Qwen3-32B", max_tokens=700,
                    base_url="https://api.siliconflow.cn/v1", key_env="SILICONFLOW_API_KEY",
                    extra_body={"enable_thinking": False}),
    "glm45air": dict(id="zai-org/GLM-4.5-Air", label="GLM-4.5-Air", max_tokens=700,
                     base_url="https://api.siliconflow.cn/v1", key_env="SILICONFLOW_API_KEY",
                     extra_body={"enable_thinking": False}),
}

PROTOCOLS = ["free_text", "alternating_offers", "sealed_bid", "mediated",
             "sealed_compromise", "sealed_compromise_verify"]


# ────────────────────────────────────────────────────────────── credentials
def load_api_key(name="DEEPSEEK_API_KEY"):
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.strip().startswith(name):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    key = os.environ.get(name)
    if not key:
        raise SystemExit(f"{name} not found")
    return key


# ────────────────────────────────────────────────────────────── budget
class Budget:
    def __init__(self, max_tokens):
        self.max_tokens = max_tokens
        self.tokens = 0
        self.lock = threading.Lock()

    def add(self, n):
        with self.lock:
            self.tokens += n
            if self.tokens > self.max_tokens:
                raise BudgetExceeded(f"token budget exceeded: {self.tokens} > {self.max_tokens}")

    def spent(self):
        with self.lock:
            return self.tokens


class BudgetExceeded(Exception):
    pass


class CallFailure(Exception):
    pass


# ────────────────────────────────────────────────────────────── agent
class Agent:
    def __init__(self, role, system_prompt, model_key, temperature, budget, max_tokens):
        self.role = role
        self.system_prompt = system_prompt
        spec = MODELS[model_key]
        self.model_id = spec["id"]
        self.label = spec["label"]
        self.model_key = model_key
        self.temperature = temperature
        self.budget = budget
        self.max_tokens = max_tokens or spec["max_tokens"]
        self.extra_body = spec.get("extra_body") or {}
        self.client = OpenAI(api_key=load_api_key(spec["key_env"]),
                             base_url=spec["base_url"], timeout=150.0)
        self.messages = [{"role": "system", "content": system_prompt}]
        self.usage_in = 0
        self.usage_out = 0
        self.calls = 0

    def reset(self):
        self.messages = [{"role": "system", "content": self.system_prompt}]

    def call(self, user_message, retries=4):
        self.messages.append({"role": "user", "content": user_message})
        last_err = None
        for attempt in range(retries):
            kwargs = dict(model=self.model_id, messages=self.messages,
                          max_tokens=self.max_tokens, temperature=self.temperature,
                          extra_body=self.extra_body)
            try:
                t0 = time.time()
                resp = self.client.chat.completions.create(**kwargs)
                text = (resp.choices[0].message.content or "").strip()
                u = resp.usage
                if u is not None:
                    self.usage_in += getattr(u, "prompt_tokens", 0) or 0
                    self.usage_out += getattr(u, "completion_tokens", 0) or 0
                    self.budget.add((getattr(u, "prompt_tokens", 0) or 0) +
                                    (getattr(u, "completion_tokens", 0) or 0))
                self.calls += 1
                if not text:
                    # reasoning model may have burned the budget on thinking
                    raise CallFailure("empty content")
                self.messages.append({"role": "assistant", "content": text})
                _ = time.time() - t0
                return text
            except BudgetExceeded:
                raise
            except Exception as e:  # noqa: BLE001
                last_err = e
                self.messages.pop()  # drop the user turn, retry cleanly
                if attempt < retries - 1:
                    is_rl = "429" in str(e) or "rate limit" in str(e).lower() or "TPM" in str(e)
                    wait = (25 + 10 * attempt) if is_rl else min(30, 2 ** attempt + random.random())
                    time.sleep(wait)
                    self.messages.append({"role": "user", "content": user_message})
                else:
                    break
        raise CallFailure(f"{self.role}/{self.model_id}: {type(last_err).__name__}: {last_err}")


# ────────────────────────────────────────────────────────────── prompts
def game_rules(R, T):
    return (
        f"The game has {R} points in round 1. Each round the pie shrinks: the pie in round r is\n"
        f"{R} * 0.9^(r-1), rounded to the nearest integer. Points that are not allocated in the\n"
        f"round in which they are available are lost forever.\n"
        f"You are one of two agents, named {{role}}. Your payoff is the number of points you receive.\n"
        f"Rules:\n"
        f"1. If you reach an agreement in round r, the pie of round r is split as agreed.\n"
        f"2. If you do not reach agreement within {T} rounds, both agents receive 0 points.\n"
        f"3. Your goal is to maximise your own payoff. You do not care about the other agent's payoff.\n"
    )


def pie(R, r, delta=0.9):
    """Pie available in round r (integer, at least 1)."""
    return max(1, int(round(R * (delta ** (r - 1)))))


PROTOCOL_RULES = {
    "free_text": (
        "Communication protocol: free-form dialogue.\n"
        "- Each round you and the other agent exchange one message.\n"
        "- You may write anything: propose splits, argue, accept, or reject.\n"
        "- There is no required format.\n"
    ),
    "free_text_exec": (
        "Communication protocol: free-form dialogue with an execution hook.\n"
        "- Each round you and the other agent exchange one message, in free text.\n"
        "- To close the negotiation, use a formal line. To propose: PROPOSE: A=<int>, B=<int>\n"
        "  To accept the other agent's most recent proposal: ACCEPT: A=<int>, B=<int>\n"
        "- An ACCEPT is binding: if it matches the other agent's most recent proposal,\n"
        "  the deal is executed immediately at that split and the episode ends.\n"
        "- Plain phrases such as 'I accept' are not binding; only the formal line is.\n"
    ),
    "free_text_loose": (
        "Communication protocol: free-form dialogue with a forgiving execution hook.\n"
        "- Each round you and the other agent exchange one message, in free text.\n"
        "- If you accept a split, say so explicitly and state it (e.g. 'I accept 5-5').\n"
        "- The system executes a deal as soon as one agent clearly accepts a split that the\n"
        "  other agent has just proposed. No special syntax is required.\n"
    ),
    "alternating_offers": (
        "Communication protocol: alternating offers.\n"
        "- Odd rounds: Agent A proposes a split. Even rounds: Agent B proposes.\n"
        "- A proposal must be written exactly as: SPLIT: me=<int>, other=<int>\n"
        "  where 'me' is what you ask for yourself and 'other' is what the other agent gets.\n"
        "- The other agent replies either ACCEPT or REJECT.\n"
        "- If the reply is REJECT, the next round the other agent proposes a new split.\n"
    ),
    "sealed_bid": (
        "Communication protocol: simultaneous sealed demands.\n"
        "- Each round both agents privately state how many points they demand for themselves.\n"
        "- You never see the other agent's demand.\n"
        "- If your demand plus the other agent's demand is at most the total, a deal is executed\n"
        "  and each receives exactly what they demanded. Any unclaimed points are discarded.\n"
        "- Otherwise no deal is executed this round and the next round begins.\n"
        "- A demand must be written exactly as: DEMAND: <int>\n"
    ),
    "mediated": (
        "Communication protocol: mediation.\n"
        "- A neutral mediator helps the two agents reach a deal.\n"
        "- Before each proposal you may send the mediator a private message saying what you want and why.\n"
        "- The mediator sees both messages and proposes a split.\n"
        "- You then reply either ACCEPT or REJECT to the mediator's proposal.\n"
        "- If both accept, the deal is executed at the mediator's proposal. Otherwise the mediator may try again.\n"
    ),
    "sealed_compromise": (
        "Communication protocol: simultaneous sealed demands with a compromise fallback.\n"
        "- Each round both agents privately state how many points they demand for themselves.\n"
        "- You never see the other agent's demand.\n"
        "- If your demand plus the other agent's demand is at most the total, a deal is executed\n"
        "  and each receives exactly what they demanded. Any unclaimed points are discarded.\n"
        "- If the demands are incompatible, the mechanism does NOT discard the points. Instead it\n"
        "  proposes a compromise split that is proportional to the two demands, and each of you\n"
        "  replies ACCEPT or REJECT.\n"
        "- If both accept the compromise, the deal is executed. Otherwise the next round begins.\n"
        "- A demand must be written exactly as: DEMAND: <int>\n"
    ),
    "sealed_compromise_verify": (
        "Communication protocol: simultaneous sealed demands with a compromise fallback and a\n"
        "commitment check.\n"
        "- Each round both agents privately state how many points they demand for themselves.\n"
        "- If your demand plus the other agent's demand is at most the total, a deal is proposed\n"
        "  at exactly those demands.\n"
        "- If the demands are incompatible, the mechanism proposes a compromise split that is\n"
        "  proportional to the two demands, instead of discarding the points.\n"
        "- Whenever a deal is proposed (either at the demanded split or at the compromise split),\n"
        "  BOTH agents must confirm it. Reply CONFIRM to commit, or REJECT to refuse.\n"
        "- If either agent refuses or fails to confirm, the deal is void and the next round begins.\n"
        "- A demand must be written exactly as: DEMAND: <int>\n"
    ),
}

MEDIATOR_SYSTEM = (
    "You are a neutral mediator. Two agents, Agent A and Agent B, are bargaining over how to divide "
    "{R} points. Listen to both, then propose a split that you believe both will accept.\n"
    "Propose exactly one split per round. Reply exactly as: SPLIT: A=<int>, B=<int>\n"
    "with A+B <= {R}."
)

ELICIT_PROMPT = (
    "The session is over. Answer two questions about what actually happened.\n"
    "1) Did you and the other agent agree on a split of the pie, i.e. did BOTH of you state\n"
    "   acceptance of the same split? A deal counts even if no separate mechanism executed it.\n"
    "2) If YES, what was the agreed split? Give the number of points for Agent A and for Agent B.\n"
    "Reply with exactly these two lines and nothing else:\n"
    "AGREED: YES|NO\n"
    "SPLIT: A=<integer>, B=<integer>\n"
    "(If you did not reach an agreement, write SPLIT: A=0, B=0.)"
)


def build_system(role, protocol, R, T):
    return (f"You are Agent {role}.\n\n" + game_rules(R, T).format(role=role)
            + "\n" + PROTOCOL_RULES[protocol])


# ────────────────────────────────────────────────────────────── parsers
def parse_me_other(text):
    m = re.search(r'me\s*=\s*(\d+)\s*,\s*other\s*=\s*(\d+)', text, re.I)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.search(r'\[?\s*(\d+)\s*,\s*(\d+)\s*\]?', text)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


def parse_demand(text):
    m = re.search(r'DEMAND\s*[:\-]?\s*(\d+)', text, re.I)
    if m:
        return int(m.group(1))
    m = re.search(r'(?<!\d)(\d+)(?!\d)', text)
    if m:
        return int(m.group(1))
    return None


def parse_ab(text):
    m = re.search(r'A\s*=\s*(\d+)\s*,\s*B\s*=\s*(\d+)', text, re.I)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.search(r'(\d+)\s*[,，/]\s*(\d+)', text)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


def parse_accept(text):
    """ACCEPT / REJECT parser.

    Negation-first: "I cannot accept" must not be read as acceptance.
    (Hardened 2026-10-28; the archived structured runs contain no such reply.)
    """
    t = text.strip().lower()
    if re.fullmatch(r"\s*(no|nope|nay)\.?\s*", t):
        return False
    if re.search(r"\b(not|n't|cannot|can't|won't|don't|unable|never)\b[^.]{0,25}"
                 r"\b(accept|agree|deal|yes)\b", t):
        return False
    if re.search(r"\b(reject|rejected|decline|declined|refuse|refused)\b", t):
        return False
    if re.search(r"\b(accept|accepted|agree|agreed|deal|yes)\b", t):
        return True
    if '接受' in t or '同意' in t or '成交' in t:
        return True
    if '拒绝' in t or '不同意' in t:
        return False
    return None


# ────────────────────────────────────────────────────────────── protocols
@dataclass
class Episode:
    rounds: int = 0
    native_split: tuple = None
    legacy_split: tuple = None
    legacy_round: int = 0
    bids: list = field(default_factory=list)      # sealed: (round, demand_a, demand_b)
    proposals: list = field(default_factory=list)
    transcript: list = field(default_factory=list)
    fallback_offered: int = 0
    fallback_accepted: int = 0
    confirm_failed: int = 0

    def log(self, **kw):
        self.transcript.append(kw)


def _ask_parsed(agent, prompt, parser, repair, retries=2):
    """Call, parse, and if parsing fails ask once more with a strict reminder."""
    txt = agent.call(prompt)
    out = parser(txt)
    if out is None:
        for _ in range(retries - 1):
            txt = agent.call("Your previous reply could not be parsed. " + repair)
            out = parser(txt)
            if out is not None:
                break
    return out, txt


def run_free_text(a, b, R, T, ep):
    # Free dialogue: strict turn-taking A -> B -> A -> B, each agent sees the
    # other's previous utterance. (Fixed 2026-10-28: A previously never received
    # B's replies, which made A deaf after round 1.)
    last_b = ""
    for r in range(1, T + 1):
        P = pie(R, r)
        if last_b:
            prompt_a = (f"Round {r}. The pie this round is {P} points. "
                        f"Agent B says: \"{last_b}\"\nReply to Agent B.")
        else:
            prompt_a = f"Round {r}. The pie this round is {P} points. Say anything you want to the other agent."
        ma = a.call(prompt_a)
        mb = b.call(f"Round {r}. The pie this round is {P} points. Agent A says: \"{ma}\"\nReply to Agent A.")
        last_b = mb
        ep.log(round=r, a=ma, b=mb)
        ep.rounds = r
        hit = legacy_free_text_agreement(ma, mb)
        if ep.legacy_split is None and hit is not None:
            ep.legacy_split = hit
            ep.legacy_round = r
    return ep



_PROF = re.compile(r"PROPOSE\s*:\s*A\s*=\s*(\d+)\s*,\s*B\s*=\s*(\d+)", re.I)
_ACCF = re.compile(r"ACCEPT\s*:\s*A\s*=\s*(\d+)\s*,\s*B\s*=\s*(\d+)", re.I)


def run_free_text_exec(a, b, R, T, ep):
    """Free dialogue with an execution hook: a formal ACCEPT of a formal PROPOSE executes."""
    pending = None
    last_b = ""
    for r in range(1, T + 1):
        P = pie(R, r)
        if last_b:
            prompt_a = (f"Round {r}. The pie this round is {P} points. Agent B says: \"{last_b}\"\n"
                        f"Reply to Agent B. To close, use PROPOSE: / ACCEPT: with A= and B=.")
        else:
            prompt_a = (f"Round {r}. The pie this round is {P} points. Say anything you want to the other agent. "
                        f"To close, use PROPOSE: A=<int>, B=<int> or ACCEPT: A=<int>, B=<int>.")
        ma = a.call(prompt_a)
        mb = b.call(f"Round {r}. The pie this round is {P} points. Agent A says: \"{ma}\"\n"
                    f"Reply to Agent A. To close, use PROPOSE: / ACCEPT: with A= and B=.")
        last_b = mb
        pa, aa = _PROF.search(ma), _ACCF.search(ma)
        pb, ab = _PROF.search(mb), _ACCF.search(mb)
        deal = None
        if aa:
            split = (int(aa.group(1)), int(aa.group(2)))
            if pending == split:
                deal = split
        if deal is None and ab:
            split = (int(ab.group(1)), int(ab.group(2)))
            if pending == split:
                deal = split
        if deal is None:
            if pa:
                pending = (int(pa.group(1)), int(pa.group(2)))
            if pb:
                pending = (int(pb.group(1)), int(pb.group(2)))
        ep.log(round=r, a=ma, b=mb, pending=pending, deal=deal)
        ep.rounds = r
        if deal and sum(deal) <= P:
            ep.native_split = deal
            a.call(f"Deal executed at your formal ACCEPT: you get {deal[0]}, the other agent gets {deal[1]}.")
            b.call(f"Deal executed at the formal ACCEPT: you get {deal[1]}, the other agent gets {deal[0]}.")
            break
    return ep


def run_free_text_loose(a, b, R, T, ep):
    """Free dialogue with a forgiving execution hook.

    A deal executes when one agent states acceptance and names a split that the other
    agent proposed in the previous turn (or when both accept the same split in a round).
    No special syntax is required: ordinary 'I accept 5-5' closes the negotiation.
    """
    pending = None
    last_b = ""
    for r in range(1, T + 1):
        P = pie(R, r)
        if last_b:
            prompt_a = (f"Round {r}. The pie this round is {P} points. Agent B says: \"{last_b}\"\n"
                        f"Reply to Agent B. If you accept a split, say so explicitly and state it.")
        else:
            prompt_a = (f"Round {r}. The pie this round is {P} points. Say anything you want to the other agent. "
                        f"If you accept a split, say so explicitly and state it.")
        ma = a.call(prompt_a)
        mb = b.call(f"Round {r}. The pie this round is {P} points. Agent A says: \"{ma}\"\n"
                    f"Reply to Agent A. If you accept a split, say so explicitly and state it.")
        last_b = mb

        deal = None
        # 1) one side accepts the pending proposal
        for msg in (ma, mb):
            sp = _splits_in(msg)
            if _has_accept(msg) and pending is not None and pending in sp:
                deal = pending
                break
        # 2) both sides accept the same split within the round
        if deal is None:
            sa, sb = _splits_in(ma), _splits_in(mb)
            if _has_accept(ma) and _has_accept(mb):
                common = [x for x in sa if x in sb]
                if common:
                    deal = common[0]

        # update the pending proposal from this round's text
        for msg in (ma, mb):
            sp = [x for x in _splits_in(msg) if 0 < sum(x) <= P]
            if sp:
                pending = sp[-1]

        ep.log(round=r, a=ma, b=mb, pending=pending, deal=deal)
        ep.rounds = r
        if deal is not None and 0 < sum(deal) <= P:
            ep.native_split = deal
            a.call(f"Deal executed: you get {deal[0]}, the other agent gets {deal[1]}.")
            b.call(f"Deal executed: you get {deal[1]}, the other agent gets {deal[0]}.")
            break
    return ep

def run_alternating(a, b, R, T, ep):
    for r in range(1, T + 1):
        P = pie(R, r)
        proposer, responder = (a, b) if r % 2 == 1 else (b, a)
        pname, rname = ("A", "B") if r % 2 == 1 else ("B", "A")
        prop, raw = _ask_parsed(
            proposer,
            f"Round {r}. The pie this round is {P} points. Propose a split. "
            f"Reply exactly: SPLIT: me=<int>, other=<int>",
            parse_me_other,
            "Reply exactly: SPLIT: me=<int>, other=<int>")
        if prop is None:
            ep.log(round=r, status="unparseable_proposal", raw=raw, by=pname)
            continue
        mine, theirs = prop
        if mine + theirs > P:
            mine = min(mine, P)
            theirs = P - mine
        reply = responder.call(
            f"Round {r}. The pie this round is {P} points. Agent {pname} proposes: "
            f"SPLIT: me={mine}, other={theirs} "
            f"(Agent {pname} gets {mine}, you get {theirs}).\nReply ACCEPT or REJECT.")
        acc = parse_accept(reply)
        ep.log(round=r, by=pname, proposal=[mine, theirs], reply=reply, accepted=acc)
        ep.rounds = r
        ep.proposals.append((pname, mine, theirs))
        if acc:
            if pname == "A":
                ep.native_split = (mine, theirs)
                a.call(f"The other agent ACCEPTED your proposal. Deal executed: you get {mine}, "
                       f"the other agent gets {theirs}.")
            else:
                ep.native_split = (theirs, mine)
                b.call(f"The other agent ACCEPTED your proposal. Deal executed: you get {mine}, "
                       f"the other agent gets {theirs}.")
            break
    return ep


def run_sealed(a, b, R, T, ep):
    prior = ""
    for r in range(1, T + 1):
        P = pie(R, r)
        da, ra = _ask_parsed(
            a, f"Round {r}. The pie this round is {P} points. {prior}"
               f"State your demand for yourself. Reply exactly: DEMAND: <int>",
            parse_demand, "Reply exactly: DEMAND: <int>")
        db, rb = _ask_parsed(
            b, f"Round {r}. The pie this round is {P} points. {prior}"
               f"State your demand for yourself. Reply exactly: DEMAND: <int>",
            parse_demand, "Reply exactly: DEMAND: <int>")
        if da is None or db is None:
            ep.log(round=r, status="unparseable_demand", raw_a=ra, raw_b=rb)
            prior = "Last round a reply could not be read, so no deal was executed. "
            continue
        da, db = max(0, da), max(0, db)
        ep.bids.append((r, da, db))
        ok = (da + db) <= P
        ep.log(round=r, pie=P, demand_a=da, demand_b=db, compatible=ok)
        ep.rounds = r
        if ok:
            ep.native_split = (da, db)
            a.call(f"A deal was executed: you get {da}, the other agent gets {db}. "
                   f"The {P - da - db} unclaimed points are lost.")
            b.call(f"A deal was executed: you get {db}, the other agent gets {da}. "
                   f"The {P - da - db} unclaimed points are lost.")
            break
        prior = f"Last round your demand plus the other agent's demand exceeded {P}, so no deal was executed. "
    return ep


def parse_confirm(text):
    t = text.strip().lower()
    if re.search(r"\b(not|n't|cannot|can't|won't|don't|unable)\b[^.]{0,25}\bconfirm", t):
        return False
    if re.search(r'\breject|refuse|decline\b', t):
        return False
    if re.search(r'\bconfirm(ed|ation)?\b', t):
        return True
    return None


_ACC = re.compile(r'\b(accept|accepted|agree|agreed|deal|yes|ok|okay)\b')


def _splits_in(text):
    t = (text or "").lower().replace("–", "-").replace("—", "-")
    out = []
    for m in re.finditer(r'\b(\d{1,2})\s*[-/]\s*(\d{1,2})\b', t):
        out.append((int(m.group(1)), int(m.group(2))))
    for m in re.finditer(r'(\d{1,2})\s*for me[^.\n]{0,20}?(\d{1,2})\s*for you', t):
        out.append((int(m.group(1)), int(m.group(2))))
    for m in re.finditer(r'(\d{1,2})\s*for you[^.\n]{0,20}?(\d{1,2})\s*for me', t):
        out.append((int(m.group(2)), int(m.group(1))))
    for m in re.finditer(r'(\d{1,2})\s*each\b', t):
        out.append((int(m.group(1)), int(m.group(1))))
    return out



_NEG_ACCEPT = re.compile(r"\b(not|no|n't|cannot|can't|won't|don't|unable|never)\b[^.]{0,25}"
                         r"\b(accept|accepted|agree|agreed|deal|yes)\b")


def _has_accept(text):
    """Acceptance token present and not negated ("no deal", "cannot accept")."""
    t = (text or "").lower()
    return bool(_ACC.search(t)) and not bool(_NEG_ACCEPT.search(t))

def strict_free_text_deal(transcript, R):
    """Deterministic detector for free dialogue.

    A deal is certified only when, in the SAME round, both agents use an explicit
    acceptance token and both name the same symmetric split that fits the pie of
    that round. Asymmetric agreements are deliberately not certified (they are
    flagged for manual audit) because the 'me/you' orientation cannot be resolved
    without a parser that reads intent.
    """
    for e in reversed(transcript):
        a, b = e.get("a"), e.get("b")
        if not a or not b:
            continue
        if not _has_accept(a) or not _has_accept(b):
            continue
        rnd = e.get("round") or 1
        P = pie(R, rnd)
        sa, sb = set(_splits_in(a)), set(_splits_in(b))
        for x, y in sa & sb:
            if x == y and 0 < x + y <= P:
                return (x, y), rnd
    return None, None


def run_sealed_compromise(a, b, R, T, ep, verify):
    """SPB: sealed demands + equal-split fallback (+ optional commitment check)."""
    prior = ""

    def close(split_a, split_b, why):
        if verify:
            ra = a.call(f"A deal is proposed ({why}): you get {split_a}, the other agent gets {split_b}. "
                        f"Reply CONFIRM to commit, or REJECT.")
            rb = b.call(f"A deal is proposed ({why}): you get {split_b}, the other agent gets {split_a}. "
                        f"Reply CONFIRM to commit, or REJECT.")
            ca, cb = parse_confirm(ra), parse_confirm(rb)
            ep.log(status="commitment_check", proposed=[split_a, split_b], confirm_a=ca, confirm_b=cb)
            if ca and cb:
                return True
            ep.confirm_failed += 1
            return False
        a.call(f"Deal executed: you get {split_a}, the other agent gets {split_b}.")
        b.call(f"Deal executed: you get {split_b}, the other agent gets {split_a}.")
        return True

    for r in range(1, T + 1):
        P = pie(R, r)
        da, ra_ = _ask_parsed(
            a, f"Round {r}. The pie this round is {P} points. {prior}"
               f"State your demand for yourself. Reply exactly: DEMAND: <int>",
            parse_demand, "Reply exactly: DEMAND: <int>")
        db, rb_ = _ask_parsed(
            b, f"Round {r}. The pie this round is {P} points. {prior}"
               f"State your demand for yourself. Reply exactly: DEMAND: <int>",
            parse_demand, "Reply exactly: DEMAND: <int>")
        if da is None or db is None:
            ep.log(round=r, status="unparseable_demand", raw_a=ra_, raw_b=rb_)
            prior = "Last round a reply could not be read, so no deal was executed. "
            continue
        da, db = max(0, da), max(0, db)
        ep.bids.append((r, da, db))
        ep.rounds = r
        compatible = (da + db) <= P
        ep.log(round=r, pie=P, demand_a=da, demand_b=db, compatible=compatible)

        if compatible:
            if close(da, db, "both demands compatible"):
                ep.native_split = (da, db)
                break
            prior = f"Last round the deal at your demand was not confirmed, so it is void. "
            continue

        # incompatible -> proportional compromise on the current pie
        tot = da + db
        x = int(round(da / tot * P)) if tot > 0 else P // 2
        x = max(1, min(P - 1, x))
        y = P - x
        ep.fallback_offered += 1
        aa = a.call(f"Your demands are incompatible. The mechanism proposes a proportional "
                    f"compromise: you get {x}, the other agent gets {y}. Reply ACCEPT or REJECT.")
        ab = b.call(f"Your demands are incompatible. The mechanism proposes a proportional "
                    f"compromise: you get {y}, the other agent gets {x}. Reply ACCEPT or REJECT.")
        ca, cb = parse_accept(aa), parse_accept(ab)
        ep.log(round=r, fallback=[x, y], accept_a=ca, accept_b=cb)
        if ca and cb:
            ep.fallback_accepted += 1
            if close(x, y, "equal-split compromise"):
                ep.native_split = (x, y)
                break
            prior = "Last round the equal-split compromise was not confirmed, so it is void. "
        else:
            prior = ("Last round the demands were incompatible and the equal-split offer was "
                     "rejected, so no deal was executed. ")
    return ep


def run_mediated(a, b, med, R, T, ep):
    prior = ""
    for r in range(1, T + 1):
        P = pie(R, r)
        pa = a.call(f"Round {r}. The pie this round is {P} points. "
                    f"Send a private message to the mediator: what split do you want, and why?")
        pb = b.call(f"Round {r}. The pie this round is {P} points. "
                    f"Send a private message to the mediator: what split do you want, and why?")
        sug, raw = _ask_parsed(
            med,
            f"Round {r}. The pie this round is {P} points.\n{prior}"
            f"Agent A says: \"{pa}\"\nAgent B says: \"{pb}\"\n"
            f"Propose a split. Reply exactly: SPLIT: A=<int>, B=<int>",
            parse_ab, "Reply exactly: SPLIT: A=<int>, B=<int>")
        if sug is None:
            ep.log(round=r, status="unparseable_suggestion", raw=raw)
            prior = "Last round your reply could not be read, so no proposal was made. "
            continue
        x, y = sug
        if x + y > P:
            y = max(0, P - x)
        ra = a.call(f"The mediator proposes: A gets {x}, B gets {y}.\nReply ACCEPT or REJECT.")
        rb = b.call(f"The mediator proposes: A gets {x}, B gets {y}.\nReply ACCEPT or REJECT.")
        aa, ab = parse_accept(ra), parse_accept(rb)
        ep.log(round=r, suggestion=[x, y], reply_a=ra, reply_b=rb, accept_a=aa, accept_b=ab)
        ep.rounds = r
        if aa and ab:
            ep.native_split = (x, y)
            a.call(f"Both agents accepted. Deal executed: you get {x}.")
            b.call(f"Both agents accepted. Deal executed: you get {y}.")
            break
        prior = (f"Last round you proposed A={x}, B={y}. Agent A "
                 f"{'accepted' if aa else 'rejected'}; Agent B "
                 f"{'accepted' if ab else 'rejected'}. ")
    return ep


# ────────────────────────────────────────────────────────────── elicitation
def elicit(agent):
    """Protocol-neutral outcome measurement. Returns (agreed, split, raw)."""
    txt = agent.call(ELICIT_PROMPT)
    raws = [txt]
    for attempt in range(2):
        m_ag = re.search(r'AGREED\s*:\s*(YES|NO)', txt, re.I)
        m_sp = re.search(r'SPLIT\s*:\s*A\s*=\s*(\d+)\s*,\s*B\s*=\s*(\d+)', txt, re.I)
        if m_ag:
            if m_ag.group(1).upper() == "NO":
                return False, None, " | ".join(raws)
            if m_sp:
                return True, (int(m_sp.group(1)), int(m_sp.group(2))), " | ".join(raws)
        if attempt == 0:
            txt = agent.call("Reply with exactly two lines:\nAGREED: YES or NO\nSPLIT: A=<int>, B=<int>")
            raws.append(txt)
    return None, None, " | ".join(raws)


# ────────────────────────────────────────────────────────────── one episode
def play_episode(model_key, protocol, R, T, trial, k, K, prior, temperature,
                 budget, transcript_dir):
    a = Agent("A", build_system("A", protocol, R, T), model_key, temperature, budget, None)
    b = Agent("B", build_system("B", protocol, R, T), model_key, temperature, budget, None)
    med = None
    if protocol == "mediated":
        med = Agent("Mediator", MEDIATOR_SYSTEM.format(R=R), model_key,
                    temperature, budget, None)

    # episode context (outcome feedback only)
    if k == 1:
        ctx = f"Episode {k} of {K}. This is the first episode."
    else:
        lines = [f"Episode {k} of {K}. You are playing again with the same other agent."]
        for p in prior:
            if p["agreed"]:
                lines.append(f"Episode {p['episode']}: agreement reached. "
                             f"Agent A received {p['split_a']}, Agent B received {p['split_b']}.")
            else:
                lines.append(f"Episode {p['episode']}: no agreement. Both received 0.")
        lines.append("You may use this history however you wish.")
        ctx = "\n".join(lines)

    if protocol == "mediated":
        med.call(ctx + f"\nThis is episode {k} of {K} with the same two agents. Start mediating.")
        a.call(ctx + "\nRound 1 begins.")
        b.call(ctx + "\nRound 1 begins.")
    else:
        a.call(ctx + "\nRound 1 begins.")
        b.call(ctx + "\nRound 1 begins.")

    ep = Episode()
    t0 = time.time()
    error = ""
    try:
        if protocol == "free_text":
            run_free_text(a, b, R, T, ep)
        elif protocol == "free_text_exec":
            run_free_text_exec(a, b, R, T, ep)
        elif protocol == "free_text_loose":
            run_free_text_loose(a, b, R, T, ep)
        elif protocol == "alternating_offers":
            run_alternating(a, b, R, T, ep)
        elif protocol == "sealed_bid":
            run_sealed(a, b, R, T, ep)
        elif protocol == "sealed_compromise":
            run_sealed_compromise(a, b, R, T, ep, verify=False)
        elif protocol == "sealed_compromise_verify":
            run_sealed_compromise(a, b, R, T, ep, verify=True)
        elif protocol == "mediated":
            run_mediated(a, b, med, R, T, ep)
    except BudgetExceeded:
        raise
    except Exception as e:  # noqa: BLE001
        error = f"{type(e).__name__}: {e}"

    # measurement, identical across protocols
    ag_a, sp_a = (None, None)
    ag_b, sp_b = (None, None)
    raw_a = raw_b = ""
    try:
        ag_a, sp_a, raw_a = elicit(a)
        ag_b, sp_b, raw_b = elicit(b)
    except BudgetExceeded:
        raise
    except Exception as e:  # noqa: BLE001
        error = (error + " | " if error else "") + f"elicit:{type(e).__name__}:{e}"

    # ── outcome measurement ────────────────────────────────────────────────
    # Structured protocols have a mechanism-defined executed deal. Free dialogue
    # has none, so its only operational definition of a deal is a meeting of
    # minds: both agents independently report the same split. Elicitation is
    # therefore primary for free_text and secondary (a perception check) for the
    # structured protocols.
    native = ep.native_split
    strict_split, strict_round = (None, None)
    if protocol == "free_text":
        strict_split, strict_round = strict_free_text_deal(ep.transcript, R)

    deal_round = strict_round if protocol == "free_text" else ep.rounds
    P_final = pie(R, deal_round or (ep.rounds or T))
    elicit_split = None
    if ag_a and ag_b and sp_a and sp_b and sp_a == sp_b and (sp_a[0] + sp_a[1]) <= P_final:
        elicit_split = sp_a

    deal = strict_split if protocol == "free_text" else native
    agreed = deal is not None
    split = deal if deal is not None else (None, None)

    row = dict(
        model=model_key, model_id=MODELS[model_key]["id"], protocol=protocol,
        R=R, T=T, K=K, trial=trial, episode=k,
        deal=int(agreed),
        deal_strict=int(strict_split is not None),
        strict_round=strict_round or 0,
        agree_elicited=int(elicit_split is not None),
        agree_native=int(native is not None),
        agree_legacy=int(ep.legacy_split is not None),
        legacy_round=ep.legacy_round,
        perceived_mismatch=int((native is not None) != (elicit_split is not None)),
        split_a=split[0], split_b=split[1],
        allocated=(split[0] + split[1]) if agreed else 0,
        welfare=(split[0] + split[1]) if agreed else 0,
        pie_final=P_final,
        rounds=ep.rounds or T,
        one_sided_claim=int(bool(ag_a) != bool(ag_b)),
        tokens_in=a.usage_in + b.usage_in + (med.usage_in if med else 0),
        tokens_out=a.usage_out + b.usage_out + (med.usage_out if med else 0),
        calls=a.calls + b.calls + (med.calls if med else 0),
        wall_s=round(time.time() - t0, 1),
        status="ok" if not error else "error",
        error=error,
    )

    if protocol.startswith("sealed") and ep.bids:
        sums = [x + y for _, x, y in ep.bids]
        row["bid_sum_first"] = sums[0]
        row["bid_sum_mean"] = round(sum(sums) / len(sums), 2)
        row["lowball_any"] = int(any(s < R for s in sums))
        row["lowball_rounds"] = sum(1 for s in sums if s < R)
        row["demand_a_mean"] = round(sum(x for _, x, _ in ep.bids) / len(ep.bids), 2)
        row["demand_b_mean"] = round(sum(y for _, _, y in ep.bids) / len(ep.bids), 2)
    if protocol.startswith("sealed_compromise"):
        row["fallback_offered"] = ep.fallback_offered
        row["fallback_accepted"] = ep.fallback_accepted
        row["confirm_failed"] = ep.confirm_failed

    transcript = dict(row=row, transcript=ep.transcript,
                      bids=ep.bids, proposals=ep.proposals,
                      elicit=dict(a=raw_a, b=raw_b, agreed_a=ag_a, agreed_b=ag_b,
                                  split_a=sp_a, split_b=sp_b),
                      legacy=dict(split=ep.legacy_split, round=ep.legacy_round))
    path = transcript_dir / f"{model_key}_{protocol}_{trial:03d}_e{k}.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(transcript, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
    return row


# ────────────────────────────────────────────────────────────── main
FIELDS = ["model", "model_id", "protocol", "R", "T", "K", "trial", "episode",
          "deal", "deal_strict", "strict_round", "agree_elicited", "agree_native", "agree_legacy", "legacy_round",
          "perceived_mismatch", "split_a", "split_b",
          "allocated", "welfare", "pie_final", "rounds", "one_sided_claim", "tokens_in", "tokens_out",
          "calls", "wall_s", "status", "error",
          "bid_sum_first", "bid_sum_mean", "lowball_any", "lowball_rounds",
          "demand_a_mean", "demand_b_mean", "fallback_offered", "fallback_accepted",
          "confirm_failed"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["flash"])
    p.add_argument("--protocols", nargs="+", default=PROTOCOLS)
    p.add_argument("--trials", type=int, default=30)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--R", type=int, default=10)
    p.add_argument("--temp", type=float, default=0.3)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--budget-tokens", type=int, default=60_000_000)
    p.add_argument("--out", default=str(ROOT / "output_v3" / "run1"))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--tag", default="")
    args = p.parse_args()

    if args.dry_run:
        args.trials, args.episodes, args.workers = 1, 1, 1

    out = Path(args.out)
    tdir = out / "transcripts"
    tdir.mkdir(parents=True, exist_ok=True)
    csv_path = out / "episodes.csv"
    log_path = out / "run.log"

    budget = Budget(args.budget_tokens)
    lock = threading.Lock()

    tasks = []
    for model_key in args.models:
        for protocol in args.protocols:
            for trial in range(1, args.trials + 1):
                tasks.append((model_key, protocol, trial))
    random.Random(20261023).shuffle(tasks)

    done = set()
    if csv_path.exists():
        with csv_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("status") == "ok":
                    done.add((row["model"], row["protocol"], row["trial"], row["episode"]))

    def work(task):
        model_key, protocol, trial = task
        prior = []
        rows = []
        for k in range(1, args.episodes + 1):
            if (model_key, protocol, str(trial), str(k)) in done:
                continue
            tpath = tdir / f"{model_key}_{protocol}_{trial:03d}_e{k}.json"
            if tpath.exists():
                # resume from a transcript written before a restart
                try:
                    d = json.loads(tpath.read_text(encoding="utf-8"))
                    row = d["row"]
                    if row.get("status") != "ok":
                        raise ValueError("previous attempt failed; re-run")
                    prior.append(dict(episode=k, agreed=bool(row["deal"]),
                                      split_a=row["split_a"], split_b=row["split_b"]))
                    rows.append(row)
                    continue
                except Exception:  # noqa: BLE001
                    pass
            row = play_episode(model_key, protocol, args.R, args.rounds, trial, k,
                               args.episodes, prior, args.temp, budget, tdir)
            prior.append(dict(episode=k, agreed=bool(row["deal"]),
                              split_a=row["split_a"], split_b=row["split_b"]))
            rows.append(row)
        return rows

    print(f"[start] {len(tasks)} trials, {args.episodes} episodes each, "
          f"workers={args.workers}, budget={args.budget_tokens/1e6:.0f}M tokens", flush=True)

    fout = csv_path.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(fout, fieldnames=FIELDS, extrasaction="ignore")
    if fout.tell() == 0:
        writer.writeheader()

    n_ok = n_err = 0
    t_start = time.time()
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(work, t): t for t in tasks}
            for fut in as_completed(futs):
                task = futs[fut]
                try:
                    rows = fut.result()
                except BudgetExceeded as e:
                    print(f"[budget] {e}", flush=True)
                    pool.shutdown(wait=False, cancel_futures=True)
                    break
                except Exception as e:  # noqa: BLE001
                    print(f"[fail] {task}: {type(e).__name__}: {e}", flush=True)
                    traceback.print_exc()
                    n_err += 1
                    continue
                with lock:
                    for row in rows:
                        writer.writerow(row)
                    fout.flush()
                n_ok += len(rows)
                msg = (f"[{n_ok + n_err}] {task[0]}/{task[1]}/{task[2]:03d} "
                       f"rows={len(rows)} tokens={budget.spent()/1e6:.2f}M "
                       f"elapsed={(time.time()-t_start)/60:.1f}m")
                print(msg, flush=True)
                with log_path.open("a", encoding="utf-8") as lf:
                    lf.write(msg + "\n")
    finally:
        fout.close()

    print(f"[done] rows={n_ok} errors={n_err} tokens={budget.spent()} "
          f"minutes={(time.time()-t_start)/60:.1f}", flush=True)


if __name__ == "__main__":
    main()
