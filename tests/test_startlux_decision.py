"""CPU business-semantics tests for the StartLux-Decision v1.1 0.8B Instant P2a bridge.

The read-out backend is controlled: logits are HIGH (2.0) exactly for the option
letters whose rendered option line carries the marker GOLDMARK (the backend parses
the official "Options:" block of the prompt), LOW (0.0) otherwise, plus a gold_last
mode that scores the extra "None of the above" position.  Expected answers are
computed in this file from the documented business spec with independent formulas
(the spec's softmax/confidence/None/knock-out math, re-derived here, plus literal
anchor numbers); the implementation's own helpers are never used to produce them.

The tokenizer is the official FakeTokenizer from the vendored package (the chat
layout with the thinking-off suffix, one token per letter).  Real model tokenizer
materials are not downloaded yet: real-tokenizer integration is pending (P3) and is
NOT claimed by these tests.
"""
import asyncio
import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path

MODEL_DIR = Path("models/startlux-decision-v1.1-0.8b-instant").resolve()
_spec = importlib.util.spec_from_file_location("startlux_decision_p2a_bridge", MODEL_DIR / "decider" / "decision.py")
bridge = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bridge)      # also puts the vendored package on sys.path

from startlux_decide.backends.base import Backend          # noqa: E402
from startlux_decide.backends.fake import FakeTokenizer    # noqa: E402
from startlux_decide.controls import RequestError          # noqa: E402
from startlux_decide.prompt import THINK_OFF_SUFFIX        # noqa: E402

TEMPS = {"choice": 0.7, "noul": 1.3, "score": 0.9}      # distinct, none equal to 1.0
WIDE = {"group": 25, "keep": 3, "residual": 0.001}
LETTERS = list(range(65, 91))                            # FakeTokenizer: ord('A')..ord('Z')
HIGH, LOW, MARK = 2.0, 0.0, "GOLDMARK"
PICK = {"instructions": "Pick one."}
RATE = {"instructions": "Rate the evidence."}


def make_folder(root, temperatures=TEMPS, letters=LETTERS, attention="bidirectional",
                decision=None, wide=WIDE):
    d = Path(root) / "model"
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps(
        {"jev_prompt_attention": attention, "max_position_embeddings": 262144}))
    cfg = {"prompt_attention": "bidirectional", "temperature_by_type": temperatures,
           "letter_token_ids": letters, "max_options_per_pass": 26, "wide_choice": wide,
           "format": "StartLux-Decision-v1.1", "release_date": "2026-10-30"}
    cfg.update(decision or {})
    (d / "decision_config.json").write_text(json.dumps(cfg))
    return d


# ---------------------------------------------------------------- controlled read-out backend

class ControlledBackend(Backend):
    """Read-outs driven by markers in the rendered option lines: a letter scores HIGH
    when its option line's body contains MARK; gold_last scores the last option (the
    extra "None of the above") HIGH instead.  Every call is recorded."""

    name = "controlled"

    def __init__(self, tokenizer, mark=MARK, high=HIGH, low=LOW, gold_last=False):
        self.tokenizer = tokenizer
        self.mark, self.high, self.low, self.gold_last = mark, high, low, gold_last
        self.calls = []

    async def readout(self, prompt, letter_ids, media=None):
        self.calls.append((prompt, tuple(letter_ids)))
        block = prompt.rsplit("\nOptions:\n", 1)[1]
        out = []
        for k in range(len(letter_ids)):
            prefix = f"{chr(65 + k)}) "
            body = next((l[len(prefix):] for l in block.split("\n") if l.startswith(prefix)), "")
            if self.gold_last and k == len(letter_ids) - 1:
                out.append(self.high)
            else:
                out.append(self.high if self.mark in body else self.low)
        return out

    async def generate(self, *args, **kwargs):
        raise NotImplementedError("the controlled backend never generates")


# ---------------------------------------------------------------- independent spec formulas

def softmax(z, t):
    """Spec: softmax(z / temperature) over the listed options."""
    m = max(z)
    e = [math.exp((x - m) / t) for x in z]
    s = sum(e)
    return [x / s for x in e]


def choice_confidence(p):
    """Spec: (n * max(p) - 1) / (n - 1), clamped to [0, 1]."""
    n = len(p)
    return 1.0 if n < 2 else min(1.0, max(0.0, (n * max(p) - 1) / (n - 1)))


def score_confidence(p):
    """Spec: 1 - probability-weighted distance in levels from the argmax, relative to
    the even-spread distance from the middle level, floored at 0."""
    n = len(p)
    m = max(range(n), key=lambda i: p[i])
    spread = sum(v * abs(i - m) for i, v in enumerate(p))
    even = sum(abs(i - (n - 1) / 2) for i in range(n)) / n
    return max(0.0, 1 - spread / even) if even else 1.0


def split_none(p):
    """Spec: drop the last (extra) option; renormalise the listed ones to sum to 1;
    abstain when the extra one is strictly above every listed one."""
    listed, none = list(p[:-1]), p[-1]
    total = sum(listed)
    renorm = [v / total for v in listed] if total > 0 else [1.0 / len(listed)] * len(listed)
    return renorm, none, none > max(listed)


def tokens(tok, prompt):
    return len(tok.encode(prompt, add_special_tokens=False))


# ---------------------------------------------------------------- test base

class BridgeCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.folder = make_folder(self._tmp.name)
        self.tok = FakeTokenizer()
        self.backend = ControlledBackend(self.tok)
        self.d = bridge.StartluxDecision(self.folder, self.tok, backend=self.backend)

    def tearDown(self):
        self._tmp.cleanup()

    def make(self, backend=None, **kw):
        return bridge.StartluxDecision(self.folder, self.tok, backend=backend or self.backend, **kw)


# ---------------------------------------------------------------- folder / tokenizer contract

class FolderAndTokenizerContract(BridgeCase):

    def test_folder_is_instant_with_configured_values(self):
        f = self.d.folder
        self.assertEqual(f.kind, "instant")
        self.assertTrue(f.bidirectional)
        self.assertIsNone(f.auto_think)                        # Instant folders never think
        self.assertEqual(dict(f.temperature_by_type), TEMPS)   # temperatures come from the folder
        self.assertEqual(list(f.letter_token_ids), LETTERS)
        self.assertEqual((f.wide_group, f.wide_keep, f.wide_residual), (25, 3, 0.001))

    def test_letter_ids_come_from_decision_config_and_match_the_tokenizer(self):
        # Decider.__init__ runs the official check_tokenizer (single-token letters right
        # after the thinking-off prompt) and compares against decision_config.json.
        self.assertEqual(self.d.letter_ids, tuple(LETTERS))
        self.assertEqual(self.d.temperature("choice"), 0.7)
        self.assertEqual(self.d.temperature("noul"), 1.3)
        self.assertEqual(self.d.temperature("score"), 0.9)

    def test_mismatched_letter_token_ids_are_refused(self):
        folder = make_folder(self._tmp.name, letters=[x + 1 for x in LETTERS])
        with self.assertRaises(ValueError) as e:
            bridge.StartluxDecision(folder, self.tok, backend=self.backend)
        self.assertIn("letter ids differ", str(e.exception))

    def test_instant_folder_must_not_carry_auto_think(self):
        folder = make_folder(self._tmp.name, decision={"auto_think": {"gate": 0.5}})
        with self.assertRaises(ValueError) as e:
            bridge.StartluxDecision(folder, self.tok, backend=self.backend)
        self.assertIn("does not think", str(e.exception))

    def test_prompt_attention_must_agree_between_the_two_configs(self):
        folder = make_folder(self._tmp.name, attention="causal")   # decision_config still bidirectional
        with self.assertRaises(ValueError) as e:
            bridge.StartluxDecision(folder, self.tok, backend=self.backend)
        self.assertIn("prompt attention", str(e.exception))

    def test_bridge_refuses_non_instant_folders(self):
        folder = make_folder(self._tmp.name, attention="causal",
                             decision={"prompt_attention": None})
        cfg = json.loads((folder / "decision_config.json").read_text())
        del cfg["prompt_attention"]            # a causal folder without overrides
        (folder / "decision_config.json").write_text(json.dumps(cfg))
        with self.assertRaises(ValueError) as e:
            bridge.StartluxDecision(folder, self.tok, backend=self.backend)
        self.assertIn("Instant", str(e.exception))

    def test_bad_temperature_by_type_is_refused(self):
        folder = make_folder(self._tmp.name, temperatures={"choice": -0.5, "noul": 1.3, "score": 0.9})
        with self.assertRaises(ValueError):
            bridge.StartluxDecision(folder, self.tok, backend=self.backend)


# ---------------------------------------------------------------- choice semantics

class ChoiceSemantics(BridgeCase):

    def test_choice_probabilities_choice_and_confidence(self):
        # bare ids + criteria everywhere: the ids are hidden, the criteria are the bodies
        d = self.make(none_check=False)
        decision = d.decide("s", {"q": dict(PICK, criteria={
            "a": "plain", "b": "has GOLDMARK inside", "c": "plain"})})
        p = softmax([LOW, HIGH, LOW], TEMPS["choice"])
        ans = decision.answers["q"]
        self.assertEqual(ans["type"], "choice")
        self.assertEqual(ans["choice"], "b")
        self.assertAlmostEqual(ans["probabilities"]["b"], p[1], places=12)
        self.assertAlmostEqual(ans["probabilities"]["b"], 0.8969, places=3)   # literal anchor
        self.assertAlmostEqual(ans["probabilities"]["a"], p[0], places=12)
        self.assertAlmostEqual(ans["probabilities"]["c"], p[2], places=12)
        self.assertAlmostEqual(ans["confidence"], choice_confidence(p), places=12)
        self.assertNotIn("none_probability", ans)
        self.assertEqual(list(ans), ["type", "choice", "confidence", "probabilities"])
        # one read-out, over the configured letter ids, on the official one-pass prompt
        self.assertEqual(len(self.backend.calls), 1)
        self.assertEqual(self.backend.calls[0][1], tuple(LETTERS[:3]))
        self.assertTrue(self.backend.calls[0][0].endswith(THINK_OFF_SUFFIX))

    def test_option_order_follows_the_request(self):
        d = self.make(none_check=False)
        spec = dict(PICK, criteria={"zzz": None, "mmm": "GOLDMARK", "aaa": None})
        decision = d.decide("s", {"q": spec})
        prompt = self.backend.calls[0][0]
        lines = [l for l in prompt.split("\n") if len(l) > 3 and l[1:3] == ") "]
        self.assertEqual(lines, ["A) zzz", "B) mmm: GOLDMARK", "C) aaa<|im_end|>"])  # request order -> letters
        self.assertEqual(decision.answers["q"]["choice"], "mmm")            # letter B is the 2nd listed
        self.assertEqual(list(decision.answers["q"]["probabilities"]), ["zzz", "mmm", "aaa"])

    def test_ties_go_to_the_first_listed_option(self):
        d = self.make(none_check=False)
        decision = d.decide("s", {"q": dict(PICK, criteria={"x": "d1", "y": "d2", "z": "d3"})})
        ans = decision.answers["q"]
        self.assertEqual(ans["choice"], "x")
        for v in ans["probabilities"].values():
            self.assertAlmostEqual(v, 1 / 3, places=12)
        self.assertAlmostEqual(ans["confidence"], 0.0, places=12)

    def test_single_option_is_answered_without_the_model(self):
        d = self.make(none_check=False)
        decision = d.decide("s", {"q": {"criteria": {"only": "the one"}}})
        self.assertEqual(decision.answers["q"], {"type": "choice", "choice": "only",
                                                  "confidence": 1.0, "probabilities": {"only": 1.0}})
        self.assertEqual(self.backend.calls, [])                 # no read-out at all
        self.assertEqual(decision.usage, {"input_tokens": 0, "output_tokens": 0})

    def test_none_check_default_adds_the_extra_option_without_null_choice(self):
        # bridge default none_check=True (the official serve default): every choice
        # question carries "None of the above" and reports none_probability, but the
        # choice stays a listed option (allow_none is not set).
        decision = self.d.decide("s", {"q": dict(PICK, criteria={"x": "d1", "y": "d2", "z": "d3"})})
        p = softmax([LOW, LOW, LOW, LOW], TEMPS["choice"])
        listed, none, abstain = split_none(p)
        ans = decision.answers["q"]
        self.assertEqual(ans["choice"], "x")                      # tie -> first listed
        self.assertAlmostEqual(ans["none_probability"], none, places=12)
        self.assertAlmostEqual(ans["none_probability"], 0.25, places=12)
        self.assertFalse(abstain)
        for i, k in enumerate(("x", "y", "z")):
            self.assertAlmostEqual(ans["probabilities"][k], listed[i], places=12)
        self.assertEqual(len(self.backend.calls), 1)
        self.assertEqual(self.backend.calls[0][1], tuple(LETTERS[:4]))   # 3 listed + the extra option

    def test_allow_none_reports_null_choice_when_none_wins(self):
        self.backend.gold_last = True
        decision = self.d.decide("s", {"q": dict(PICK, criteria={"x": "d1", "y": "d2"})},
                                 body={"allow_none": True})
        p = softmax([LOW, LOW, HIGH], TEMPS["choice"])
        listed, none, abstain = split_none(p)
        ans = decision.answers["q"]
        self.assertIsNone(ans["choice"])
        self.assertTrue(abstain)
        self.assertAlmostEqual(ans["none_probability"], none, places=12)
        self.assertAlmostEqual(ans["none_probability"], 0.8969, places=3)
        self.assertAlmostEqual(ans["probabilities"]["x"], listed[0], places=12)
        self.assertAlmostEqual(ans["confidence"], choice_confidence(listed), places=12)

    def test_none_check_alone_keeps_a_listed_choice_when_none_wins(self):
        self.backend.gold_last = True
        decision = self.d.decide("s", {"q": dict(PICK, criteria={"x": "d1", "y": "d2"})})  # no allow_none
        ans = decision.answers["q"]
        self.assertEqual(ans["choice"], "x")                     # null_ok is False under none_check
        self.assertAlmostEqual(ans["none_probability"], softmax([0, 0, 2], 0.7)[2], places=12)

    def test_question_level_allow_none_overrides_the_request(self):
        self.backend.gold_last = True
        decision = self.d.decide("s", {"q": dict(PICK, allow_none=False,
                                                 criteria={"x": "d1", "y": "d2"})},
                                 body={"allow_none": True})
        self.assertEqual(decision.answers["q"]["choice"], "x")   # question override wins
        self.assertIn("none_probability", decision.answers["q"])

    def test_question_level_allow_none_true_nulls_the_choice(self):
        self.backend.gold_last = True
        decision = self.d.decide("s", {"q": dict(PICK, allow_none=True,
                                                 criteria={"x": "d1", "y": "d2"})},
                                 body={"allow_none": False})
        self.assertIsNone(decision.answers["q"]["choice"])

    def test_single_option_with_allow_none_keeps_two_options(self):
        decision = self.d.decide("s", {"q": dict(PICK, criteria={"only": "GOLDMARK"})},
                                 body={"allow_none": True})
        p = softmax([HIGH, LOW], TEMPS["choice"])                # the extra option is the second
        ans = decision.answers["q"]
        self.assertEqual(ans["choice"], "only")
        self.assertAlmostEqual(ans["probabilities"]["only"], 1.0, places=12)   # renormalised listed
        self.assertAlmostEqual(ans["none_probability"], p[1], places=12)
        self.assertAlmostEqual(ans["none_probability"], 0.0543, places=3)
        self.assertAlmostEqual(ans["confidence"], 1.0, places=12)
        self.assertEqual(len(self.backend.calls), 1)             # the model was consulted

    def test_options_key_and_question_key_aliases(self):
        d = self.make(none_check=False)
        decision = d.decide("s", {
            "q": dict(PICK, options=["first", "second-GOLDMARK"]),
            "r": {"type": "choice", "question": "Which?",
                  "criteria": {"first": None, "second": "GOLDMARK"}}})
        self.assertEqual(decision.answers["q"]["choice"], "second-GOLDMARK")
        self.assertEqual(decision.answers["r"]["choice"], "second")
        self.assertIn("Question: Pick one.", self.backend.calls[0][0])
        self.assertIn("Question: Which?", self.backend.calls[1][0])


# ---------------------------------------------------------------- noul / score semantics

class NoulAndScoreSemantics(BridgeCase):

    def test_noul_answers_p_of_true_at_the_noul_temperature(self):
        decision = self.d.decide("s", {"q": {"type": "noul", "criteria": {
            "true": "the evidence says GOLDMARK", "false": "or it does not"}}})
        p = softmax([HIGH, LOW], TEMPS["noul"])                  # 1.3, not the choice temperature
        self.assertEqual(decision.answers["q"], {"type": "noul", "noul": p[0]})
        self.assertAlmostEqual(decision.answers["q"]["noul"], 0.823, places=3)   # literal anchor
        self.assertEqual(len(self.backend.calls), 1)
        self.assertEqual(self.backend.calls[0][1], tuple(LETTERS[:2]))          # none_check skips noul
        self.assertIn("A) yes: the evidence says GOLDMARK", self.backend.calls[0][0])

    def test_noul_default_instructions_and_plain_bodies(self):
        decision = self.d.decide("s", {"q": {"type": "noul"}})
        self.assertEqual(decision.answers["q"]["type"], "noul")
        self.assertAlmostEqual(decision.answers["q"]["noul"], 0.5, places=12)   # no marker: a tie
        self.assertIn("Question: Which answer fits the evidence?", self.backend.calls[0][0])
        self.assertIn("A) yes", self.backend.calls[0][0])
        self.assertIn("B) no", self.backend.calls[0][0])

    def test_score_levels_numeric_order_expectation_and_legend(self):
        decision = self.d.decide("s", {"q": dict(RATE, type="score", criteria={
            "2": "high-GOLDMARK", "0": "low", "1": "mid level"})})
        p = softmax([LOW, LOW, HIGH], TEMPS["score"])            # levels sorted 0, 1, 2 (lowest first)
        ans = decision.answers["q"]
        self.assertEqual(ans["type"], "score")
        self.assertAlmostEqual(ans["score"], sum(i * v for i, v in enumerate(p)), places=12)
        self.assertAlmostEqual(ans["score"], 1.733, places=3)                  # literal anchor
        self.assertAlmostEqual(ans["confidence"], score_confidence(p), places=12)
        self.assertEqual(ans["legend"], {"0": "low", "1": "mid level", "2": "high-GOLDMARK"})
        self.assertEqual(ans["probabilities"], {str(i): p[i] for i in range(3)})
        self.assertNotIn("none_probability", ans)                # none_check skips score
        self.assertEqual(self.backend.calls[0][1], tuple(LETTERS[:3]))

    def test_score_from_a_list_keeps_the_given_level_order(self):
        decision = self.d.decide("s", {"q": dict(RATE, type="score",
                                                criteria=["bad", "ok-GOLDMARK"])})
        p = softmax([LOW, HIGH], TEMPS["score"])
        ans = decision.answers["q"]
        self.assertEqual(ans["legend"], {"0": "bad", "1": "ok-GOLDMARK"})
        self.assertAlmostEqual(ans["score"], p[1], places=12)

    def test_score_with_ten_levels_and_max_probability_anchor(self):
        levels = ["l%d" % i for i in range(10)]
        levels[7] = "l7-GOLDMARK"
        decision = self.d.decide("s", {"q": dict(RATE, type="score", criteria=levels)})
        p = softmax([LOW] * 7 + [HIGH] + [LOW] * 2, TEMPS["score"])
        ans = decision.answers["q"]
        self.assertAlmostEqual(ans["probabilities"]["7"], p[7], places=12)
        self.assertAlmostEqual(ans["score"], sum(i * v for i, v in enumerate(p)), places=12)
        self.assertAlmostEqual(ans["confidence"], score_confidence(p), places=12)


# ---------------------------------------------------------------- wide lists (knock-out)

class WideListKnockout(BridgeCase):

    def test_26_options_without_allow_none_stay_in_one_pass(self):
        d = self.make(none_check=False)
        crit = {f"o{i:02d}": ("GOLDMARK" if i == 26 else None) for i in range(1, 27)}
        decision = d.decide("s", {"q": dict(PICK, criteria=crit)})
        p = softmax([LOW] * 25 + [HIGH], TEMPS["choice"])
        ans = decision.answers["q"]
        self.assertEqual(ans["choice"], "o26")
        self.assertAlmostEqual(ans["probabilities"]["o26"], p[25], places=12)
        self.assertAlmostEqual(ans["probabilities"]["o01"], p[0], places=12)
        self.assertEqual(len(self.backend.calls), 1)             # no knock-out at exactly 26
        self.assertEqual(self.backend.calls[0][1], tuple(LETTERS[:26]))

    def test_wide_list_of_27_is_knocked_out_group_by_group(self):
        crit = {f"o{i:02d}": ("GOLDMARK" if i == 27 else None) for i in range(1, 28)}
        decision = self.d.decide("s", {"q": dict(PICK, criteria=crit)})
        S = math.exp(HIGH / TEMPS["choice"])
        # first round: near-equal groups of at most 25 -> [o01..o14], [o15..o27]
        gold_g2, rest_g2 = S / (S + 12), 1 / (S + 12)
        # keep 3 per group, ties to the first listed: [o01, o02, o03] and [o27, o15, o16]
        finalists = ["o01", "o02", "o03", "o27", "o15", "o16"]
        first = {f"o{i:02d}": (1 / 14 if i <= 14 else (gold_g2 if i == 27 else rest_g2))
                 for i in range(1, 28)}
        # final round: the finalists plus the extra option (none_check default) = 7 options
        final = softmax([LOW] * 3 + [HIGH] + [LOW] * 3, TEMPS["choice"])
        listed, none, abstain = split_none(final)
        rest = [f"o{i:02d}" for i in range(1, 28) if f"o{i:02d}" not in finalists]
        mass = sum(first[c] for c in rest)
        expected = {c: v * (1 - WIDE["residual"]) for c, v in zip(finalists, listed)}
        expected.update({c: WIDE["residual"] * first[c] / mass for c in rest})
        ans = decision.answers["q"]
        self.assertEqual(ans["choice"], "o27")
        self.assertFalse(abstain)                                # none_check alone: choice stays listed
        self.assertAlmostEqual(ans["none_probability"], none, places=12)
        for k, v in expected.items():
            self.assertAlmostEqual(ans["probabilities"][k], v, places=12)
        self.assertAlmostEqual(sum(ans["probabilities"].values()), 1.0, places=12)
        self.assertAlmostEqual(ans["probabilities"]["o27"], 0.999 * S / (S + 5), places=12)
        self.assertAlmostEqual(ans["confidence"], choice_confidence(list(expected.values())), places=12)
        # read-out rounds: two group passes then the final pass, in group order
        self.assertEqual([len(c[1]) for c in self.backend.calls], [14, 13, 7])
        self.assertIn("D) o27: GOLDMARK", self.backend.calls[2][0])   # finalists keep group-order letters
        # usage counts every pass the question ran
        self.assertEqual(decision.usage["input_tokens"],
                         sum(tokens(self.tok, c[0]) for c in self.backend.calls))
        self.assertEqual(decision.usage["output_tokens"], 0)

    def test_26_options_with_allow_none_are_knocked_out_for_the_none_letter(self):
        crit = {f"o{i:02d}": ("GOLDMARK" if i == 26 else None) for i in range(1, 27)}
        decision = self.d.decide("s", {"q": dict(PICK, criteria=crit)}, body={"allow_none": True})
        S = math.exp(HIGH / TEMPS["choice"])
        gold_g2, rest_g2 = S / (S + 12), 1 / (S + 12)            # groups [o01..o13], [o14..o26]
        finalists = ["o01", "o02", "o03", "o26", "o14", "o15"]
        first = {f"o{i:02d}": (1 / 13 if i <= 13 else (gold_g2 if i == 26 else rest_g2))
                 for i in range(1, 27)}
        final = softmax([LOW] * 3 + [HIGH] + [LOW] * 3, TEMPS["choice"])
        listed, none, abstain = split_none(final)
        rest = [f"o{i:02d}" for i in range(1, 27) if f"o{i:02d}" not in finalists]
        mass = sum(first[c] for c in rest)
        expected = {c: v * (1 - WIDE["residual"]) for c, v in zip(finalists, listed)}
        expected.update({c: WIDE["residual"] * first[c] / mass for c in rest})
        ans = decision.answers["q"]
        self.assertEqual(ans["choice"], "o26")
        self.assertFalse(abstain)                                # the gold option beats the extra one
        self.assertAlmostEqual(ans["none_probability"], none, places=12)
        for k, v in expected.items():
            self.assertAlmostEqual(ans["probabilities"][k], v, places=12)
        self.assertEqual([len(c[1]) for c in self.backend.calls], [13, 13, 7])
        # the extra option lives in the final round, not in the group rounds
        self.assertNotIn("None of the above", self.backend.calls[0][0])
        self.assertIn("G) None of the above", self.backend.calls[2][0])

    def test_wide_list_without_none_check_reports_no_none_probability(self):
        d = self.make(none_check=False)
        crit = {f"o{i:02d}": ("GOLDMARK" if i == 27 else None) for i in range(1, 28)}
        decision = d.decide("s", {"q": dict(PICK, criteria=crit)})
        S = math.exp(HIGH / TEMPS["choice"])
        gold_g2, rest_g2 = S / (S + 12), 1 / (S + 12)
        finalists = ["o01", "o02", "o03", "o27", "o15", "o16"]
        first = {f"o{i:02d}": (1 / 14 if i <= 14 else (gold_g2 if i == 27 else rest_g2))
                 for i in range(1, 28)}
        final = softmax([LOW] * 3 + [HIGH] + [LOW] * 2, TEMPS["choice"])   # 6 options, no extra
        rest = [f"o{i:02d}" for i in range(1, 28) if f"o{i:02d}" not in finalists]
        mass = sum(first[c] for c in rest)
        expected = {c: v * (1 - WIDE["residual"]) for c, v in zip(finalists, final)}
        expected.update({c: WIDE["residual"] * first[c] / mass for c in rest})
        ans = decision.answers["q"]
        self.assertEqual(ans["choice"], "o27")
        self.assertNotIn("none_probability", ans)
        for k, v in expected.items():
            self.assertAlmostEqual(ans["probabilities"][k], v, places=12)
        self.assertEqual([len(c[1]) for c in self.backend.calls], [14, 13, 6])

    def test_knockout_settings_come_from_decision_config(self):
        # a folder with wide_choice {group: 10, keep: 1, residual: 0.5}: a wide list of
        # 27 options is knocked out into three groups of 9 (not the default two groups
        # of 14/13), keeping 1 finalist per group, and the residual share is 0.5.
        folder = make_folder(self._tmp.name + "b", wide={"group": 10, "keep": 1, "residual": 0.5})
        backend = ControlledBackend(self.tok)
        d = bridge.StartluxDecision(folder, self.tok, backend=backend, none_check=False)
        crit = {f"o{i:02d}": ("GOLDMARK" if i == 27 else None) for i in range(1, 28)}
        decision = d.decide("s", {"q": dict(PICK, criteria=crit)})
        S = math.exp(HIGH / TEMPS["choice"])
        first = {**{f"o{i:02d}": 1 / 9 for i in range(1, 19)},            # groups 1 and 2: uniform
                 **{f"o{i:02d}": (S / (S + 8) if i == 27 else 1 / (S + 8)) for i in range(19, 28)}}
        finalists = ["o01", "o10", "o27"]                          # keep 1 of each group
        final = softmax([LOW, LOW, HIGH], TEMPS["choice"])
        rest = [f"o{i:02d}" for i in range(1, 28) if f"o{i:02d}" not in finalists]
        mass = sum(first[c] for c in rest)
        expected = {c: v * 0.5 for c, v in zip(finalists, final)}
        expected.update({c: 0.5 * first[c] / mass for c in rest})
        ans = decision.answers["q"]
        self.assertEqual(ans["choice"], "o27")
        for k, v in expected.items():
            self.assertAlmostEqual(ans["probabilities"][k], v, places=12)
        self.assertEqual([len(c[1]) for c in backend.calls], [9, 9, 9, 3])


# ---------------------------------------------------------------- Instant controls / refusal

class InstantControlsAndRefusal(BridgeCase):

    def test_think_mode_auto_and_always_are_refused_for_instant(self):
        for mode in ("always", "auto", "hybrid"):                # hybrid is the older name of auto
            with self.assertRaises(RequestError) as e:
                self.d.decide("s", {"q": {"type": "noul"}}, body={"think_mode": mode})
            self.assertIn("Instant", str(e.exception))
            self.assertEqual(e.exception.loc, ["body", "think_mode"])

    def test_think_mode_never_and_plain_are_accepted(self):
        for mode in ("never", "plain"):
            decision = self.d.decide("s", {"q": {"type": "noul"}}, body={"think_mode": mode})
            self.assertIn("q", decision.answers)

    def test_stream_thought_generation_is_refused_on_a_non_streaming_backend(self):
        with self.assertRaises(RequestError) as e:
            self.d.decide("s", {"q": {"type": "noul"}}, body={"thought_generation": "stream"})
        self.assertEqual(e.exception.loc, ["body", "thought_generation"])

    def test_control_field_validation(self):
        for body in ({"gate": 1.5}, {"mix": -0.1}, {"think_budget": 0},
                     {"allow_none": "yes"}, {"none_check": 1}, {"return_details": "no"}):
            with self.assertRaises(RequestError):
                self.d.decide("s", {"q": {"type": "noul"}}, body=body)

    def test_details_report_the_first_pass_only_for_instant(self):
        decision = self.d.decide("s", {"q": dict(PICK, criteria={"x": "GOLDMARK", "y": "d"})},
                                 body={"return_details": True})
        det = decision.answers["q"]["details"]
        p = softmax([HIGH, LOW, LOW], TEMPS["choice"])           # 2 listed + the extra option
        self.assertAlmostEqual(det["first_pass"]["top"], max(p), places=12)
        self.assertEqual(list(det["first_pass"]["probabilities"]), ["x", "y", "None of the above"])
        self.assertFalse(det["thought"]["used"])
        self.assertEqual(det["thought"]["ended"], "skipped")
        self.assertIsNone(det["after_thought"])
        self.assertIsNone(det["gate"])                            # think_mode never: no gate
        self.assertIsNone(det["mix"])
        self.assertIn("wall", decision.timing_ms)


# ---------------------------------------------------------------- request errors

class RequestErrors(BridgeCase):

    def assert_question_error(self, questions, needle, key="q", loc=None):
        with self.assertRaises(RequestError) as e:
            self.d.decide("s", questions)
        self.assertIn(needle, str(e.exception))
        self.assertEqual(e.exception.loc, loc or ["body", "questions", key])

    def test_questions_must_be_a_non_empty_object(self):
        for bad in ({}, [], "q", None):
            with self.assertRaises(RequestError) as e:
                self.d.decide("s", bad)
            self.assertIn("questions must be a non-empty object", str(e.exception))

    def test_question_must_be_an_object(self):
        self.assert_question_error({"q": "not a dict"}, "a question must be an object")

    def test_unknown_question_type(self):
        self.assert_question_error({"q": {"type": "rank"}}, "unknown question type")

    def test_choice_needs_criteria(self):
        self.assert_question_error({"q": {"instructions": "pick"}}, "a choice question needs criteria")

    def test_choice_needs_instructions(self):
        self.assert_question_error({"q": {"criteria": {"a": None, "b": None}}},
                                   "instructions are required")

    def test_choice_with_no_options(self):
        d = self.make(none_check=False)               # with none_check on, {} gets the extra option: got 1
        with self.assertRaises(RequestError) as e:
            d.decide("s", {"q": dict(PICK, criteria={})})
        self.assertIn("got 0", str(e.exception))
        self.assertEqual(e.exception.loc, ["body", "questions", "q"])

    def test_score_level_bounds(self):
        self.assert_question_error({"q": dict(RATE, type="score", criteria=["a"] * 11)},
                                   "2 to 10 levels, got 11")
        self.assert_question_error({"q": dict(RATE, type="score", criteria=["a"] * 27)},
                                   "got 27")

    def test_bad_option_ids(self):
        self.assert_question_error({"q": dict(PICK, criteria={"a": None, "x" * 201: None})},
                                   "at most 200 characters")
        self.assert_question_error({"q": dict(PICK, criteria={"a": None, "b\nc": None})},
                                   "single-line strings")

    def test_allow_none_option_name_clash(self):
        with self.assertRaises(RequestError) as e:
            self.d.decide("s", {"q": dict(PICK, criteria={"None Of The Above": "x", "b": "y"})},
                          body={"allow_none": True})
        self.assertIn("allow_none adds the option", str(e.exception))

    def test_question_level_allow_none_must_be_boolean(self):
        self.assert_question_error({"q": dict(PICK, allow_none="maybe",
                                              criteria={"a": None, "b": None})},
                                   "allow_none must be true or false",
                                   loc=["body", "questions", "q", "allow_none"])


# ---------------------------------------------------------------- request / question / read-out alignment

class RequestQuestionReadoutAlignment(BridgeCase):

    def test_three_questions_align_answers_letters_and_usage(self):
        questions = {
            "c": dict(PICK, criteria={"x": "GOLDMARK", "y": "d", "z": "d"}),  # + extra option: 4 letters
            "n": {"type": "noul", "criteria": {"true": "GOLDMARK", "false": "d"}},
            "s": dict(RATE, type="score", criteria=["low", "GOLDMARK", "high"]),
        }
        decision = self.d.decide("evidence", questions)
        self.assertEqual(set(decision.answers), {"c", "n", "s"})
        self.assertEqual(decision.answers["c"]["type"], "choice")
        self.assertEqual(decision.answers["n"]["type"], "noul")
        self.assertEqual(decision.answers["s"]["type"], "score")
        self.assertEqual([len(c[1]) for c in self.backend.calls], [4, 2, 3])
        for (prompt, letters), k in zip(self.backend.calls, (4, 2, 3)):
            self.assertEqual(letters, tuple(LETTERS[:k]))                  # config letters, in order
            self.assertTrue(prompt.endswith(THINK_OFF_SUFFIX))
        self.assertEqual(decision.usage["input_tokens"],
                         sum(tokens(self.tok, c[0]) for c in self.backend.calls))
        self.assertEqual(decision.usage["output_tokens"], 0)               # Instant never generates

    def test_requests_do_not_leak_into_each_other(self):
        first = self.d.decide("state one", {"q": dict(PICK, criteria={"a": "GOLDMARK", "b": "d"})})
        second = self.d.decide("state two", {"q": dict(PICK, criteria={"a": "d", "b": "GOLDMARK"})})
        self.assertEqual(first.answers["q"]["choice"], "a")
        self.assertEqual(second.answers["q"]["choice"], "b")
        self.assertEqual(len(self.backend.calls), 2)                       # one read-out per request
        self.assertNotIn("state one", self.backend.calls[1][0])
        self.assertIn("state two", self.backend.calls[1][0])
        # a repeated request is answered identically: no state carries over
        again = self.d.decide("state one", {"q": dict(PICK, criteria={"a": "GOLDMARK", "b": "d"})})
        self.assertEqual(again.answers, first.answers)


# ---------------------------------------------------------------- prompt contract

class PromptContract(BridgeCase):

    def test_one_pass_prompt_layout(self):
        d = self.make(none_check=False)
        d.decide("some evidence", {"q": dict(PICK, criteria={"aa": "one", "bb": "two"})})
        prompt = self.backend.calls[0][0]
        self.assertTrue(prompt.endswith(THINK_OFF_SUFFIX))                 # thinking switched off
        self.assertTrue(prompt.startswith(
            "<|im_start|>system\nApply the criterion to the evidence. Choose exactly one "
            "listed option. Answer with its letter only.<|im_end|>\n<|im_start|>user\nEvidence:\n"))
        self.assertIn("\n\nQuestion: Pick one.\nOptions:\nA) aa: one\nB) bb: two<|im_end|>\n"
                      "<|im_start|>assistant\n", prompt)

    def test_bare_ids_are_hidden_when_every_option_has_a_description(self):
        d = self.make(none_check=False)
        d.decide("s", {"q": dict(PICK, criteria={"a": "one", "b": "two"})})
        prompt = self.backend.calls[0][0]
        self.assertIn("A) one", prompt)                      # the bare id is not shown
        self.assertNotIn("A) a:", prompt)

    def test_empty_state_renders_as_none(self):
        d = self.make(none_check=False)
        d.decide("", {"q": dict(PICK, criteria={"aa": None, "bb": None})})
        self.assertIn("Evidence:\n(none)\n\nQuestion: Pick one.", self.backend.calls[0][0])

    def test_long_list_states_carry_indices(self):
        d = self.make(none_check=False)
        d.decide(["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"],
                 {"q": dict(PICK, criteria={"aa": None, "bb": None})})
        self.assertIn('"_index": 7', self.backend.calls[0][0])             # 8 elements >= 8 get indexed


# ---------------------------------------------------------------- the read-out callback seam

class ReadoutCallbackBackendBehavior(BridgeCase):

    def test_sync_callback_drives_the_decider(self):
        seen = []

        def readout(prompt, letter_ids):
            seen.append((prompt, tuple(letter_ids)))
            return [LOW] * len(letter_ids)                     # uniform: ties -> first listed

        d = bridge.StartluxDecision(self.folder, self.tok, readout=readout, none_check=False)
        decision = d.decide("s", {"q": dict(PICK, criteria={"x": "d1", "y": "d2"})})
        self.assertEqual(decision.answers["q"]["choice"], "x")
        self.assertEqual(len(seen), 1)                         # the callback saw every call
        self.assertEqual(seen[0][1], tuple(LETTERS[:2]))

    def test_async_callback_drives_the_decider(self):
        async def readout(prompt, letter_ids):
            return [HIGH, LOW][:len(letter_ids)]

        d = bridge.StartluxDecision(self.folder, self.tok, readout=readout, none_check=False)
        decision = d.decide("s", {"q": dict(PICK, criteria={"x": "d1", "y": "d2"})})
        self.assertEqual(decision.answers["q"]["choice"], "x")

    def test_wrong_length_callback_is_a_refused_question(self):
        def readout(prompt, letter_ids):
            return [0.0, 0.0]                                   # always two, whatever was asked

        d = bridge.StartluxDecision(self.folder, self.tok, readout=readout, none_check=False)
        with self.assertRaises(RequestError) as e:
            d.decide("s", {"q": dict(PICK, criteria={"x": "d1", "y": "d2", "z": "d3"})})
        self.assertIn("refused this question", str(e.exception))

    def test_decide_async_entry_point(self):
        async def readout(prompt, letter_ids):
            return [LOW] * len(letter_ids)

        d = bridge.StartluxDecision(self.folder, self.tok, readout=readout, none_check=False)
        decision = asyncio.run(d.decide_async("s", {"q": dict(PICK, criteria={"x": "d1", "y": "d2"})}))
        self.assertEqual(set(decision.answers), {"q"})


# ---------------------------------------------------------------- real pinned model materials

REAL_MODEL_DIR = (Path(__file__).resolve().parents[1] / "work" / "startlux-materials"
                  / "model-6496ae641cc71b149ce9dad69bd97f2a44e7bf11")
REAL_TEMPS = {"choice": 1.4364, "noul": 1.0411, "score": 1.4021}
REAL_LETTERS = list(range(32, 58))


@unittest.skipUnless(REAL_MODEL_DIR.is_dir(), "the pinned real model materials are not present")
class RealTokenizerIntegration(unittest.TestCase):
    """The pinned real tokenizer + real decision_config.json through the bridge, on the
    controlled marker read-out backend: the real chat template (thinking-off suffix),
    the real letter token ids (32..57), the real per-type temperatures and the official
    orchestration (None handling, wide-list knock-out).  Model-weight inference is NOT
    exercised here: the independent CPU FP32 oracle for that is reference.py."""

    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer
        cls.tok = AutoTokenizer.from_pretrained(REAL_MODEL_DIR, local_files_only=True)

    def setUp(self):
        self.backend = ControlledBackend(self.tok)
        self.d = bridge.StartluxDecision(REAL_MODEL_DIR, self.tok, backend=self.backend)

    def make(self, **kw):
        return bridge.StartluxDecision(REAL_MODEL_DIR, self.tok, backend=self.backend, **kw)

    def test_real_folder_and_letter_ids(self):
        f = self.d.folder
        self.assertEqual(f.kind, "instant")
        self.assertFalse(f.v1)                                     # format lodestar-letter-readout-v1
        self.assertIsNone(f.auto_think)
        self.assertEqual(dict(f.temperature_by_type), REAL_TEMPS)
        self.assertEqual(list(f.letter_token_ids), REAL_LETTERS)
        self.assertEqual(self.d.letter_ids, tuple(REAL_LETTERS))
        cfg = json.loads((REAL_MODEL_DIR / "decision_config.json").read_text())
        self.assertEqual(cfg["letter_token_ids"], REAL_LETTERS)
        self.assertEqual((f.wide_group, f.wide_keep, f.wide_residual), (25, 3, 0.001))

    def test_letters_are_single_tokens_right_after_the_real_prompt(self):
        self.d.decide("evidence text", {"q": dict(PICK, criteria={"aa": "x", "bb": "y"})},
                      body={"none_check": False})
        prompt = self.backend.calls[0][0]
        ids = self.tok.encode(prompt, add_special_tokens=False)
        for letter_id, letter in zip(REAL_LETTERS[:2], "AB"):
            t = self.tok.encode(letter, add_special_tokens=False)
            self.assertEqual(len(t), 1)
            self.assertEqual(t[0], letter_id)
            self.assertEqual(self.tok.encode(prompt + letter, add_special_tokens=False), ids + t)

    def test_real_choice_uses_the_real_temperature(self):
        d = self.make(none_check=False)
        decision = d.decide("evidence text", {"q": dict(PICK, criteria={
            "aa": "plain", "bb": "has GOLDMARK", "cc": "plain"})})
        p = softmax([LOW, HIGH, LOW], REAL_TEMPS["choice"])
        ans = decision.answers["q"]
        self.assertEqual(ans["choice"], "bb")
        self.assertAlmostEqual(ans["probabilities"]["bb"], p[1], places=12)
        self.assertAlmostEqual(ans["probabilities"]["aa"], p[0], places=12)
        self.assertAlmostEqual(ans["confidence"], choice_confidence(p), places=12)
        prompt, letters = self.backend.calls[0]
        self.assertEqual(letters, tuple(REAL_LETTERS[:3]))
        self.assertTrue(prompt.endswith(THINK_OFF_SUFFIX))
        self.assertTrue(prompt.startswith("<|im_start|>system\nApply the criterion to the evidence."))
        self.assertIn("A) aa: plain", prompt)

    def test_real_noul_and_score_at_their_real_temperatures(self):
        decision = self.d.decide("evidence text", {
            "n": {"type": "noul", "criteria": {"true": "GOLDMARK", "false": "no"}},
            "s": dict(RATE, type="score", criteria=["low", "mid", "GOLDMARK"])})
        p_true = softmax([HIGH, LOW], REAL_TEMPS["noul"])[0]
        self.assertAlmostEqual(decision.answers["n"]["noul"], p_true, places=12)
        p = softmax([LOW, LOW, HIGH], REAL_TEMPS["score"])
        ans = decision.answers["s"]
        self.assertAlmostEqual(ans["score"], sum(i * v for i, v in enumerate(p)), places=12)
        self.assertAlmostEqual(ans["confidence"], score_confidence(p), places=12)
        self.assertEqual([len(c[1]) for c in self.backend.calls], [2, 3])
        for prompt, letters in self.backend.calls:
            self.assertEqual(letters, tuple(REAL_LETTERS[:len(letters)]))
            self.assertTrue(prompt.endswith(THINK_OFF_SUFFIX))

    def test_real_none_check_default_and_allow_none(self):
        decision = self.d.decide("evidence text", {"q": dict(PICK, criteria={"aa": "x", "bb": "y"})})
        p = softmax([LOW, LOW, LOW], REAL_TEMPS["choice"])          # 2 listed + the extra option
        listed, none, _ = split_none(p)
        ans = decision.answers["q"]
        self.assertEqual(ans["choice"], "aa")
        self.assertAlmostEqual(ans["none_probability"], none, places=12)
        self.assertAlmostEqual(ans["probabilities"]["aa"], listed[0], places=12)
        self.assertEqual(len(self.backend.calls[0][1]), 3)          # none_check adds a letter
        self.backend.gold_last = True
        decision = self.d.decide("evidence text", {"q": dict(PICK, criteria={"aa": "x", "bb": "y"})},
                                 body={"allow_none": True})
        p = softmax([LOW, LOW, HIGH], REAL_TEMPS["choice"])
        self.assertIsNone(decision.answers["q"]["choice"])
        self.assertAlmostEqual(decision.answers["q"]["none_probability"], p[2], places=12)

    def test_real_wide_list_knockout_orchestration(self):
        crit = {f"o{i:02d}": ("GOLDMARK" if i == 27 else None) for i in range(1, 28)}
        state = "evidence text"
        spec = dict(PICK, criteria=crit)
        decision = self.d.decide(state, {"q": spec})
        S = math.exp(HIGH / REAL_TEMPS["choice"])
        gold_g2, rest_g2 = S / (S + 12), 1 / (S + 12)
        finalists = ["o01", "o02", "o03", "o27", "o15", "o16"]
        first = {f"o{i:02d}": (1 / 14 if i <= 14 else (gold_g2 if i == 27 else rest_g2))
                 for i in range(1, 28)}
        final = softmax([LOW] * 3 + [HIGH] + [LOW] * 3, REAL_TEMPS["choice"])
        listed, none, abstain = split_none(final)
        rest = [f"o{i:02d}" for i in range(1, 28) if f"o{i:02d}" not in finalists]
        mass = sum(first[c] for c in rest)
        expected = {c: v * (1 - 0.001) for c, v in zip(finalists, listed)}
        expected.update({c: 0.001 * first[c] / mass for c in rest})
        ans = decision.answers["q"]
        self.assertEqual(ans["choice"], "o27")
        self.assertFalse(abstain)
        self.assertAlmostEqual(ans["none_probability"], none, places=12)
        self.assertAlmostEqual(sum(ans["probabilities"].values()), 1.0, places=12)
        self.assertAlmostEqual(ans["probabilities"]["o27"], 0.999 * listed[3], places=12)
        self.assertAlmostEqual(ans["probabilities"]["o04"], 0.001 * (1 / 14) / mass, places=12)
        self.assertAlmostEqual(ans["confidence"], choice_confidence(list(expected.values())), places=12)
        # the group prompts the bridge sent are the ones the official renderer would send
        from startlux_decide.prompt import from_systemone, messages, plain_prompt
        from startlux_decide.readout import knockout_groups
        groups = knockout_groups(list(crit), 25)
        self.assertEqual([len(c[1]) for c in self.backend.calls], [len(g) for g in groups] + [7])
        for (prompt, _), g in zip(self.backend.calls, groups):
            row = from_systemone(state, dict(spec, criteria={c: crit[c] for c in g}))
            self.assertEqual(prompt, plain_prompt(self.tok, messages(row)))

    def test_real_readout_callback_receives_real_prompts(self):
        seen = []

        def readout(prompt, letter_ids):
            seen.append((prompt, tuple(letter_ids)))
            return [LOW] * len(letter_ids)

        d = bridge.StartluxDecision(REAL_MODEL_DIR, self.tok, readout=readout, none_check=False)
        decision = d.decide("evidence text", {"q": dict(PICK, criteria={"aa": "x", "bb": "y"})})
        self.assertEqual(decision.answers["q"]["choice"], "aa")     # uniform: first listed
        self.assertEqual(len(seen), 1)
        self.assertTrue(seen[0][0].endswith(THINK_OFF_SUFFIX))
        self.assertEqual(seen[0][1], tuple(REAL_LETTERS[:2]))


_reference_spec = importlib.util.spec_from_file_location(
    "startlux_decision_reference_review", MODEL_DIR / "decider" / "reference.py")
reference = importlib.util.module_from_spec(_reference_spec)
_reference_spec.loader.exec_module(reference)

PINNED_SHARD = "model-00001-of-00001.safetensors"
PINNED_BYTES = 1746942632
PINNED_SHA256 = "f788c8e01fd453dfbf2effb23a06148d38238b9e8a872ab8faacbb49a03cecd1"


class ReferenceReviewFixes(unittest.TestCase):
    """The P2b main-agent review fixes: offline loading flags, and a weight identity
    that refuses anything but the pinned shard (size + SHA256), never reporting an
    arbitrary folder -- or a weightless one -- as the pinned revision."""

    def test_expected_weights_pin_the_reviewed_shard(self):
        self.assertEqual(set(reference.EXPECTED_WEIGHTS), {PINNED_SHARD})
        self.assertEqual(reference.EXPECTED_WEIGHTS[PINNED_SHARD],
                         {"bytes": PINNED_BYTES, "sha256": PINNED_SHA256})

    def test_backends_default_to_fp32_and_the_bf16_variant_declares_itself(self):
        fp32 = reference.GoldenTransformersBackend("/nowhere", None, list(range(32, 58)))
        self.assertEqual(fp32.dtype_name, "float32")
        self.assertEqual(fp32.name, "cpu-fp32-golden")
        bf16 = reference.BF16GoldenTransformersBackend("/nowhere", None, list(range(32, 58)))
        self.assertEqual(bf16.dtype_name, "bfloat16")
        self.assertEqual(bf16.name, "cpu-bf16-golden")
        self.assertEqual(bf16.gdr_precision, {})

    def test_weight_identity_refuses_a_folder_without_weights(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):
                reference.weight_identity(d)

    def test_weight_identity_refuses_shards_that_are_not_the_pinned_set(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "someone-elses-00001-of-00002.safetensors").write_bytes(b"x")
            with self.assertRaises(ValueError):
                reference.weight_identity(d)
        with tempfile.TemporaryDirectory() as d:
            Path(d, PINNED_SHARD).write_bytes(b"x")
            Path(d, "extra-00001-of-00002.safetensors").write_bytes(b"x")
            with self.assertRaises(ValueError):
                reference.weight_identity(d)

    def test_weight_identity_refuses_the_pinned_name_with_other_bytes(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, PINNED_SHARD).write_bytes(b"not the pinned weights")
            with self.assertRaises(ValueError) as ctx:
                reference.weight_identity(d)
            self.assertIn("pinned", str(ctx.exception))

    @unittest.skipUnless(REAL_MODEL_DIR.is_dir(), "the pinned real model materials are not present")
    def test_real_folder_weight_identity_verifies_the_pinned_shard(self):
        identity = reference.weight_identity(REAL_MODEL_DIR)
        self.assertEqual(identity["revision"], reference.MODEL_REVISION)
        files = identity["files"][PINNED_SHARD]
        self.assertEqual((files["bytes"], files["sha256"]), (PINNED_BYTES, PINNED_SHA256))
        self.assertTrue(files["verified"])


if __name__ == "__main__":
    unittest.main()
