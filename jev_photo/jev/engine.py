"""Orchestrate visual typed judgments on one CUDA device."""
import threading
import time

from .grounding import locate
from .numeric import decode_numbers
from .preprocessing import Prompter, read_image
from .schema import Question, Request, answer
from .scoring import Session


class ImageSession:
    """One image + context, prefilled once; judge() may be called repeatedly."""

    def __init__(self, engine, image, state, mode="shared", temperature=1.0):
        self.engine, self.temperature = engine, temperature
        self.prompter = Prompter(engine.processor, state)
        self.session = Session(engine.adapter, self.prompter.prefix, image, mode, engine.batch_size)

    def judge(self, questions: dict):
        questions = {k: q if isinstance(q, Question) else Question.model_validate(q) for k, q in questions.items()}
        simple = {k: q for k, q in questions.items() if q.type not in ("number", "box")}
        numbers = {k: q for k, q in questions.items() if q.type == "number"}
        boxes = {k: q.instructions for k, q in questions.items() if q.type == "box"}
        answers, rounds = {}, 0
        if simple:
            plans = [self.prompter.plan(q) for q in simple.values()]
            scores = self.session.score(plans)
            for (key, question), values, plan in zip(simple.items(), scores, plans):
                result = answer(question, values, self.temperature)
                result.update({
                    "scoring": plan.scoring, "candidate_scores": values,
                    "score_kind": "sequence_log_probability_including_eos" if plan.scoring == "sequence"
                    else "candidate_logit",
                    "candidate_token_ids": plan.targets, "prompt_sha256": plan.prompt_sha256})
                answers[key] = result
        if numbers:
            decoded, rounds = decode_numbers(numbers, self.prompter, self.session, self.temperature)
            answers.update(decoded)
        if boxes:
            located, steps = locate(self.prompter, self.session, boxes)
            answers.update({k: {"type": "box", **v} for k, v in located.items()})
            rounds += steps
        return {key: answers[key] for key in questions}, rounds


class Engine:
    def __init__(self, model_path=None, device="cuda:0", batch_size=16, *, adapter=None, dtype="bfloat16"):
        if not 1 <= batch_size <= 64:
            raise ValueError("batch_size must be 1..64")
        if adapter is None:
            from .adapter_torch import load_adapter
            self.adapter, self.model_source = load_adapter(model_path, device, dtype)
        else:
            self.adapter, self.model_source = adapter, "injected adapter"
        self.device, self.batch_size = device, batch_size
        self.processor, self.tokenizer = self.adapter.processor, self.adapter.tokenizer
        self.lock = threading.Lock()

    def session(self, image, state="", mode="shared", temperature=1.0):
        """Caller must hold self.lock for the lifetime of the session."""
        return ImageSession(self, image, state, mode, temperature)

    def judge(self, request: Request, *, allow_path=True):
        with self.lock:
            self.adapter.reset()
            started = time.perf_counter()
            image = read_image(request.image, allow_path=allow_path)
            session = self.session(image, request.state, request.mode, request.temperature)
            answers, rounds = session.judge(request.questions)
            return {"model": self.model_source, "answers": answers,
                    "probability_semantics": "normalized candidate probability conditional on supplied candidates; "
                                             "not calibrated",
                    "metrics": self.metrics(started, session, len(request.questions), rounds)}

    def metrics(self, started, session, decisions, number_rounds):
        elapsed = (time.perf_counter() - started) * 1000
        return {**self.adapter.timings.dict(), "mode": session.session.mode, "device": self.device,
                "elapsed_ms": elapsed, "prefix_tokens": session.session.prefix_tokens,
                "number_rounds": number_rounds, "generated_tokens": 0,
                "peak_gpu_memory_gb": self.adapter.peak_memory_gb(),
                "decisions_per_second": decisions * 1000 / elapsed}
