# -*- coding: utf-8 -*-
"""Wake-word detection, ported from sadie-voice/wakeword.py.

Uses vosk in *grammar* mode: it only ever recognizes the words in
WAKE_ALIASES (+ "[unk]"), not open-vocabulary speech. That means adding
or renaming wake phrases is a pure config.py change -- no model
retraining needed (confirmed while investigating the Jarvis naming
question, see /Users/apple/Projects/jarvis-project/tasks/TASK-0001.md).

listen(callback) blocks until a wake word is heard.
"""
import json
import queue
import sys
import threading

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import sounddevice as sd
from vosk import Model, KaldiRecognizer, SetLogLevel

from .config import WAKE_ALIASES

SetLogLevel(-1)

RATE = 16000
CHUNK_MS = 60
FRAMES_PER_CHUNK = int(RATE * CHUNK_MS / 1000)

WAKE_GRAMMAR = json.dumps(list(WAKE_ALIASES) + ["[unk]"], ensure_ascii=False)


def _boost(indata):
    """Auto-gain: leave quiet background noise alone (RMS < 30), boost
    real speech."""
    arr = np.frombuffer(indata, dtype=np.int16).astype(np.float32)
    rms = float(np.sqrt(np.mean(arr * arr)))
    if rms < 30.0:
        return bytes(indata)
    gain = min(60.0, 6000.0 / rms)
    amp = np.clip(arr * gain, -32767, 32767).astype(np.int16)
    return amp.tobytes()


class WakewordListener:
    def __init__(self, model_path, device=None):
        self.model = Model(str(model_path))
        self.device = device
        self._stop = threading.Event()

    def _new_recognizer(self):
        return KaldiRecognizer(self.model, RATE, WAKE_GRAMMAR)

    def _match(self, text):
        t = (text or "").strip().lower()
        for w in WAKE_ALIASES:
            if w in t:
                return w
        return None

    def listen(self, on_wake=None, device=None):
        """Block until a wake word hits. Returns the matched alias, or
        None if stop() was called first."""
        self._stop.clear()
        rec = self._new_recognizer()
        q = queue.Queue()
        dev = device if device is not None else self.device

        def cb(indata, frames, time_info, status):
            q.put(_boost(bytes(indata)))

        with sd.RawInputStream(
            samplerate=RATE,
            blocksize=FRAMES_PER_CHUNK,
            dtype="int16",
            channels=1,
            device=dev,
            callback=cb,
        ):
            while not self._stop.is_set():
                try:
                    data = q.get(timeout=0.2)
                except queue.Empty:
                    continue
                if rec.AcceptWaveform(data):
                    hit = self._match(json.loads(rec.Result()).get("text", ""))
                else:
                    hit = self._match(json.loads(rec.PartialResult()).get("partial", ""))
                if hit:
                    if on_wake:
                        on_wake(hit)
                    return hit
        return None

    def stop(self):
        self._stop.set()
