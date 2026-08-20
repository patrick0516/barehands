# -*- coding: utf-8 -*-
"""Jarvis's voice conversation core, ported from sadie-voice/sadie_voice.py.

Scope kept from the original: wake word -> Gemini Live real-time voice ->
run_opencode background-task dispatch, plus the reconnect/pause/resume/
idle-timeout machinery around it.

Scope deliberately left out (still lives only in sadie-voice, untouched):
computer_control (desktop automation), memory, calendar, todo/habit/
reminder life-management, the 3-column HUD. See TASK-0005.md.

Board integration: instead of the old two-process design (an HTTP tool
calling back into Jarvis' own /cmd endpoint), this runs *inside* the
same Python process as Jarvis' server.py, so the assistant's
listening/thinking/speaking state is written straight to Jarvis'
state/state file -- no network round-trip to itself.

Voice "thinking" filler (absorbed from TASK-0003): the system prompt asks
the model to say a short spoken transition line before invoking
run_opencode, in the same turn, the way GPT voice mode does it -- this
needs a live test to confirm Gemini Live actually honors "speak, then
call the tool" as one turn; a short ack tone still plays immediately as a
non-negotiable fallback cue either way.
"""
import asyncio
import base64
import datetime
import json
import os
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import urllib.parse
import urllib.request

import numpy as np
import sounddevice as sd
import websockets

from . import config
from .wakeword import WakewordListener

MODEL = config.LIVE_MODEL
VOICE = config.VOICE
MIC_INDEX = config.MIC_INDEX
OUT_DEVICE = config.OUT_DEVICE
PLAY_GAIN = config.PLAY_GAIN
WORKDIR = config.WORKDIR
OPENCODE = config.OPENCODE
WAKE_MODEL = config.WAKE_MODEL
STATE_DIR = config.STATE_DIR

EXIT_PHRASES = ("掰掰", "再見", "結束", "退出", "晚安", "你先忙", "沒事了")
PAUSE_PHRASES = ("暫停", "閉嘴", "安靜", "先停", "別講了", "停下")
RESUME_PHRASES = ("繼續", "回來", "恢復", "接著講", "好了")
IDLE_TIMEOUT = 25.0
MAX_TURNS = 20

BG_LOCK = threading.Lock()  # background opencode tasks are serialized

# ---- manual sleep/wake, driven from stage.html (S key / the sleep
# button) via server.py's /assistant/sleep and /assistant/wake -- see
# _wake_loop_main() for where _SLEEPING is checked and sleep_assistant()/
# wake_assistant() below for how the browser controls it.
_SLEEPING = threading.Event()
ACTIVE_SESSION = [None]  # the live LiveSession, if a conversation is in
                         # progress -- sleep_assistant() ends it early
_LISTENER = [None]       # the live WakewordListener -- sleep_assistant()
                         # calls .stop() on it so a currently-blocking
                         # listen() call returns immediately instead of
                         # only taking effect after the next wake


def is_sleeping():
    return _SLEEPING.is_set()


def sleep_assistant():
    """Stop listening for the wake word, and end any conversation
    that's live right now. The wake-word loop's InputStream only opens
    while actively listening, so sleeping also releases the mic."""
    _SLEEPING.set()
    listener = _LISTENER[0]
    if listener is not None:
        listener.stop()   # interrupts a currently-blocking listen() call
    sess = ACTIVE_SESSION[0]
    if sess is not None:
        sess.exit_evt.set()
    return True


def wake_assistant():
    _SLEEPING.clear()
    return True

SYS = """你是 Sadie——Chou 老大的 JARVIS 式駐守助理。你是冷靜沉穩、聰明俐落、忠誠直接的夥伴，帶一點 JARVIS 式冷幽默但絕不油膩；全程用繁體中文、口語，簡短直接、先講重點。

規則：
1. 喚醒後先簡短說「Sir，我在。」
2. 一般聊天、問答、哈拉 -> 直接語音回答，3 句話以內。
3. 老大要求「動手做事」（記事、查/寫檔案、研究、上網查資料、整理筆記、設定等）-> 呼叫 run_opencode 工具，把任務寫成一句清楚的繁體中文指令；需要查網路資料就直接在指令裡寫清楚要查什麼，opencode 自己會上網。呼叫前先用一句話口頭說一下你要處理了（例如「我來處理一下」「稍等，我看看」），不要憑空沉默——這句話跟呼叫工具算同一輪，講完立刻呼叫，不用等回覆。
4. 工具結果回來後，用 2~3 句口語轉述重點給老大。如果結果是查資料/研究類的內容（不是單純的「已完成」這種動作回報），額外呼叫 show_on_board 把整理過的重點秀在畫面上，老大自己問「幫我查/搜尋/看一下」這類要求時尤其要秀出來，不要只用講的。
5. 老大明確說「秀出來/顯示在畫面上/放到板子上」時，直接呼叫 show_on_board，不用等 run_opencode。
6. 老大要你召喚/顯示一個 3D 模型時 -> 先呼叫 list_models 看有哪些檔案可選，再呼叫 summon_model，file 參數要用 list_models 回傳的確切路徑，不要自己編檔名。如果 list_models 是空的，老實跟老大說目前沒有模型檔可以召喚，不要假裝有。
7. 老大說「掰掰/結束/晚安」等 -> 簡短道別後，不要再多話。
8. 紅線確認（執行前必須先口頭問老大，等明確同意才動手）：會花錢的、會對外發布的、刪除檔案/移除東西、觸及權限/帳號/不可回復的操作。問法要簡短：「這會刪掉 X，確定嗎？」老大說好才做。其他操作直接做，不要每件事都問。"""

TOOLS = [{
    "function_declarations": [{
        "name": "run_opencode",
        "description": "把任務交給桌面端 opencode 實際執行：記事、查/寫檔案、研究、上網搜尋/瀏覽網頁、整理筆記等需要工具的操作。opencode 自己有網路存取能力，直接把要查的東西寫進 task 就好，不需要另外的瀏覽器工具。",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "task": {"type": "STRING",
                         "description": "要執行的任務，寫成一句清楚的繁體中文指令"}
            },
            "required": ["task"]
        }
    }, {
        "name": "show_on_board",
        "description": "把一段內容（通常是查資料/研究的結果）用卡片的形式秀在 Jarvis 的畫面上，飛到畫面中央。老大要求「秀出來」「顯示」「放到板子上」時，或查資料結果值得用眼睛看而不只是用聽的時候呼叫。",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "title": {"type": "STRING", "description": "卡片標題，簡短"},
                "body": {"type": "STRING", "description": "卡片內文，整理過的重點文字"}
            },
            "required": ["title", "body"]
        }
    }, {
        "name": "list_models",
        "description": "列出目前 media 資料夾裡有哪些 3D 模型檔可以召喚到畫面上。想幫老大召喚模型、但不確定確切檔名時，先呼叫這個。",
        "parameters": {"type": "OBJECT", "properties": {}}
    }, {
        "name": "summon_model",
        "description": "把一個 3D 模型檔召喚到畫面正中央，放大展示（鋼鐵人式的全息召喚）。file 必須是 list_models 回傳過的其中一個確切路徑，不要自己編。",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "file": {"type": "STRING", "description": "list_models 回傳的其中一個確切檔案路徑"},
                "title": {"type": "STRING", "description": "顯示用的標題，簡短（選填）"}
            },
            "required": ["file"]
        }
    }]
}]


def _ws_url():
    # A plain Gemini API key authenticates as a `key=` query param, NOT
    # as a Bearer token -- Bearer is for real OAuth access tokens, which
    # this port doesn't use (see config.py's docstring on why the
    # OAuth-refresh flow was dropped). Confirmed live 2026-08-21: Bearer
    # + a plain API key gets rejected with "invalid authentication
    # credentials. Expected OAuth 2 access token..."
    return ("wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta."
            "GenerativeService.BidiGenerateContent?key=" + config.gemini_key())


def _ws_headers():
    return {}


def start_opencode_bg(task):
    """Fire-and-forget opencode dispatch; caller polls job['done'] (via
    bg_monitor). Serialized through BG_LOCK so two `--continue` calls
    never race the same opencode session."""
    if not OPENCODE:
        job = {"proc": None, "out_file": None, "task": task, "done": True,
               "result": "opencode 執行檔找不到（PATH 裡沒有 opencode），無法執行。",
               "timed_out": False, "started": time.time()}
        return job
    today_tag = datetime.datetime.now().strftime("%Y-%m-%d")
    prompt = ("你是 Sadie（JARVIS 式助理），執行老大的語音指令。完成後用簡短口語回報結果（150 字內）。\n\n"
              "指令：" + task)
    out_file = os.path.join(tempfile.gettempdir(), f"jarvis_assistant_{int(time.time())}.out")
    job = {"proc": None, "out_file": out_file, "task": task,
           "done": False, "result": None, "timed_out": False, "started": None}

    def _run():
        with BG_LOCK:
            job["started"] = time.time()
            try:
                f = open(out_file, "w", encoding="utf-8", errors="replace")
                p = subprocess.Popen(
                    [OPENCODE, "run", "--dir", WORKDIR, "--title", "jarvis-live-" + today_tag,
                     "--continue", prompt],
                    stdout=f, stderr=subprocess.STDOUT, text=True)
                job["proc"] = p
                f.close()
                p.wait()
            except Exception:
                pass

    os.makedirs(WORKDIR, exist_ok=True)
    threading.Thread(target=_run, daemon=True).start()
    return job


def _out_device():
    if not OUT_DEVICE:
        return None
    for i, d in enumerate(sd.query_devices()):
        if d["max_output_channels"] > 0 and OUT_DEVICE.lower() in d["name"].lower():
            return i
    return None


def _out_ctx():
    return sd.OutputStream(samplerate=24000, channels=1, dtype="int16", device=_out_device())


def play_chime():
    r = 24000
    t = np.arange(int(0.22 * r)) / r
    tone = (np.sin(2 * np.pi * 880 * t) * 0.35 + np.sin(2 * np.pi * 1320 * t) * 0.18)
    tone = (tone * 32767).astype(np.int16)
    try:
        with _out_ctx() as out:
            out.write(tone)
    except Exception:
        pass


def play_ack_tone():
    """Fallback non-verbal ack (two-note chime), queued immediately when a
    tool call is dispatched -- the spoken filler is the primary cue (see
    SYS rule 3), this is a backstop in case the model doesn't say one."""
    r = 24000
    seg = int(0.14 * r)
    t = np.arange(seg) / r
    a = (np.sin(2 * np.pi * 740 * t) * 0.4).astype(np.int16)
    b = (np.sin(2 * np.pi * 988 * t) * 0.4).astype(np.int16)
    return np.concatenate([a, b])


class LiveSession:
    """One Gemini Live conversation, from wake to idle-timeout/exit."""

    def __init__(self, on_status=None):
        self.on_status = on_status or (lambda *a: None)
        self.mic_q = queue.Queue()
        self.play_q = queue.Queue()
        self.text_q = queue.Queue()
        self.exit_evt = threading.Event()
        self.pause_evt = threading.Event()
        self.speaking = False
        self.turn_count = 0
        self.last_activity = time.time()
        self.session_open = True
        self.bg_jobs = []
        self.bg_lock = threading.Lock()
        self.transcript = []
        self._input_buf = ""
        self._output_buf = ""
        threading.Thread(target=self.bg_monitor, daemon=True).start()

    # ── face state (Jarvis' ring) ──────────────────────────────
    def _face(self, state):
        try:
            STATE_DIR.mkdir(exist_ok=True)
            (STATE_DIR / "state").write_text(state, encoding="utf-8")
        except Exception:
            pass

    # ── background opencode jobs ───────────────────────────────────
    def bg_monitor(self):
        while True:
            done = []
            with self.bg_lock:
                jobs = list(self.bg_jobs)
            for job in jobs:
                if job["done"]:
                    continue
                proc = job["proc"]
                if proc is None or job["started"] is None:
                    continue
                timed_out = time.time() - job["started"] > 480
                if proc.poll() is not None or timed_out:
                    if timed_out and proc.poll() is None:
                        job["timed_out"] = True
                        try:
                            proc.terminate()
                        except Exception:
                            pass
                        self.on_status(f"⏱ 背景任務「{job['task']}」執行太久，已回收")
                    job["done"] = True
                    done.append(job)
            if done:
                with self.bg_lock:
                    self.bg_jobs = [j for j in self.bg_jobs if not j["done"]]
                for job in done:
                    self.on_bg_done(job)
            if self.exit_evt.is_set() and not jobs:
                return
            time.sleep(2)

    def on_bg_done(self, job):
        try:
            with open(job["out_file"], encoding="utf-8", errors="replace") as f:
                out = f.read()
            out = re.sub(r"\x1b\[[0-9;]*m", "", out)
            out = re.sub(r"\r", "", out)
            lines = [l.strip() for l in out.splitlines() if l.strip()]
            result = "\n".join(lines[-20:])[-3500:].strip() or "（執行完成，無輸出）"
            if job.get("timed_out"):
                result = "⚠️ 任務執行超過 8 分鐘被強制結束，以下為目前進度：\n" + result
        except Exception as e:
            result = job.get("result") or ("（無法讀取執行結果：%r）" % e)
        job["result"] = result
        self.play_q.put(play_ack_tone())
        if not self.session_open:
            self.on_status(f"✅ 背景任務完成（session 已結束）：「{job['task']}」")
            return
        self.on_status(f"✅ 背景任務完成：「{job['task']}」")
        self.text_q.put(
            "（系統通知：你叫我處理的背景任務「%s」已完成。請用口語 2~3 句簡短告訴老大結果，不需要再執行任何工具。）\n結果：%s"
            % (job["task"], result))

    # ── mic input ───────────────────────────────────────────────────
    def mic_callback(self, indata, frames, time_info, status):
        arr = indata[:, 0].copy()
        rms = float(np.sqrt(np.mean(arr.astype(np.float32) ** 2)))
        if self.speaking or rms < 30.0:
            arr = np.zeros_like(arr)
        else:
            gain = min(60.0, 6000.0 / rms)
            arr = np.clip(arr.astype(np.float32) * gain, -32767, 32767).astype(np.int16)
        self.mic_q.put(arr)

    # ── speaker output ──────────────────────────────────────────────
    def playback_worker(self):
        try:
            out = _out_ctx()
            out.start()
        except Exception as e:
            print(f"❌ 播放初始化失敗：{e!r}", flush=True)
            return
        try:
            while not self.exit_evt.is_set():
                try:
                    data = self.play_q.get(timeout=0.15)
                except queue.Empty:
                    continue
                if self.pause_evt.is_set():
                    continue
                try:
                    if PLAY_GAIN > 1.0:
                        arr = np.frombuffer(data, dtype=np.int16).astype(np.float32)
                        arr *= PLAY_GAIN
                        arr = np.clip(arr, -32767, 32767)
                        data = arr.astype(np.int16)
                    self.speaking = True
                    self._face("speaking")
                    out.write(data)
                except Exception as e:
                    print(f"❌ 播放 write 失敗：{e!r}", flush=True)
                finally:
                    self.speaking = False
        finally:
            try:
                out.stop(); out.close()
            except Exception:
                pass

    def clear_playback(self):
        while True:
            try:
                self.play_q.get_nowait()
            except queue.Empty:
                break

    async def mic_sender(self, ws):
        while self.session_open and not self.exit_evt.is_set():
            try:
                text = self.text_q.get_nowait()
            except queue.Empty:
                pass
            else:
                if text:
                    await ws.send(json.dumps({
                        "client_content": {"turns": [{"role": "user",
                                                      "parts": [{"text": text}]}],
                                           "turn_complete": True}}))
                    self.on_status(f"你：「{text}」")
                    self.last_activity = time.time()
                continue
            if self.pause_evt.is_set():
                try:
                    self.mic_q.get_nowait()
                except queue.Empty:
                    pass
                await asyncio.sleep(0.1)
                continue
            try:
                chunk = self.mic_q.get_nowait()
            except queue.Empty:
                if not self.speaking:
                    self._face("listening")
                await asyncio.sleep(0.05)
                continue
            await ws.send(json.dumps({
                "realtime_input": {"audio": {
                    "data": base64.b64encode(np.ascontiguousarray(chunk)).decode(),
                    "mime_type": "audio/pcm;rate=16000"}}}))
            await asyncio.sleep(0)

    async def receiver(self, ws):
        async for raw in ws:
            try:
                msg = json.loads(raw)
            except Exception:
                continue
            tc = msg.get("tool_call") or msg.get("toolCall")
            if tc:
                fcs = tc.get("function_calls") or tc.get("functionCalls") or []
                await self.handle_tool_calls(ws, fcs)
                continue

            sc = msg.get("server_content") or msg.get("serverContent")
            if not sc:
                continue

            if sc.get("interrupted"):
                self.clear_playback()
                self.on_status("（插話打斷）")

            mt = sc.get("model_turn") or sc.get("modelTurn") or {}
            for part in mt.get("parts", []):
                if "inline_data" in part or "inlineData" in part:
                    raw_a = part.get("inline_data") or part.get("inlineData")
                    audio = np.frombuffer(base64.b64decode(raw_a.get("data", "")), dtype=np.int16)
                    self.play_q.put(audio.copy())
                    self.last_activity = time.time()

            it = sc.get("input_transcription") or sc.get("inputTranscription")
            if it and it.get("text"):
                t = it["text"].strip()
                if t.startswith(self._input_buf):
                    new_part = t[len(self._input_buf):]
                    self._input_buf = t
                else:
                    new_part = t
                    self._input_buf = t
                if self.transcript and self.transcript[-1].startswith("老大："):
                    self.transcript[-1] = f"老大：{t}"
                else:
                    self.transcript.append(f"老大：{t}")
                if new_part:
                    self.on_status(f"你▸{new_part}")
                if any(p in t for p in PAUSE_PHRASES):
                    self.pause_evt.set()
                    self.clear_playback()
                    self.on_status("⏸ 已暫停（說『繼續』恢復）")
                if any(p in t for p in RESUME_PHRASES):
                    self.pause_evt.clear()
                    self.on_status("▶ 已恢復")
                if any(p in t for p in EXIT_PHRASES):
                    self.exit_evt.set()

            if sc.get("turn_complete") or sc.get("turnComplete"):
                self.turn_count += 1
                self.last_activity = time.time()
                self._input_buf = ""
                self._output_buf = ""
                self.on_status("◉turn-done")

            ot = sc.get("output_transcription") or sc.get("outputTranscription")
            if ot and ot.get("text"):
                t = ot["text"].strip()
                if t.startswith(self._output_buf):
                    new_part = t[len(self._output_buf):]
                    self._output_buf = t
                else:
                    new_part = t
                    self._output_buf = t
                if self.transcript and self.transcript[-1].startswith("Sadie："):
                    self.transcript[-1] = f"Sadie：{t}"
                else:
                    self.transcript.append(f"Sadie：{t}")
                if new_part:
                    self.on_status("Sadie▸" + new_part)

    async def handle_tool_calls(self, ws, function_calls):
        responses = []
        for fc in function_calls:
            fc_id = fc.get("id", "")
            name = fc.get("name", "")
            args = fc.get("args") or {}
            if name == "run_opencode":
                task = args.get("task", "")
                self._face("thinking")
                self.on_status(f"🤔 思考中…（執行：「{task}」）")
                self.play_q.put(play_ack_tone())
                self.turn_count += 1
                self.last_activity = time.time()
                job = start_opencode_bg(task)
                if job.get("proc") is not None or not job["done"]:
                    with self.bg_lock:
                        self.bg_jobs.append(job)
                    msg = "任務「%s」已丟到背景執行，完成後我會通知你。" % task
                    self.on_status(f"🔧 背景執行中：「{task}」")
                else:
                    msg = job.get("result") or "背景執行啟動失敗，請稍後再試。"
                    self.on_status("❌ 背景執行啟動失敗")
                responses.append({"id": fc_id, "name": name, "response": {"result": msg}})
            elif name == "show_on_board":
                title = str(args.get("title", "")).strip() or "Sadie"
                body = str(args.get("body", "")).strip()
                if _queue_cmd is None:
                    msg = "board 沒有連上，秀不出來。"
                else:
                    try:
                        _queue_cmd({"a": "add_card", "title": title, "body": body})
                        msg = "已經秀在畫面上了。"
                        self.on_status(f"🗂 秀上畫面：「{title}」")
                    except Exception as e:
                        msg = f"秀上畫面失敗：{e!r}"
                responses.append({"id": fc_id, "name": name, "response": {"result": msg}})
            elif name == "list_models":
                if _list_models is None:
                    msg = "沒接上 board，看不到有哪些模型。"
                else:
                    try:
                        files = _list_models()
                        msg = ("目前可召喚的模型：" + "、".join(files)) if files \
                              else "media 資料夾裡目前沒有任何 .glb/.gltf 模型檔。"
                    except Exception as e:
                        msg = f"列模型失敗：{e!r}"
                responses.append({"id": fc_id, "name": name, "response": {"result": msg}})
            elif name == "summon_model":
                file = str(args.get("file", "")).strip()
                title = str(args.get("title", "")).strip() or file
                if _queue_cmd is None or _resolve_media_src is None:
                    msg = "board 沒有連上，召喚不出來。"
                elif not file:
                    msg = "沒有指定要召喚哪個模型檔。"
                else:
                    try:
                        src = _resolve_media_src(file)
                        _queue_cmd({"a": "present", "src": src, "title": title})
                        msg = f"「{title}」已經召喚到畫面中央了。"
                        self.on_status(f"🔮 召喚模型：「{title}」")
                    except Exception as e:
                        msg = f"找不到這個模型檔，或不在允許的資料夾裡：{e!r}"
                responses.append({"id": fc_id, "name": name, "response": {"result": msg}})
            else:
                # Only the tools above are registered in TOOLS -- an
                # unknown call shouldn't be able to happen, but fail
                # closed if it does rather than silently drop it.
                responses.append({"id": fc_id, "name": name,
                                   "response": {"result": "此工具未啟用。"}})
                continue
            self.last_activity = time.time()
        if responses:
            await ws.send(json.dumps({"tool_response": {"function_responses": responses}}))

    async def run(self):
        attempts = 0
        while True:
            reason = await self._round()
            if reason != "connection-closed" or attempts >= config.RECONNECT_MAX:
                return reason
            attempts += 1
            self.on_status(f"⚠️ 連線中斷，正在自動重連（{attempts}/{config.RECONNECT_MAX}）…")
            await asyncio.sleep(2 * attempts)
            self.exit_evt.clear()
            self.session_open = True

    async def _round(self):
        if not config.gemini_key():
            print("❌ 沒有 Gemini 憑證。請設定環境變數 GEMINI_API_KEY。", flush=True)
            return "no-key"

        ACTIVE_SESSION[0] = self
        try:
            self.on_status("⏳ 連線 Gemini Live…")
            self._face("thinking")
            ws = await websockets.connect(_ws_url(), additional_headers=_ws_headers(),
                                          max_size=64 * 1024 * 1024)
            setup = {
                "setup": {
                    "model": "models/" + MODEL,
                    "generation_config": {
                        "response_modalities": ["AUDIO"],
                        "speech_config": {"voice_config": {"prebuilt_voice_config": {"voice_name": VOICE}}},
                    },
                    "system_instruction": {"parts": [{"text": SYS}]},
                    "tools": TOOLS,
                    "realtime_input_config": {
                        "automatic_activity_detection": {"silence_duration_ms": 1200}
                    },
                    "input_audio_transcription": {},
                    "output_audio_transcription": {},
                }
            }
            await ws.send(json.dumps(setup))
            got_setup = False
            for _ in range(12):
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=8)
                except asyncio.TimeoutError:
                    break
                m = json.loads(raw)
                if "setupComplete" in m:
                    got_setup = True
                    break
                if "error" in m:
                    print(f"❌ setup 錯誤：{json.dumps(m)[:300]}", flush=True)
                    break
            if not got_setup:
                self.on_status("❌ 連線未完成")
                await ws.close()
                self._face("idle")
                return "setup-failed"
            self.on_status("✅ 連線成功，等老大開口…")
        except Exception as e:
            print(f"❌ 連線失敗：{e!r}", flush=True)
            self._face("idle")
            return "connect-failed"

        threading.Thread(target=self.playback_worker, daemon=True).start()
        dev = int(MIC_INDEX) if MIC_INDEX is not None else None

        await asyncio.to_thread(self._prime_input_stream)
        try:
            await ws.send(json.dumps({
                "client_content": {"turns": [{"role": "user",
                                              "parts": [{"text": "（喚醒成功，請開始講話）"}]}],
                                   "turn_complete": True}}))
            self.on_status("✅ 連線成功，Sadie 說話中…")
        except Exception as e:
            print(f"❌ 送初始 turn 失敗：{e!r}", flush=True)
        reason = "closed"
        try:
            with sd.InputStream(samplerate=16000, channels=1, dtype="int16",
                                callback=self.mic_callback, device=dev):
                self.on_status("✅ 對話中（說「掰掰」或 25 秒沒講話就回待機）")
                self._face("listening")
                self.play_q.put(np.zeros(int(24000 * 0.5), dtype=np.int16))
                monitor = asyncio.create_task(self.monitor(ws))
                try:
                    await asyncio.gather(self.mic_sender(ws), self.receiver(ws), monitor)
                except websockets.exceptions.ConnectionClosed as e:
                    self.on_status(f"（連線中斷：{e.code}）")
                    reason = "connection-closed"
        except Exception as e:
            print(f"❌ 對話層錯誤：{e!r}", flush=True)
            traceback.print_exc()
            reason = "error"
        finally:
            self.exit_evt.set()
            self.session_open = False
            self.clear_playback()
            self._face("idle")
            if ACTIVE_SESSION[0] is self:
                ACTIVE_SESSION[0] = None
            try:
                await ws.close()
            except Exception:
                pass
        return reason

    def _prime_input_stream(self):
        """Open a throwaway 0.4s input stream to flush the tail of the
        wake word out of the buffer, so it can't be misread as a command."""
        try:
            with sd.InputStream(samplerate=16000, channels=1, dtype="int16",
                                blocksize=6400, device=None):
                time.sleep(0.4)
        except Exception:
            pass

    async def monitor(self, ws):
        while not self.exit_evt.is_set():
            await asyncio.sleep(0.5)
            if self.turn_count >= MAX_TURNS:
                self.on_status("（對話結束：已達輪數上限）")
                self.exit_evt.set()
                break
            with self.bg_lock:
                bg_pending = any(not j["done"] for j in self.bg_jobs)
            if bg_pending:
                continue
            if self.turn_count > 0 and time.time() - self.last_activity > IDLE_TIMEOUT:
                self.on_status("（對話結束：閒置超過 25 秒）")
                self.exit_evt.set()
                break
        if self.exit_evt.is_set():
            try:
                await ws.close()
            except Exception:
                pass


async def _run_live_round():
    sess = LiveSession(on_status=print)
    return await sess.run()


def _wake_loop_main():
    """Blocking wake-word -> conversation loop. Run this in a background
    thread from server.py; never call it on the main thread, it owns the
    microphone/speaker for as long as Jarvis is up."""
    print("═══════════════ Jarvis assistant (Sadie) ═══════════════", flush=True)
    if not WAKE_MODEL.is_dir():
        print(f"⚠️ 找不到喚醒詞模型：{WAKE_MODEL}（需要另外下載，語音助手不會啟動；board 仍正常運作）", flush=True)
        return
    if not config.gemini_key():
        print("⚠️ 未設定 GEMINI_API_KEY，語音助手不會啟動；board 仍正常運作。", flush=True)
        return
    print(f"待機中——喊「{config.WAKE_HINT}」喚醒。", flush=True)
    listener = WakewordListener(WAKE_MODEL,
                                device=int(MIC_INDEX) if MIC_INDEX is not None else None)
    _LISTENER[0] = listener
    was_sleeping = False
    while True:
        if _SLEEPING.is_set():
            if not was_sleeping:
                print("😴 已暫停，麥克風已釋放（在網頁上按喚醒才能恢復——睡著就是真的聽不到）", flush=True)
                was_sleeping = True
            time.sleep(0.3)
            continue
        if was_sleeping:
            print(f"⏰ 已恢復——喊「{config.WAKE_HINT}」喚醒。", flush=True)
            was_sleeping = False
        try:
            word = listener.listen()
        except Exception as e:
            print(f"⚠️ 喚醒詞監聽發生錯誤：{e!r}", flush=True)
            time.sleep(2)
            continue
        if not word:
            continue
        print(f"\n🎤 喚醒：{word}", flush=True)
        play_chime()
        try:
            reason = asyncio.run(_run_live_round())
            if reason:
                print(f"（對話結束：{reason}）", flush=True)
        except Exception:
            print("⚠️ 對話層發生未預期錯誤：", flush=True)
            traceback.print_exc()
        print(f"回待機——喊「{config.WAKE_HINT}」再次喚醒。", flush=True)


_queue_cmd = None          # set by start_assistant_thread(); see show_on_board
_resolve_media_src = None  # set by start_assistant_thread(); see summon_model
_list_models = None        # set by start_assistant_thread(); see list_models


def start_assistant_thread(queue_cmd=None, resolve_media_src=None, list_models=None):
    """Start the wake-word/conversation loop as a background daemon
    thread. Safe to call from server.py's __main__: if dependencies or
    credentials are missing, this prints a message and returns without
    starting anything -- it never crashes the HTTP server.

    queue_cmd: optional callable(cmd: dict) that appends a board command
    directly into server.py's in-process _CMDS queue -- same process, so
    no HTTP round-trip to /cmd is needed. Without it, show_on_board just
    reports it can't reach the board instead of raising.

    resolve_media_src / list_models: optional callables from server.py
    (resolve_media_src(src) -> "/media/..." URL, raises if outside the
    media jail; list_models() -> list of .glb/.gltf paths) -- used by
    summon_model. The assistant runs in-process now, so it must still
    go through the same media-jail validation the HTTP /cmd path uses,
    not skip it just because there's no network hop."""
    global _queue_cmd, _resolve_media_src, _list_models
    _queue_cmd = queue_cmd
    _resolve_media_src = resolve_media_src
    _list_models = list_models
    t = threading.Thread(target=_wake_loop_main, daemon=True, name="jarvis-assistant")
    t.start()
    return t
