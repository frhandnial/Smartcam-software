"""Plain-English descriptions of the camera picture from Google Gemini (the free key JARVIS and ZACK use)."""
import base64
import json
import os
import threading
import urllib.error
import urllib.request

API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
MODELS = ["gemini-flash-lite-latest", "gemini-3.1-flash-lite", "gemini-flash-latest"]  # tried in this order
RULES = ("Never guess anyone's identity, name, age, ethnicity, religion or other personal traits; describe "
         "clothing, actions and objects instead. Don't invent things you can't see.")

_lock = threading.Lock()


class AIError(Exception):
    pass


def api_key():
    v = os.environ.get("GEMINI_API_KEY", "").strip()
    if v or os.name != "nt":
        return v
    try:  # set with setx after this program's parent started, so not in our environment yet
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
            return str(winreg.QueryValueEx(k, "GEMINI_API_KEY")[0]).strip()
    except OSError:
        return ""


def _call(parts):
    key = api_key()
    if not key:
        raise AIError("No Gemini API key. Set GEMINI_API_KEY (the free key from aistudio.google.com).")
    body = json.dumps({"contents": [{"role": "user", "parts": parts}],
                       "generationConfig": {"temperature": 0.3}}).encode("utf-8")
    with _lock:  # one request at a time on the shared free key
        last = None
        for model in MODELS:
            req = urllib.request.Request(API_URL.format(model=model), data=body,
                                         headers={"Content-Type": "application/json", "x-goog-api-key": key})
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    data = json.load(r)
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", "replace")
                if e.code in (429, 500, 503, 404):
                    last = e.code
                    continue
                if e.code in (400, 403) and "API key" in detail:
                    raise AIError("Gemini rejected the API key. Check GEMINI_API_KEY.")
                raise AIError("Gemini error {}: {}".format(e.code, detail[:200]))
            except urllib.error.URLError as e:
                raise AIError("Can't reach Gemini (no internet?): {}".format(e.reason))
            cands = data.get("candidates") or []
            text = "".join(p.get("text", "") for p in (cands[0].get("content", {}).get("parts") or [])
                           if not p.get("thought")) if cands else ""
            if text.strip():
                return text.strip()
            last = "empty"
        if last == 429:
            raise AIError("Gemini's free limit is used up for now. Try again in a minute (or tomorrow).")
        raise AIError("Gemini is busy right now. Try again in a minute.")


def _image(jpeg):
    return {"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(jpeg).decode("ascii")}}


def describe(jpeg, camera, context):
    prompt = ("You are the analyst watching a CCTV camera called '{}'. In one or two short sentences, say what "
              "is happening in this frame right now: who or what is there, what they seem to be doing, and "
              "anything unusual or worth attention. Plain everyday English, no preamble. {}\n\n"
              "The object detector currently sees: {}").format(camera, RULES, context)
    return _call([{"text": prompt}, _image(jpeg)])


def ask(jpeg, camera, question, context):
    prompt = ("You are looking at a live frame from a CCTV camera called '{}'. Answer the user's question about "
              "it in at most three sentences. If the picture doesn't show enough to tell, say so. {}\n\n"
              "The object detector currently sees: {}\n\nQuestion: {}").format(camera, RULES, context, question)
    return _call([{"text": prompt}, _image(jpeg)])


def summarise(camera, day, lines):
    prompt = ("Below is the event log from the CCTV camera '{}' for {}. Write a short summary of the day for the "
              "owner: 3 to 6 bullet points (start each with '- '), busiest times, anything unusual, and every "
              "alert worth checking. Plain everyday English, no headings.\n\n{}").format(camera, day,
                                                                                     "\n".join(lines))
    return _call([{"text": prompt}])
