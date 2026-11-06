#!/usr/bin/env python3
"""Output-quality battery for any OpenAI-compatible server, production sampling.
Saves every answer to answers-<label>.jsonl and prints one JSON summary line.
usage: [EVAL_SAMPLING=server] evalq.py <label> <base_url> <model>  (copy of ~/code/exl3/evalq.py)"""
import json, os, re, random, subprocess, sys, tempfile, time, urllib.request
import pandas as pd

label, base, model = sys.argv[1:4]
D = os.path.expanduser("~/code/exl3/evaldata")
SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.05, "reasoning_effort": "low"}
# EVAL_SAMPLING=server: send no sampling fields, so the server preset applies (production: qwen38 = temp 0.7,
# presence_penalty 1.0, top_k 20, top_p 0.95, min_p 0.05; reasoning_effort low from template_vars_default)
if os.environ.get("EVAL_SAMPLING") == "server": SAMPLING = {}
MAXTOK = 6144
_ap = os.path.expanduser(f"~/code/exl3/answers-{label}.jsonl")
DONE = {}
if os.environ.get("EVAL_RESUME") == "1" and os.path.exists(_ap):
    for _l in open(_ap):
        _r = json.loads(_l); DONE[(_r["task"], _r["i"])] = _r
out = open(_ap, "a" if DONE else "w")

CUR = [None]
def chat(messages, tools=None):
    if CUR[0] in DONE:
        r = DONE[CUR[0]]
        return {"content": r["content"], "tool_calls": r["tool_calls"], "reasoning_content": r["reasoning"]}, r["finish"], r["usage"], 0
    body = {"model": model, "messages": messages, "max_tokens": MAXTOK, **SAMPLING}
    if tools: body["tools"] = tools
    req = urllib.request.Request(f"{base}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=3600))
    m = r["choices"][0]["message"]
    return m, r["choices"][0].get("finish_reason"), r.get("usage", {}), time.time() - t

def log(task, i, prompt, m, fin, usage, ok, extra=None):
    if (task, i) in DONE: return
    out.write(json.dumps({"task": task, "i": i, "ok": ok, "finish": fin, "usage": usage, "prompt": prompt,
                          "content": m.get("content"), "reasoning": m.get("reasoning_content") or m.get("reasoning"),
                          "tool_calls": m.get("tool_calls"), **(extra or {})}) + "\n"); out.flush()

rng = random.Random(1234)
res = {}

# --- GSM8K: 150 problems
g = pd.read_parquet(f"{D}/gsm8k/main/test-00000-of-00001.parquet")
idx = rng.sample(range(len(g)), 150); ok = 0
for i in idx:
    q, a = g.question[i], g.answer[i].split("####")[-1].strip().replace(",", "")
    p = q + "\n\nSolve the problem. End your reply with a line of the form: ANSWER: <number>"
    CUR[0] = ("gsm8k", int(i)); m, fin, u, _ = chat([{"role": "user", "content": p}])
    got = re.findall(r"ANSWER:\s*\$?(-?[\d,]*\.?\d+)", m.get("content") or "")
    good = bool(got) and float(got[-1].replace(",", "")) == float(a)
    ok += good; log("gsm8k", int(i), p, m, fin, u, good, {"expected": a})
res["gsm8k"] = round(ok / len(idx) * 100, 1)

# --- HumanEval: all 164, run against the tests
h = pd.read_parquet(f"{D}/humaneval/openai_humaneval/test-00000-of-00001.parquet"); ok = 0
for i, row in h.iterrows():
    p = ("Complete this Python function. Reply with the full function in one ```python code block.\n\n```python\n"
         + row.prompt + "```")
    CUR[0] = ("humaneval", int(i)); m, fin, u, _ = chat([{"role": "user", "content": p}])
    blocks = re.findall(r"```(?:python)?\n(.*?)```", m.get("content") or "", re.S)
    code = blocks[-1] if blocks else ""
    prog = row.prompt + "\n" + code + "\n\n" + row.test + f"\n\ncheck({row.entry_point})\n"
    with tempfile.TemporaryDirectory() as td:
        f = os.path.join(td, "t.py"); open(f, "w").write(prog)
        try:
            good = subprocess.run([sys.executable, "-I", f], cwd=td, capture_output=True, timeout=20).returncode == 0
        except subprocess.TimeoutExpired:
            good = False
    ok += good; log("humaneval", int(i), p, m, fin, u, good)
res["humaneval"] = round(ok / len(h) * 100, 1)

# --- MMLU-Pro: 150 questions
mp = pd.read_parquet(f"{D}/mmlupro/data/test-00000-of-00001.parquet")
idx = rng.sample(range(len(mp)), 150); ok = 0
for i in idx:
    r = mp.iloc[i]; letters = "ABCDEFGHIJ"
    opts = "\n".join(f"{letters[k]}. {o}" for k, o in enumerate(r.options))
    p = f"{r.question}\n\n{opts}\n\nThink it through, then end your reply with: The answer is (X)"
    CUR[0] = ("mmlupro", int(i)); m, fin, u, _ = chat([{"role": "user", "content": p}])
    got = re.findall(r"answer is \(?([A-J])\)?", m.get("content") or "")
    good = bool(got) and got[-1] == r.answer
    ok += good; log("mmlupro", int(i), p, m, fin, u, good, {"expected": r.answer, "category": r.category})
res["mmlupro"] = round(ok / len(idx) * 100, 1)

# --- Tool calls: right function, valid JSON args with the required keys
T = lambda n, d, props, req: {"type": "function", "function": {"name": n, "description": d,
     "parameters": {"type": "object", "properties": props, "required": req}}}
S = {"type": "string"}; N = {"type": "number"}
TOOLS = [T("get_weather", "Current weather for a city", {"city": S}, ["city"]),
         T("send_email", "Send an email", {"to": S, "subject": S, "body": S}, ["to", "subject", "body"]),
         T("search_web", "Search the web", {"query": S}, ["query"]),
         T("read_file", "Read a file from disk", {"path": S}, ["path"]),
         T("run_shell", "Run a shell command", {"command": S}, ["command"]),
         T("create_event", "Create a calendar event", {"title": S, "start": S, "duration_minutes": N}, ["title", "start"]),
         T("convert_currency", "Convert money", {"amount": N, "from": S, "to": S}, ["amount", "from", "to"])]
CASES = [("What's the weather in Perth right now?", "get_weather"),
         ("Email bob@example.com with subject 'Dinner' saying I'll be 10 minutes late.", "send_email"),
         ("Find recent news about the RX 9070 launch.", "search_web"),
         ("Show me what's in /etc/hostname.", "read_file"),
         ("How much free disk space is there? Check with df.", "run_shell"),
         ("Book a 30 minute call called 'Dentist' for 2026-11-02 09:00.", "create_event"),
         ("How many euros is 250 Australian dollars?", "convert_currency"),
         ("Is it raining in Tokyo?", "get_weather"),
         ("List the files in my home directory.", "run_shell"),
         ("Look up who won the 2026 AFL grand final.", "search_web"),
         ("Open ~/notes/todo.md and tell me what's on it.", "read_file"),
         ("Send alice@example.org an email titled 'Report' with the body 'Attached is the Q3 report.'", "send_email"),
         ("Convert 99.5 USD to JPY.", "convert_currency"),
         ("Put 'Team sync' in my calendar tomorrow at 2pm, 2026-10-16 14:00, for an hour.", "create_event"),
         ("What's the temperature in Reykjavik?", "get_weather"),
         ("Check which process is using port 8080 with ss.", "run_shell"),
         ("Search for the llama.cpp DFlash2 pull request.", "search_web"),
         ("Read /var/log/syslog and summarise the last errors.", "read_file"),
         ("Schedule 'Gym' on 2026-10-20 at 06:30.", "create_event"),
         ("Tell me how many British pounds 1200 Swiss francs buys.", "convert_currency")]
ok = 0
for i, (p, want) in enumerate(CASES):
    CUR[0] = ("tools", i); m, fin, u, _ = chat([{"role": "system", "content": "You are a helpful assistant. Use the tools when they help."},
                         {"role": "user", "content": p}], TOOLS)
    good = False
    for tc in m.get("tool_calls") or []:
        fn = tc.get("function", {})
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except Exception:
            continue
        spec = next(t for t in TOOLS if t["function"]["name"] == want)["function"]["parameters"]["required"]
        if fn.get("name") == want and all(k in args for k in spec):
            good = True
    ok += good; log("tools", i, p, m, fin, u, good, {"expected": want})
res["tools"] = round(ok / len(CASES) * 100, 1)

print(json.dumps({"label": label, **res}), flush=True)
