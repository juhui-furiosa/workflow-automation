import os, json, re, time, requests
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from typing import Dict, Any, List, Optional, Literal
from pathlib import Path

from dotenv import load_dotenv
from rich import print as rprint

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from ics import Calendar

BASE = Path(__file__).parent

load_dotenv()
FURIOSA_ENDPOINT = os.getenv("FURIOSA_ENDPOINT", "http://127.0.0.1:8000/v1/chat/completions")
FURIOSA_MODEL = None # 자동 초기화

SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "").strip()
NOTION_TOKEN = os.getenv("NOTION_TOKEN", "").strip()
NOTION_DB_ID = os.getenv("NOTION_DB_ID", "").strip()
NOTION_TITLE_PROP = "Title"
NOTION_DATE_PROP = "Date"

# Timezone settings
TZ = ZoneInfo("Asia/Seoul")
TZ_LABEL = "KST"

UUID32_RE = re.compile(r"([0-9a-fA-F]{32})")
def _canonical_uuid(s: str) -> str:
    m = UUID32_RE.search(s or "")
    if not m:
        return s
    raw = m.group(1).lower()
    return f"{raw[0:8]}-{raw[8:12]}-{raw[12:16]}-{raw[16:20]}-{raw[20:32]}"

def _normalize_notion_ids():
    global NOTION_DB_ID
    if NOTION_DB_ID:
        canon = _canonical_uuid(NOTION_DB_ID)
        if canon != NOTION_DB_ID:
            rprint(f"[dim]Notion DB ID normalization: {canon}[/dim]")
            NOTION_DB_ID = canon

_normalize_notion_ids()


def _ensure_model():
    global FURIOSA_MODEL
    if FURIOSA_MODEL:
        return FURIOSA_MODEL
    try:
        base_url = FURIOSA_ENDPOINT.split("/v1/")[0] + "/v1/models"
        resp = requests.get(base_url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            models = []
            if isinstance(data, dict) and "data" in data:
                for m in data["data"]:
                    mid = m.get("id")
                    if mid:
                        models.append(mid)
            if models:
                FURIOSA_MODEL = models[0]
                rprint(f"[dim]Automatically selected model: {FURIOSA_MODEL}[/dim]")
                return FURIOSA_MODEL
    except Exception as e:
        rprint(f"[red]Model auto-selection failed: {e}[/red]")
    FURIOSA_MODEL = "unknown-model"
    return FURIOSA_MODEL

def llm_chat(prompt: str, max_tokens: int = 512, temperature: float = 0.2) -> str:
    model = _ensure_model()
    payload = {
        "model": model,
        "messages": [
            {"role":"system","content":"You are a precise assistant. Always reply in the user's language (Korean if Korean present else English). Maintain concise style and respect JSON format when requested."},
            {"role":"user","content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    try:
        r = requests.post(FURIOSA_ENDPOINT, json=payload, timeout=60)
        if r.status_code >= 400:
            rprint(f"[red]Fail to request LLM {r.status_code}: {r.text[:300]}[/red]")
            raise requests.HTTPError(f"LLM error {r.status_code}")
        data = r.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        rprint(f"[yellow]Fail to request LLM, fallback used: {e}[/yellow]")
        return "{\n  \"action\": \"noop\",\n  \"params\": {\n    \"days\": 1,\n    \"post_to_slack\": false,\n    \"write_to_notion\": false\n  }\n}"

JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
def extract_json(s: str) -> Dict[str, Any]:
    m = JSON_RE.search(s)
    if not m:
        raise ValueError("No JSON found")
    return json.loads(m.group(0))

def fetch_events_from_ical(days: int = 1, start: Optional[datetime] = None) -> List[Dict[str, Any]]:
    cal_text = None

    default_local = BASE / "data" / "basic.ics"
    if default_local.exists():
        source = str(default_local)
    else:
        return []

    # Interpret as local path if it looks like a path (exists or no scheme)
    def _looks_like_path(s: str) -> bool:
        return s.startswith("./") or s.startswith("../") or s.startswith("/") or (not re.match(r"^[a-zA-Z]+://", s))

    path_candidate = None
    if _looks_like_path(source):
        # Try as absolute or relative to BASE
        p = Path(source)
        if not p.exists():
            p2 = (BASE / source).resolve()
            if p2.exists():
                p = p2
        if p.exists():
            path_candidate = p

    if path_candidate is not None and path_candidate.exists():
        try:
            cal_text = path_candidate.read_text(encoding="utf-8")
            rprint(f"[dim]Loaded ICS locally: {path_candidate}[/dim]")
        except Exception as e:
            rprint(f"[yellow]Fail to read local ICS: {e}[/yellow]")
            return []
    else:
        # Fetch remote
        try:
            resp = requests.get(source, timeout=20)
            resp.raise_for_status()
            cal_text = resp.text
            rprint(f"[dim]Fetched ICS remotely: {source}[/dim]")
        except Exception as e:
            rprint(f"[yellow]Fail to fetch ICS: {e}[/yellow]")
            return []
    try:
        cal = Calendar(cal_text)
    except Exception as e:
        rprint(f"[yellow]Fail to parse ICS: {e}[/yellow]")
        return []

    utc = timezone.utc
    now_utc = (start or datetime.now(utc)).astimezone(utc)
    end_utc = now_utc + timedelta(days=days)

    out = []
    for ev in cal.events:
        b = getattr(ev, "begin", None)
        e = getattr(ev, "end", None)
        if not b or not e:
            continue
        bdt = b.datetime if hasattr(b, "datetime") else None
        edt = e.datetime if hasattr(e, "datetime") else None
        if not bdt or not edt:
            continue
        if edt < now_utc or bdt > end_utc:
            continue
        out.append({
            "title": ev.name or "No title",
            "start": bdt.astimezone(TZ).strftime("%Y-%m-%d %H:%M"),
            "end": edt.astimezone(TZ).strftime("%Y-%m-%d %H:%M"),
            "location": getattr(ev, "location", "") or "",
            "tz": TZ_LABEL
        })
    return sorted(out, key=lambda x: x["start"])

def summarize_events(events: List[Dict[str, Any]], lang: str) -> str:
    if not events:
        return "일정이 없습니다." if lang == "ko" else "No events."
    header = "📆 오늘 일정 요약\n" if lang == "ko" else "📆 Schedule Summary\n"
    lines = [header]
    for ev in events:
        start = ev.get("start")  # 'YYYY-MM-DD HH:MM'
        end = ev.get("end")
        title = ev.get("title") or ("(제목 없음)" if lang == "ko" else "(No title)")
        loc = ev.get("location")
        seg = f"{start} - {end} ({TZ_LABEL}) {title}"
        if loc:
            seg += f" @ {loc}"
        lines.append(f"• {seg}")
    return "\n".join(lines)

def post_to_slack(text: str) -> Dict[str, Any]:
    if not SLACK_WEBHOOK_URL:
        return {"ok": False, "note": "SLACK_WEBHOOK_URL 미설정"}
    try:
        r = requests.post(SLACK_WEBHOOK_URL, json={"text": text}, timeout=15)
        ok = r.status_code in (200, 204)
        return {"ok": ok, "status": r.status_code, "text": r.text if not ok else "OK"}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def write_to_notion(text: str, lang: str = "ko") -> Dict[str, Any]:
    if not NOTION_TOKEN:
        return {"ok": False, "note": "NOTION_TOKEN 미설정"}

    url = "https://api.notion.com/v1/pages"
    headers = {
        "Authorization": f"Bearer {NOTION_TOKEN}",
        "Notion-Version": "2022-06-28",
        "Content-Type": "application/json"
    }
    if lang == "ko":
        title_str = datetime.now().strftime("%Y-%m-%d 📆 일정 요약 보고\n")
    else:
        title_str = datetime.now().strftime("%Y-%m-%d 📆 Schedule Summary\n")
    today_iso = datetime.now().date().isoformat()

    if NOTION_DB_ID:
        properties = {
            NOTION_TITLE_PROP: {"title":[{"text":{"content": title_str}}]},
        }
        if NOTION_DATE_PROP:
            properties[NOTION_DATE_PROP] = {"date": {"start": today_iso}}
        payload = {
            "parent": {"type":"database_id","database_id": NOTION_DB_ID},
            "properties": properties,
            "children": [
                {
                    "object":"block","type":"paragraph",
                    "paragraph":{"rich_text":[{"type":"text","text":{"content": text}}]}
                }
            ]
        }
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=20)
        ok = r.status_code in (200, 201)
        return {"ok": ok, "status": r.status_code, "text": r.text if not ok else "OK"}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def plan_from_nl(user_text: str, lang: str) -> Dict[str, Any]:
        if lang == "ko":
                prompt = f"""
다음 명령을 읽고 워크플로우 계획을 JSON으로만 출력하세요.
가능한 action:
- "calendar_summary": 오늘/내일/이번주 일정 요약 → 슬랙 → 노션 기록
- "noop": 아무 작업 없음

명령: "{user_text}"

형식(이 키만):
{{
    "action": "calendar_summary" | "noop",
    "params": {{
        "days": 1,
        "post_to_slack": true,
        "write_to_notion": true
    }}
}}
"""
        else:
                prompt = f"""
Read the command and output ONLY a JSON workflow plan.
Allowed actions:
- "calendar_summary": summarize today's / tomorrow's / week's schedule → send to Slack → record to Notion
- "noop": do nothing

Command: "{user_text}"

Format (ONLY these keys):
{{
    "action": "calendar_summary" | "noop",
    "params": {{
        "days": 1,
        "post_to_slack": true,
        "write_to_notion": true
    }}
}}
"""
        raw = llm_chat(prompt, max_tokens=300, temperature=0.1)
        try:
                return extract_json(raw)
        except Exception:
                lowered = user_text.lower()
                if lang == "ko":
                        action = "calendar_summary" if ("일정" in user_text or "스케줄" in user_text) else "noop"
                else:
                        action = "calendar_summary" if ("schedule" in lowered or "calendar" in lowered) else "noop"
                return {"action": action, "params": {"days": 1, "post_to_slack": True, "write_to_notion": True}}


from typing import TypedDict

class WFState(TypedDict, total=False):
    user_text: str
    lang: str
    plan: Dict[str, Any]
    events: List[Dict[str, Any]]
    summary: str
    slack_result: Dict[str, Any]
    notion_result: Dict[str, Any]
    error: str

HANGUL_RE = re.compile(r"[\u3131-\u318E\uAC00-\uD7A3]")
def detect_language(text: str) -> str:
    return "ko" if HANGUL_RE.search(text) else "en"

def node_plan(state: WFState) -> WFState:
    lang = detect_language(state["user_text"])
    plan = plan_from_nl(state["user_text"], lang)
    rprint(f"[dim]Plan(lang={lang}) → {plan}[/dim]")
    return {"plan": plan, "lang": lang}

def node_calendar(state: WFState) -> WFState:
    p = state.get("plan", {})
    params = p.get("params", {})
    days = int(params.get("days", 1))
    events = fetch_events_from_ical(days=days)
    if not events and days == 1:
        expanded = fetch_events_from_ical(days=30)
        if expanded:
            state['plan']['params']['days'] = 30
            events = expanded
    return {"events": events}

def node_summarize(state: WFState) -> WFState:
    ev = state.get("events", [])
    lang = state.get("lang", "ko")
    summary = "일정이 없습니다." if lang == "ko" else "No events."
    if ev:
        summary = summarize_events(ev, lang)
    return {"summary": summary}

def node_slack(state: WFState) -> WFState:
    res = post_to_slack(state.get("summary",""))
    return {"slack_result": res}

def node_notion(state: WFState) -> WFState:
    lang = state.get("lang", "ko")
    res = write_to_notion(state.get("summary",""), lang=lang)
    return {"notion_result": res}

def should_do_calendar(state: WFState) -> Literal["calendar","end"]:
    if state.get("plan", {}).get("action") == "calendar_summary":
        return "calendar"
    return "end"

def should_post_slack(state: WFState) -> Literal["slack","skip_slack"]:
    if state.get("plan", {}).get("params", {}).get("post_to_slack", False):
        return "slack"
    return "skip_slack"

def should_write_notion(state: WFState) -> Literal["notion","end"]:
    if state.get("plan", {}).get("params", {}).get("write_to_notion", False):
        return "notion"
    return "end"

# Build graph
graph = StateGraph(WFState)
graph.add_node("plan_node", node_plan)
graph.add_node("calendar", node_calendar)
graph.add_node("summarize", node_summarize)
graph.add_node("slack", node_slack)
graph.add_node("notion", node_notion)

graph.set_entry_point("plan_node")
graph.add_conditional_edges("plan_node", should_do_calendar, {"calendar":"calendar", "end": END})
graph.add_edge("calendar", "summarize")
graph.add_conditional_edges("summarize", should_post_slack, {"slack":"slack", "skip_slack":"notion"})
graph.add_conditional_edges("slack", should_write_notion, {"notion":"notion", "end": END})
graph.add_edge("notion", END)

memory = MemorySaver()
app_graph = graph.compile(checkpointer=memory)

def run_once(nl_command: str):
    """
    Execute one-shot workflow from a natural language command.
    Example:
      run_once("내 일정 요약해서 슬랙에 보내고 노션에도 기록해")
    """
    state = {"user_text": nl_command}
    config = {"configurable": {"thread_id": "main"}}
    final = app_graph.invoke(state, config=config)
    rprint("\n[bold cyan]=== 결과 ===[/bold cyan]")
    rprint({"plan": final.get("plan"),
            "events_count": len(final.get("events", [])),
            "slack": final.get("slack_result"),
            "notion": final.get("notion_result")})
    lang = final.get("lang", "ko")
    header = "요약" if lang == "ko" else "Summary"
    none_text = "(없음)" if lang == "ko" else "(none)"
    rprint(f"\n[bold]{header}:[/bold]\n" + final.get("summary", none_text))

if __name__ == "__main__":
    import subprocess

    def is_llm_ready():
        try:
            resp = requests.get("http://127.0.0.1:8000/v1/models", timeout=2)
            return resp.status_code == 200
        except Exception:
            return False

    llm_proc = None
    if not is_llm_ready():
        rprint("[yellow]LLM server is not running. Starting it automatically...[/yellow]")
        llm_cmd = [
            "furiosa-llm", "serve", "furiosa-ai/Llama-3.1-8B-Instruct-FP8", "--devices", "npu:1"
        ]
        llm_proc = subprocess.Popen(llm_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # Wait for server to be ready
        for _ in range(60):
            if is_llm_ready():
                rprint("[green]LLM server is ready![/green]")
                break
            time.sleep(1)
        else:
            rprint("[red]LLM server did not start within 60 seconds. Exiting.[/red]")
            if llm_proc:
                llm_proc.terminate()
            exit(1)

    try:
        rprint("[green]AI Workflow Automator (LangGraph) – Local Demo[/green]")
        rprint("예) 오늘 일정 요약해서 슬랙에 보내고 노션에도 기록해 / e.g. Summarize today’s schedule and send it to Slack and Notion.")
        text = input("\n명령(Command) > ").strip()
        if not text:
            text = "오늘 일정 요약해서 슬랙에도 보내고 노션에도 기록해"
        run_once(text)
    finally:
        if llm_proc:
            llm_proc.terminate()