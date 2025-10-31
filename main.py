import os, json, re, time, requests
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional, Literal
from pathlib import Path

from dotenv import load_dotenv
from rich import print as rprint

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from ics import Calendar

BASE = Path(__file__).parent
OUT = BASE / "out"
OUT.mkdir(exist_ok=True)

load_dotenv()
FURIOSA_ENDPOINT = os.getenv("FURIOSA_ENDPOINT", "http://127.0.0.1:8000/v1/chat/completions")
FURIOSA_MODEL = os.getenv("FURIOSA_MODEL")
ICAL_URL = os.getenv("ICAL_URL")

SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "").strip()
NOTION_TOKEN = os.getenv("NOTION_TOKEN", "").strip()
NOTION_PAGE_ID = os.getenv("NOTION_PAGE_ID", "").strip()
NOTION_DB_ID = os.getenv("NOTION_DB_ID", "").strip()
NOTION_TITLE_PROP = "Title"
NOTION_DATE_PROP = "Date"

UUID32_RE = re.compile(r"([0-9a-fA-F]{32})")
def _canonical_uuid(s: str) -> str:
    m = UUID32_RE.search(s or "")
    if not m:
        return s
    raw = m.group(1).lower()
    return f"{raw[0:8]}-{raw[8:12]}-{raw[12:16]}-{raw[16:20]}-{raw[20:32]}"

def _normalize_notion_ids():
    global NOTION_PAGE_ID, NOTION_DB_ID
    if NOTION_PAGE_ID:
        canon = _canonical_uuid(NOTION_PAGE_ID)
        if canon != NOTION_PAGE_ID:
            rprint(f"[dim]Notion Page ID 정규화: {canon}[/dim]")
            NOTION_PAGE_ID = canon
    if NOTION_DB_ID:
        canon = _canonical_uuid(NOTION_DB_ID)
        if canon != NOTION_DB_ID:
            rprint(f"[dim]Notion DB ID 정규화: {canon}[/dim]")
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
                rprint(f"[dim]자동 선택된 모델: {FURIOSA_MODEL}[/dim]")
                return FURIOSA_MODEL
    except Exception as e:
        rprint(f"[red]모델 자동 선택 실패: {e}[/red]")
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
            rprint(f"[red]LLM 요청 실패 {r.status_code}: {r.text[:300]}[/red]")
            raise requests.HTTPError(f"LLM error {r.status_code}")
        data = r.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        rprint(f"[yellow]LLM 호출 오류, 폴백 사용: {e}[/yellow]")
        return "{\n  \"action\": \"noop\",\n  \"params\": {\n    \"days\": 1,\n    \"post_to_slack\": false,\n    \"write_to_notion\": false\n  }\n}"

JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
def extract_json(s: str) -> Dict[str, Any]:
    m = JSON_RE.search(s)
    if not m:
        raise ValueError("No JSON found")
    return json.loads(m.group(0))

def fetch_events_from_ical(days: int = 1, start: Optional[datetime] = None) -> List[Dict[str, Any]]:
    if not ICAL_URL:
        return []
    cal_text = None
    ics_path = Path(ICAL_URL)
    if ics_path.exists():
        try:
            cal_text = ics_path.read_text(encoding="utf-8")
        except Exception as e:
            rprint(f"[yellow].ics 로컬 파일 읽기 실패: {e}[/yellow]")
            return []
    else:
        try:
            resp = requests.get(ICAL_URL, timeout=20)
            resp.raise_for_status()
            cal_text = resp.text
        except Exception as e:
            rprint(f"[yellow]ICS 가져오기 실패: {e}[/yellow]")
            return []
    try:
        cal = Calendar(cal_text)
    except Exception as e:
        rprint(f"[yellow]ICS 파싱 실패: {e}[/yellow]")
        return []

    tz = timezone.utc
    now = start or datetime.now(tz)
    end = now + timedelta(days=days)

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
        if edt < now or bdt > end:
            continue
        out.append({
            "title": ev.name or "(제목 없음)",
            "start": bdt.astimezone(tz).isoformat(),
            "end": edt.astimezone(tz).isoformat(),
            "location": getattr(ev, "location", "") or ""
        })
    return sorted(out, key=lambda x: x["start"])

def summarize_events(events: List[Dict[str, Any]], lang: str) -> str:
    raw = json.dumps(events, ensure_ascii=False, indent=2)
    if lang == "ko":
        instr = (
            "아래 일정 JSON을 한국어로 간결하게 요약해 주세요.\n"
            "- 오늘의 주요 일정 헤더 1줄\n"
            "- 항목별로 • 불릿 3~8개\n"
            "- 시간(로컬), 제목, 장소(있으면) 포함\n"
            "- 문장은 짧고 명확하게"
        )
    else:
        instr = (
            "Summarize the schedule JSON in concise English.\n"
            "- First line: short header for today's key schedule\n"
            "- Bullet points (•) 3-8 items\n"
            "- Include local time, title, location(if any)\n"
            "- Sentences short and clear"
        )
    prompt = f"{instr}\n\nJSON:\n{raw}\n"
    return llm_chat(prompt, max_tokens=400, temperature=0.2).strip()

def post_to_slack(text: str) -> Dict[str, Any]:
    if not SLACK_WEBHOOK_URL:
        # local fallback
        (OUT / "slack_message.txt").write_text(text, encoding="utf-8")
        return {"ok": False, "note": "SLACK_WEBHOOK_URL 없음 → out/slack_message.txt로 저장"}
    r = requests.post(SLACK_WEBHOOK_URL, json={"text": text}, timeout=15)
    ok = r.status_code in (200, 204)
    return {"ok": ok, "status": r.status_code, "text": r.text if not ok else "OK"}

def write_to_notion(text: str, lang: str = "ko") -> Dict[str, Any]:
    if not NOTION_TOKEN:
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        with open(OUT / "notion_log.md", "a", encoding="utf-8") as f:
            f.write(f"\n## {now}\n{text}\n")
        return {"ok": False, "note": "NOTION_TOKEN 없음 → out/notion_log.md"}

    url = "https://api.notion.com/v1/pages"
    headers = {
        "Authorization": f"Bearer {NOTION_TOKEN}",
        "Notion-Version": "2022-06-28",
        "Content-Type": "application/json"
    }
    if lang == "ko":
        title_str = datetime.now().strftime("%Y-%m-%d 일정 요약 보고")
    else:
        title_str = datetime.now().strftime("%Y-%m-%d Schedule Summary")
    today_iso = datetime.now().date().isoformat()

    if NOTION_DB_ID:
        # Create in database
        properties = {
            NOTION_TITLE_PROP: {"title":[{"text":{"content": title_str}}]},
        }
        # Add date property (date range single day)
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
    else:
        if not NOTION_PAGE_ID:
            now = datetime.now().strftime("%Y-%m-%d %H:%M")
            with open(OUT / "notion_log.md", "a", encoding="utf-8") as f:
                f.write(f"\n## {now}\n{text}\n")
            return {"ok": False, "note": "NOTION_PAGE_ID/DB_ID 없음 → out/notion_log.md"}
        payload = {
            "parent": {"type":"page_id","page_id": NOTION_PAGE_ID},
            "properties": {
                "title": {"title":[{"text":{"content": title_str}}]}
            },
            "children": [
                {
                    "object":"block","type":"paragraph",
                    "paragraph":{"rich_text":[{"type":"text","text":{"content": text}}]}
                }
            ]
        }
    r = requests.post(url, headers=headers, json=payload, timeout=20)
    ok = r.status_code in (200, 201)
    if not ok:
        rprint(f"[yellow]Notion 생성 실패 {r.status_code}: {r.text[:200]}[/yellow]")
        # local fallback copy
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        with open(OUT / "notion_log.md", "a", encoding="utf-8") as f:
            f.write(f"\n## {now} (Notion 실패)\n{text}\n")
    return {"ok": ok, "status": r.status_code, "text": r.text if not ok else "OK"}

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
            state['plan']['params']['days'] = 30  # reflect adjusted window
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
    # Provide configuration with thread_id for checkpointer
    config = {"configurable": {"thread_id": "main"}}
    final = app_graph.invoke(state, config=config)
    # Pretty print result
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
    rprint("[green]AI Workflow Automator (LangGraph) – Local Demo[/green]")
    rprint("예) 오늘 일정 요약해서 슬랙에 보내고 노션에도 기록해 / e.g. Summarize today’s schedule and send it to Slack and Notion.")
    text = input("\n명령(Command) > ").strip()
    if not text:
        text = "오늘 일정 요약해서 슬랙에도 보내고 노션에도 기록해"
    run_once(text)