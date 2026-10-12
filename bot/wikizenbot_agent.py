#!/usr/bin/env python3
"""
WikiZenBot Agent: an actual tool-using LLM agent, not a fixed-formula script.

Unlike wikizenbot.py (which fetches everything up front and ranks it with a
hardcoded heuristic), this version gives Claude a small set of tools --
list_threads, list_open_rfcs, list_afd_debates, read_thread -- and lets the
model decide for itself which venues to scan, which threads look promising
enough to read in full, and how to weigh "interesting" against "urgent". The
model runs its own perceive-reason-act loop; the final Markdown report is
whatever it decides to write once it stops calling tools.

Requires: ANTHROPIC_API_KEY in the environment. Stdlib only otherwise --
reuses the MediaWiki fetching/parsing helpers from wikizenbot.py as tools.
"""

import argparse
import datetime
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import wikizenbot as wzb  # reuse fetching/parsing, not its scoring formula

MODEL = "claude-sonnet-5"
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"

REPORTS_DIR = Path(__file__).resolve().parent / "reports"
HISTORY_LOOKBACK = 14  # how many earlier reports to scan for repeat coverage
ONGOING_AFTER = 3  # featured this many times before => "Ongoing discussions" bullet

# Markdown links to en.wikipedia.org, allowing one level of parentheses inside
# the URL (e.g. "Village_pump_(policy)").
REPORT_LINK_RE = re.compile(
    r"\[[^\]]+\]\((https://en\.wikipedia\.org/(?:[^()\s]|\([^()\s]*\))+)\)"
)
AI_ACRONYMS_RE = re.compile(r"\bAIs?\b|\bLLMs?\b|\bGPT-?\d|LLMPROD|LLMPRV|NOLLM")
AI_PHRASES_RE = re.compile(
    r"artificial intelligence|large language model|chatgpt|openai|anthropic|"
    r"chatbot|machine[- ]generated|generative (?:ai|model|text)",
    re.IGNORECASE,
)

# normalized discussion URL -> {"count": earlier reports that linked it, "last": date}
COVERAGE_HISTORY = {}

SYSTEM_PROMPT = """You are WikiZenBot, an agent that monitors English \
Wikipedia's community discussion venues (Village Pump, administrator \
noticeboards, open Requests for Comment, and Articles for Deletion) and \
surfaces the discussions most worth one particular reader's attention \
right now.

ABOUT THE READER: they are writing a book about artificial intelligence, so \
discussions about AI -- large language models, AI-generated text or images \
on Wikipedia, AI policy and its enforcement, editors accused of misusing AI, \
and the like -- matter far more to them than other topics. Use a much LOWER \
bar for AI-related discussions than for everything else: include them even \
when activity is modest (a handful of comments is enough). Each listed \
thread carries an `ai_mentions` count of AI-related terms found in its text \
(omitted when zero), so a thread can be AI-related even if its heading never \
says so -- read any thread with a meaningful count before deciding.

You are not given a fixed scoring formula -- you decide what matters. Use \
the list_* tools to cheaply scan venues (they return activity signals: \
comment count, participant count, hours since last comment, size -- not \
full text). Use read_thread to pull the full text of anything that looks \
genuinely contentious, high-stakes, or time-sensitive before judging it. \
Don't read every thread in full -- triage with the summary stats first, \
then dig into the handful that seem to matter.

Weigh signals like: rapid back-and-forth, many distinct participants, \
disputes with real consequences (blocks, bans, article deletions, policy \
changes), and time pressure (something close to a decision or deadline). \
An AfD with 15 comments in the last few hours is a different kind of \
"urgent" than a slow-moving policy RfC with wide participation over days --
call out that distinction in your writing rather than collapsing everything
onto one scale.

Available Village Pump / noticeboard page titles you can pass to \
list_threads:
""" + "\n".join("  - " + p for p in wzb.VILLAGE_PUMP_PAGES + wzb.NOTICEBOARD_PAGES) + f"""

AVOIDING REPETITION: this report runs daily and the reader has seen the \
earlier ones. List results include `times_featured_before` (how many earlier \
reports already linked that discussion; omitted when zero) and \
`last_featured` (the date of the most recent one).
- If a discussion has been featured {ONGOING_AFTER} or more times before, do \
NOT write it up again. Give it one short bullet -- its title as a link, plus \
a few words on where things stand -- in a final section titled "Ongoing \
discussions". Exception: if something significant has happened since it last \
appeared (it was closed, a decision or sanction was announced, or the dispute \
took a major new turn), feature it normally and lead with what's new.
- If it has been featured fewer times, you may still include it, but lead \
with what has changed since it last appeared instead of re-explaining the \
background.

WRITING STYLE: spell out every acronym, initialism and Wikipedia shortcut \
the first time it appears in the report -- do not assume the reader knows \
them. For example: "Administrators' noticeboard/Incidents (ANI)", "Request \
for Comment (RfC)", "Articles for Deletion (AfD)", "large language model \
(LLM)", "biographies of living persons (BLP)", "conflict of interest (COI)". \
For policy shortcuts such as WP:BLUDGEON or WP:NOLLM, say in a few plain \
words what the policy or guideline is about rather than just citing the \
shortcut (for instance: "the guideline against dominating a discussion with \
repeated replies (WP:BLUDGEON)"). Prefer plain words over acronyms where you \
can. Everyday abbreviations such as AI, US and UK need no explanation. This \
applies everywhere, INCLUDING THE VISIBLE TEXT OF LINKS: thread titles often \
contain acronyms or shortcuts (for example "RfC", "COI", "WP:BITE", "CSAM"), \
so rewrite the link text to spell them out or paraphrase the title in plain \
words -- only the URL has to stay exact. Before you finish, check the whole \
report, including every "Ongoing discussions" bullet: an acronym that shows up \
without having been spelled out at its first appearance is a mistake.

When you're done investigating, stop calling tools and write the final \
report in Markdown, in this order: (1) a section on AI-related discussions; \
(2) the other discussions most worth attention, ranked, grouped by theme or \
urgency where that helps; (3) "Ongoing discussions". Omit any section that \
would be empty. Every discussion you mention anywhere in the report -- \
including each "Ongoing discussions" bullet -- must be a Markdown link to the \
original discussion, using the url from the tool results. For each featured \
discussion give which venue it's from and 1-2 sentences on why it matters. \
Call out patterns across discussions if you notice any."""

TOOLS = [
    {
        "name": "list_threads",
        "description": (
            "List top-level discussion threads on one Village Pump section or "
            "admin noticeboard, with activity signals (comment count, "
            "participant count, hours since last comment, size in chars). "
            "Use this to scan a venue before deciding what to read in full."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "page": {
                    "type": "string",
                    "enum": wzb.VILLAGE_PUMP_PAGES + wzb.NOTICEBOARD_PAGES,
                }
            },
            "required": ["page"],
        },
    },
    {
        "name": "list_open_rfcs",
        "description": (
            "List currently open Requests for Comment across Wikipedia "
            "(found via {{rfc}} template transclusions), each with activity "
            "signals. Use this to see what formal RfCs are in progress."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Max number of pages to scan for open RfCs (default 60)",
                }
            },
        },
    },
    {
        "name": "list_afd_debates",
        "description": (
            "List Articles for Deletion debates from the last N days' logs, "
            "each with activity signals."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "days_back": {
                    "type": "integer",
                    "description": "How many days of AfD logs to scan (default 1)",
                }
            },
        },
    },
    {
        "name": "read_thread",
        "description": (
            "Read the full text of one thread/debate you've already seen "
            "listed, to judge its content in depth. For Village Pump, "
            "noticeboard, or RfC threads, pass the page and the exact "
            "heading as returned by the list tools. For AfD debates, pass "
            "the debate page title (the 'title' field from list_afd_debates) "
            "and omit heading."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "page": {"type": "string"},
                "heading": {
                    "type": "string",
                    "description": "Exact section heading; omit for AfD debate pages",
                },
            },
            "required": ["page"],
        },
    },
]

MAX_TEXT_CHARS = 6000  # per read_thread call, to bound token usage


def _normalize_url(url):
    return urllib.parse.unquote(url).replace(" ", "_")


def load_coverage_history(today, lookback=HISTORY_LOOKBACK):
    """For each discussion URL, count how many of the last `lookback` earlier
    reports linked to it. `today`'s own report is excluded, so re-running the
    same day doesn't count itself."""
    dated = sorted(
        p for p in REPORTS_DIR.glob("*.md")
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", p.stem) and p.stem < today
    )
    history = {}
    for path in dated[-lookback:]:
        text = path.read_text(encoding="utf-8")
        for url in {_normalize_url(u) for u in REPORT_LINK_RE.findall(text)}:
            if "#" not in url and "Articles_for_deletion/" not in url:
                continue  # a link to a policy page, not to a discussion
            entry = history.setdefault(url, {"count": 0, "last": path.stem})
            entry["count"] += 1
            entry["last"] = path.stem
    return history


def _set_url(signals, url):
    signals["url"] = url
    entry = COVERAGE_HISTORY.get(_normalize_url(url))
    if entry:
        signals["times_featured_before"] = entry["count"]
        signals["last_featured"] = entry["last"]


def _thread_signals(heading, body, now):
    # Archive-bot markers like "<!-- [[User:DoNotArchiveUntil]] 12:01, 7
    # October 2026 (UTC) -->" carry a deliberately future timestamp and a
    # fake "user" link -- strip HTML comments first so they're not read as
    # a real (impossibly recent) comment.
    body = wzb.HTML_COMMENT_RE.sub("", body)
    timestamps = wzb.parse_timestamps(body)
    participants = wzb.extract_participants(body)
    hours = round((now - max(timestamps)).total_seconds() / 3600.0, 1) if timestamps else None
    text = heading + " " + body
    ai_mentions = len(AI_ACRONYMS_RE.findall(text)) + len(AI_PHRASES_RE.findall(text))
    signals = {
        "heading": heading,
        "comments": len(timestamps),
        "participants": len(participants),
        "hours_since_last_comment": hours,
        "size_chars": len(body),
    }
    if ai_mentions:
        signals["ai_mentions"] = ai_mentions
    return signals


def tool_list_threads(page):
    wikitext = wzb.get_wikitext(page)
    if wikitext is None:
        return {"error": "page not found: {}".format(page)}
    now = wzb.now_utc()
    threads = []
    for heading, body in wzb.split_level2_sections(wikitext):
        if not body.strip():
            continue
        signals = _thread_signals(heading, body, now)
        _set_url(signals, wzb.page_url(page, heading))
        threads.append(signals)
    return {"page": page, "threads": threads}


def tool_list_open_rfcs(limit=60):
    titles = wzb.embedded_in("Template:Rfc", namespaces=[1, 3, 4, 5])[:limit]
    contents = wzb.get_wikitext_multi(titles)
    now = wzb.now_utc()
    rfcs = []
    for title, wikitext in contents.items():
        headings = wzb.all_headings(wikitext)
        seen = set()
        for m in wzb.RFC_TEMPLATE_RE.finditer(wikitext):
            heading, body = wzb.enclosing_section(wikitext, m.start(), headings)
            key = heading or "(lead)"
            if key in seen or not body.strip():
                continue
            seen.add(key)
            signals = _thread_signals(heading or "(untitled RfC)", body, now)
            signals["page"] = title
            _set_url(signals, wzb.page_url(title, heading))
            rfcs.append(signals)
    return {"rfcs": rfcs}


def tool_list_afd_debates(days_back=1):
    today = wzb.now_utc().date()
    log_titles = []
    for d in range(days_back):
        day = today - datetime.timedelta(days=d)
        log_titles.append("Wikipedia:Articles for deletion/Log/{:%Y %B }{}".format(day, day.day))
    logs = wzb.get_wikitext_multi(log_titles)
    debate_titles = []
    for wikitext in logs.values():
        for m in wzb.AFD_TRANSCLUSION_RE.finditer(wikitext):
            debate_titles.append("Wikipedia:Articles for deletion/" + m.group(1).strip())
    debate_titles = list(dict.fromkeys(debate_titles))
    contents = wzb.get_wikitext_multi(debate_titles)
    now = wzb.now_utc()
    debates = []
    for title, wikitext in contents.items():
        article = title.split("/", 1)[1] if "/" in title else title
        signals = _thread_signals(article, wikitext, now)
        signals["article"] = signals.pop("heading")
        signals["title"] = title
        _set_url(signals, wzb.page_url(title))
        debates.append(signals)
    return {"afd_debates": debates}


def tool_read_thread(page, heading=None):
    wikitext = wzb.get_wikitext(page)
    if wikitext is None:
        return {"error": "page not found: {}".format(page)}
    if not heading:
        return {"page": page, "text": wikitext[:MAX_TEXT_CHARS]}
    for h, body in wzb.split_level2_sections(wikitext):
        if h == heading:
            return {"page": page, "heading": h, "text": body[:MAX_TEXT_CHARS]}
    headings = wzb.all_headings(wikitext)
    for level, h, start in headings:
        if h == heading:
            _, body = wzb.enclosing_section(wikitext, start + 1, headings)
            return {"page": page, "heading": h, "text": body[:MAX_TEXT_CHARS]}
    return {"error": "heading not found on page {}: {}".format(page, heading)}


TOOL_IMPLS = {
    "list_threads": lambda i: tool_list_threads(i["page"]),
    "list_open_rfcs": lambda i: tool_list_open_rfcs(i.get("limit", 60)),
    "list_afd_debates": lambda i: tool_list_afd_debates(i.get("days_back", 1)),
    "read_thread": lambda i: tool_read_thread(i["page"], i.get("heading")),
}


def call_claude(api_key, messages, tools=None, tool_choice=None, max_tokens=8192):
    body = {
        "model": MODEL,
        "max_tokens": max_tokens,
        "system": SYSTEM_PROMPT,
        "messages": messages,
    }
    if tools is not None:
        body["tools"] = tools
    if tool_choice is not None:
        body["tool_choice"] = tool_choice

    req = urllib.request.Request(
        ANTHROPIC_URL,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "content-type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": ANTHROPIC_VERSION,
        },
        method="POST",
    )
    # Retries guard against transient network flakiness -- notably, launchd
    # firing this job right as the Mac wakes from sleep, before Wi-Fi has
    # reconnected and DNS resolution works again (observed in practice as
    # an instant urllib.error.URLError / socket.gaierror on the first call).
    last_err = None
    for attempt, delay in enumerate([0, 5, 10, 20, 30]):
        if delay:
            time.sleep(delay)
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise RuntimeError("Claude API error {}: {}".format(e.code, e.read().decode("utf-8", "replace")))
        except urllib.error.URLError as e:
            last_err = e
    raise RuntimeError("Claude API unreachable after retries: {}".format(last_err))


def run_agent(api_key, max_steps=20, verbose=True):
    messages = [{
        "role": "user",
        "content": (
            "Find the most interesting and urgent discussions happening "
            "right now on English Wikipedia, and give me your final report."
        ),
    }]

    for step in range(max_steps):
        force_final = step == max_steps - 1
        # tool_choice is only meaningful (and only accepted by the API)
        # alongside a non-empty `tools` list, so the forced-final turn just
        # omits tools entirely -- the model can't call what isn't offered.
        response = call_claude(
            api_key, messages,
            tools=None if force_final else TOOLS,
        )

        content = response.get("content", [])
        messages.append({"role": "assistant", "content": content})

        for block in content:
            if block.get("type") == "text" and verbose:
                print("[agent] " + block["text"][:300], file=sys.stderr)
            elif block.get("type") == "tool_use" and verbose:
                print("[tool call] {}({})".format(block["name"], json.dumps(block["input"])), file=sys.stderr)

        if response.get("stop_reason") != "tool_use":
            return "".join(b.get("text", "") for b in content if b.get("type") == "text")

        tool_results = []
        for block in content:
            if block.get("type") != "tool_use":
                continue
            impl = TOOL_IMPLS.get(block["name"])
            try:
                result = impl(block["input"]) if impl else {"error": "unknown tool"}
            except Exception as e:
                result = {"error": str(e)}
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block["id"],
                "content": json.dumps(result),
            })
        messages.append({"role": "user", "content": tool_results})

    return "(hit max steps without a final answer)"


# Acronyms common enough on Wikipedia discussion pages that a missed expansion
# is worth fixing mechanically instead of trusting the model to remember (in
# testing it kept leaving "RfC" unexplained even when told not to).
ACRONYM_EXPANSIONS = {
    "RfC": "Request for Comment",
    "AfD": "Articles for Deletion",
    "XfD": "deletion discussions",
    "RfD": "Redirects for Discussion",
    "ANI": "Administrators' noticeboard/Incidents",
    "COI": "conflict of interest",
    "BLP": "biographies of living persons",
    "LLM": "large language model",
    "NPOV": "neutral point of view",
    "POV": "point of view",
    "GNG": "general notability guideline",
    "SPI": "sockpuppet investigation",
    "UPE": "undisclosed paid editing",
}
_LINK_DEST_RE = re.compile(r"\]\((?:[^()\s]|\([^()\s]*\))+\)")


def spell_out_acronyms(report):
    """Backstop for the prompt's spell-out rule: the first time a known
    acronym appears with no expansion before it (and isn't itself the
    parenthesized "(RfC)" half of an expansion), expand it in place."""
    # Hide link destinations so an acronym inside a URL (e.g. "#RfC%3A_...")
    # is never touched.
    stashed = []

    def stash(m):
        stashed.append(m.group(0))
        return "\x00{}\x00".format(len(stashed) - 1)

    text = _LINK_DEST_RE.sub(stash, report)

    for acro, expansion in ACRONYM_EXPANSIONS.items():
        m = re.search(r"\b" + acro + r"\b", text)
        if not m or text[:m.start()].endswith("("):
            continue
        before = text[:m.end()].lower().replace("’", "'")
        if expansion.lower() in before:
            continue
        lead = text[:m.start()].rstrip(" ")
        if not lead or lead[-1] in "[*.!?:\n-—":
            expansion = expansion[0].upper() + expansion[1:]
        text = text[:m.start()] + "{} ({})".format(expansion, acro) + text[m.end():]

    return re.sub(r"\x00(\d+)\x00", lambda m: stashed[int(m.group(1))], text)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-steps", type=int, default=20, help="Max agent tool-call turns before forcing a final answer (default: 20)")
    parser.add_argument("--out", default=None, help="Write the report to this file instead of stdout")
    parser.add_argument("--quiet", action="store_true", help="Don't print the agent's reasoning/tool-call trace to stderr")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ANTHROPIC_API_KEY is not set. This mode requires it -- an agent needs the model driving the loop.", file=sys.stderr)
        sys.exit(1)

    COVERAGE_HISTORY.update(load_coverage_history(datetime.date.today().isoformat()))
    if not args.quiet:
        print("Loaded coverage history: {} discussions featured in earlier reports.".format(len(COVERAGE_HISTORY)), file=sys.stderr)

    report = spell_out_acronyms(run_agent(api_key, max_steps=args.max_steps, verbose=not args.quiet))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(report)
        print("Wrote report to {}".format(args.out), file=sys.stderr)
    else:
        print(report)


if __name__ == "__main__":
    main()
