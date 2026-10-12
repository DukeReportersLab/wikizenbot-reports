#!/usr/bin/env python3
"""
WikiZenBot: surfaces the most interesting/urgent discussions happening
right now on English Wikipedia's community discussion pages.

Sources covered:
  - Village Pump (policy, technical, proposals, idea lab, miscellaneous)
  - Administrator noticeboards (ANI, AN)
  - Requests for Comment (RfC) currently open, via {{rfc}} transclusions
  - Articles for Deletion (AfD) daily logs

Ranking: heuristics first (reply count, unique participants, recency,
urgency keywords, thread size) narrow everything down to the top
candidates. If ANTHROPIC_API_KEY is set and --summarize is passed, the
top candidates are additionally sent to Claude for a short editorial
summary of what's most interesting/urgent and why.

Stdlib only -- no pip install required. Needs Python 3.8+.
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

API_URL = "https://en.wikipedia.org/w/api.php"
CONTACT = os.environ.get("WIKIZENBOT_CONTACT", "your-email@example.com")
USER_AGENT = "WikiZenBot/0.1 (personal discussion tracker; contact: {})".format(CONTACT)

MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

TIMESTAMP_RE = re.compile(
    r"\d{2}:\d{2}, \d{1,2} (?:%s) \d{4} \(UTC\)" % "|".join(MONTHS)
)
USER_LINK_RE = re.compile(r"\[\[User(?: talk)?:([^\|\]#/]+)", re.IGNORECASE)
# (=+) is greedy, so it consumes the full run of '=' before backtracking,
# which is what makes \1 land on a matching-length run at the close --
# a bounded quantifier like {2,3} would truncate a level-4+ heading and
# bleed stray '=' characters into the captured title.
ANY_HEADING_RE = re.compile(r"^(=+)\s*(.+?)\s*\1\s*$", re.MULTILINE)
RFC_TEMPLATE_RE = re.compile(r"\{\{\s*rfc\b", re.IGNORECASE)
AFD_TRANSCLUSION_RE = re.compile(
    r"\{\{\s*Wikipedia:Articles for deletion/([^}]+?)\s*\}\}", re.IGNORECASE
)
# Archive-bot markers like "<!-- [[User:DoNotArchiveUntil]] 12:01, 7 October
# 2026 (UTC) -->" carry a deliberately future timestamp and a fake "user"
# link; strip HTML comments before any signature/participant parsing so
# they aren't mistaken for a real (and impossibly recent) comment.
HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
TEMPLATE_RE = re.compile(r"\{\{[^{}]*\}\}")
PIPED_LINK_RE = re.compile(r"\[\[[^\]|]+\|([^\]]+)\]\]")
PLAIN_LINK_RE = re.compile(r"\[\[([^\]]+)\]\]")
EXTERNAL_LINK_TEXT_RE = re.compile(r"\[https?://\S+\s+([^\]]+)\]")
EXTERNAL_LINK_BARE_RE = re.compile(r"\[https?://\S+\]")
HTML_TAG_RE = re.compile(r"<[^>]+>")
QUOTE_MARKUP_RE = re.compile(r"'{2,5}")
LEADING_WIKI_SYNTAX_RE = re.compile(r"^[ \t]*[:*#=]+[ \t]*", re.MULTILINE)


def clean_snippet_text(text, limit=280):
    """Wikitext -> plain text for a display snippet: drop templates (run a
    few passes to unwrap one or two levels of nesting), unwrap [[links]] to
    their display text, drop external-link brackets and HTML tags, strip
    ''italic''/'''bold''' markup, and drop leading list/heading punctuation
    (":", "*", "#", "=") that wikitext uses for indentation/structure."""
    for _ in range(3):
        stripped = TEMPLATE_RE.sub("", text)
        if stripped == text:
            break
        text = stripped
    text = PIPED_LINK_RE.sub(r"\1", text)
    text = PLAIN_LINK_RE.sub(r"\1", text)
    text = EXTERNAL_LINK_TEXT_RE.sub(r"\1", text)
    text = EXTERNAL_LINK_BARE_RE.sub("", text)
    text = HTML_TAG_RE.sub("", text)
    text = QUOTE_MARKUP_RE.sub("", text)
    text = LEADING_WIKI_SYNTAX_RE.sub("", text)
    text = re.sub(r"={2,}", "", text)  # stray heading '===' markers, leading or trailing
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    truncated = text[:limit]
    last_space = truncated.rfind(" ")
    if last_space > limit * 0.6:  # only trim to the word boundary if it's not too far back
        truncated = truncated[:last_space]
    return truncated.rstrip(" ,;:.-") + "…"

URGENCY_KEYWORDS = [
    "urgent", "block", "blocked", "blocking", "3rr", "edit war", "edit-war",
    "sock", "sockpuppet", "vandal", "vandalism", "ban", "banned", "delete",
    "deletion", "speedy", "close", "closing", "closure", "deadline",
    "emergency", "blp violation", "legal threat", "outing", "harassment",
    "arbcom", "topic ban", "site ban", "checkuser", "oversight",
]

VILLAGE_PUMP_PAGES = [
    "Wikipedia:Village pump (policy)",
    "Wikipedia:Village pump (technical)",
    "Wikipedia:Village pump (proposals)",
    "Wikipedia:Village pump (idea lab)",
    "Wikipedia:Village pump (miscellaneous)",
]

NOTICEBOARD_PAGES = [
    "Wikipedia:Administrators' noticeboard/Incidents",
    "Wikipedia:Administrators' noticeboard",
]


# --------------------------------------------------------------------------
# MediaWiki API helpers
# --------------------------------------------------------------------------

def api_get(params, retries=3):
    params = dict(params)
    params["format"] = "json"
    url = API_URL + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last_err = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError) as e:
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError("MediaWiki API request failed after retries: {}".format(last_err))


def get_wikitext(title):
    """Fetch raw wikitext for a single page. Returns None if missing."""
    data = api_get({
        "action": "query", "prop": "revisions", "rvslots": "main",
        "rvprop": "content", "titles": title, "formatversion": 2,
    })
    pages = data.get("query", {}).get("pages", [])
    if not pages or pages[0].get("missing"):
        return None
    revs = pages[0].get("revisions")
    if not revs:
        return None
    return revs[0]["slots"]["main"]["content"]


def get_wikitext_multi(titles):
    """Batch-fetch wikitext for many titles (chunks of 50). Returns {title: content}."""
    result = {}
    titles = list(dict.fromkeys(titles))  # dedupe, keep order
    for i in range(0, len(titles), 50):
        batch = titles[i:i + 50]
        data = api_get({
            "action": "query", "prop": "revisions", "rvslots": "main",
            "rvprop": "content", "titles": "|".join(batch), "formatversion": 2,
        })
        for page in data.get("query", {}).get("pages", []):
            if page.get("missing"):
                continue
            revs = page.get("revisions")
            if revs:
                result[page["title"]] = revs[0]["slots"]["main"]["content"]
    return result


def embedded_in(template_title, namespaces):
    """List pages that transclude a given template, across given namespace ids."""
    titles = []
    cont = {}
    while True:
        params = {
            "action": "query", "list": "embeddedin", "eititle": template_title,
            "eilimit": "500", "einamespace": "|".join(str(n) for n in namespaces),
            "formatversion": 2,
        }
        params.update(cont)
        data = api_get(params)
        for page in data.get("query", {}).get("embeddedin", []):
            titles.append(page["title"])
        if "continue" in data:
            cont = data["continue"]
        else:
            break
    return titles


def now_utc():
    """Naive UTC datetime (kept naive so it subtracts cleanly against parsed
    wiki timestamps, which have no tzinfo)."""
    return datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)


def page_url(title, section_anchor=None):
    frag = ""
    if section_anchor:
        # Some headings carry a hidden HTML anchor, e.g. '<span id="X"></span>Title';
        # MediaWiki builds the section's real anchor from the visible text only.
        visible = HTML_TAG_RE.sub("", section_anchor).strip()
        if visible:
            frag = "#" + urllib.parse.quote(visible.replace(" ", "_"))
    return "https://en.wikipedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_")) + frag


# --------------------------------------------------------------------------
# Wikitext parsing helpers
# --------------------------------------------------------------------------

def all_headings(wikitext):
    """Return list of (level, heading_text, start_pos) for every heading, in order."""
    out = []
    for m in ANY_HEADING_RE.finditer(wikitext):
        level = len(m.group(1))
        out.append((level, m.group(2), m.start()))
    out.sort(key=lambda t: t[2])
    return out


def split_level2_sections(wikitext):
    """Split a page's wikitext into (heading, body) chunks on top-level '== heading =='
    threads. Deeper subheadings (===, ====, ...) stay inside their parent's body."""
    headings = all_headings(wikitext)
    top = [(heading, start) for level, heading, start in headings if level == 2]
    sections = []
    for idx, (heading, start) in enumerate(top):
        line_end = wikitext.find("\n", start)
        body_start = line_end + 1 if line_end != -1 else start
        body_end = top[idx + 1][1] if idx + 1 < len(top) else len(wikitext)
        sections.append((heading, wikitext[body_start:body_end]))
    return sections


def enclosing_section(wikitext, pos, headings):
    """Given a position in wikitext, find the (heading, body) of the section containing it."""
    containing = None
    for level, heading, start in headings:
        if start <= pos:
            containing = (level, heading, start)
        else:
            break
    if containing is None:
        return None, wikitext[:pos] + wikitext[pos:pos + 500]
    level, heading, start = containing
    end = len(wikitext)
    for lvl2, heading2, start2 in headings:
        if start2 > start and lvl2 <= level:
            end = start2
            break
    body_start = start
    for level_, heading_, s_ in headings:
        if s_ == start:
            # body begins after the heading line itself
            line_end = wikitext.find("\n", s_)
            body_start = line_end + 1 if line_end != -1 else s_
            break
    return heading, wikitext[body_start:end]


# --------------------------------------------------------------------------
# Heuristic scoring
# --------------------------------------------------------------------------

def parse_timestamps(body):
    out = []
    for match in TIMESTAMP_RE.findall(body):
        try:
            out.append(datetime.datetime.strptime(match, "%H:%M, %d %B %Y (UTC)"))
        except ValueError:
            continue
    return out


def extract_participants(body):
    names = set()
    for m in USER_LINK_RE.finditer(body):
        name = m.group(1).strip()
        if name:
            names.add(name.lower())
    return names


def count_urgency_hits(text):
    text_low = text.lower()
    return sum(1 for kw in URGENCY_KEYWORDS if kw in text_low)


def compute_score(heading, body, now, base_bonus=0.0):
    timestamps = parse_timestamps(body)
    comments = len(timestamps)
    participants = extract_participants(body)
    size = len(body)
    urgency_hits = count_urgency_hits(heading + " " + body)

    if timestamps:
        hours_since = (now - max(timestamps)).total_seconds() / 3600.0
    else:
        hours_since = None

    comment_score = min(comments, 40) / 40.0 * 100
    participant_score = min(len(participants), 15) / 15.0 * 100
    if hours_since is None:
        recency_score = 0.0
    else:
        recency_score = max(0.0, min(100.0, 100.0 - (hours_since / 96.0) * 100.0))
    urgency_score = min(urgency_hits, 5) / 5.0 * 100
    size_score = min(size, 8000) / 8000.0 * 100

    total = (
        comment_score * 0.30
        + participant_score * 0.25
        + recency_score * 0.25
        + urgency_score * 0.15
        + size_score * 0.05
    ) + base_bonus

    breakdown = {
        "comments": comments,
        "participants": len(participants),
        "hours_since_last_comment": round(hours_since, 1) if hours_since is not None else None,
        "size_chars": size,
        "urgency_keyword_hits": urgency_hits,
        "score": round(total, 1),
    }
    return total, breakdown


# --------------------------------------------------------------------------
# Source gatherers -- each returns a list of candidate dicts
# --------------------------------------------------------------------------

def candidate(source, page_title, heading, body, now, base_bonus=0.0, extra_url=None):
    body = HTML_COMMENT_RE.sub("", body)
    score, breakdown = compute_score(heading, body, now, base_bonus=base_bonus)
    snippet = clean_snippet_text(body)
    return {
        "source": source,
        "page": page_title,
        "heading": heading,
        "url": extra_url or page_url(page_title, heading),
        "score": score,
        "breakdown": breakdown,
        "snippet": snippet,
        "body": body,
    }


def gather_village_pump(now):
    results = []
    contents = get_wikitext_multi(VILLAGE_PUMP_PAGES)
    for title, wikitext in contents.items():
        for heading, body in split_level2_sections(wikitext):
            if not body.strip():
                continue
            results.append(candidate("Village Pump", title, heading, body, now))
    return results


def gather_noticeboards(now):
    results = []
    contents = get_wikitext_multi(NOTICEBOARD_PAGES)
    for title, wikitext in contents.items():
        for heading, body in split_level2_sections(wikitext):
            if not body.strip():
                continue
            # ANI/AN threads are inherently higher-stakes; small fixed bonus.
            results.append(candidate("Noticeboard", title, heading, body, now, base_bonus=8.0))
    return results


def gather_rfcs(now, max_pages=150):
    results = []
    titles = embedded_in("Template:Rfc", namespaces=[1, 3, 4, 5])
    titles = titles[:max_pages]
    contents = get_wikitext_multi(titles)
    for title, wikitext in contents.items():
        headings = all_headings(wikitext)
        seen_headings = set()
        for m in RFC_TEMPLATE_RE.finditer(wikitext):
            heading, body = enclosing_section(wikitext, m.start(), headings)
            key = heading or "(lead)"
            if key in seen_headings:
                continue
            seen_headings.add(key)
            if not body.strip():
                continue
            results.append(candidate(
                "RfC", title, heading or "(untitled RfC)", body, now, base_bonus=10.0
            ))
    return results


def gather_afd(now, days_back=2, max_debates=400):
    results = []
    today = now_utc().date()
    log_titles = []
    for d in range(days_back):
        day = today - datetime.timedelta(days=d)
        log_titles.append(
            "Wikipedia:Articles for deletion/Log/{:%Y %B }{}".format(day, day.day)
        )
    logs = get_wikitext_multi(log_titles)
    debate_titles = []
    for wikitext in logs.values():
        for m in AFD_TRANSCLUSION_RE.finditer(wikitext):
            debate_titles.append("Wikipedia:Articles for deletion/" + m.group(1).strip())
    debate_titles = list(dict.fromkeys(debate_titles))[:max_debates]
    contents = get_wikitext_multi(debate_titles)
    for title, wikitext in contents.items():
        article_name = title.split("/", 1)[1] if "/" in title else title
        results.append(candidate(
            "AfD", title, article_name, wikitext, now,
            base_bonus=5.0, extra_url=page_url(title),
        ))
    return results


# --------------------------------------------------------------------------
# Optional LLM summarization step
# --------------------------------------------------------------------------

def summarize_with_claude(candidates, model="claude-sonnet-5", max_tokens=4096):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None, "ANTHROPIC_API_KEY not set; skipping LLM summary."

    parts = []
    for c in candidates:
        parts.append(
            "### [{source}] {heading}\nPage: {page}\nURL: {url}\n"
            "Stats: {comments} comments, {participants} participants, "
            "last activity {hours}h ago, score {score}\n"
            "Excerpt: {snippet}\n".format(
                source=c["source"], heading=c["heading"], page=c["page"], url=c["url"],
                comments=c["breakdown"]["comments"],
                participants=c["breakdown"]["participants"],
                hours=c["breakdown"]["hours_since_last_comment"],
                score=c["breakdown"]["score"],
                snippet=c["snippet"],
            )
        )
    user_content = (
        "Here are the top-ranked live English Wikipedia discussions, "
        "pre-filtered by heuristics (reply count, participants, recency, "
        "urgency keywords). Write a short editorial briefing: group by "
        "theme if useful, call out what's most interesting or urgent and "
        "why, and keep it skimmable. IMPORTANT: every specific discussion "
        "you name must be a Markdown link using its URL given below, e.g. "
        "'[RfC on late signatures](https://en.wikipedia.org/wiki/...)' -- "
        "never mention one by name without linking it.\n\n" + "\n".join(parts)
    )

    payload = json.dumps({
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": user_content}],
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={
            "content-type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        text = "".join(block.get("text", "") for block in data.get("content", []))
        usage = data.get("usage", {})
        print(
            "Claude usage: {} input tokens, {} output tokens".format(
                usage.get("input_tokens"), usage.get("output_tokens")
            ),
            file=sys.stderr,
        )
        return text, None
    except urllib.error.HTTPError as e:
        return None, "Claude API error {}: {}".format(e.code, e.read().decode("utf-8", "replace"))
    except Exception as e:
        return None, "Claude API call failed: {}".format(e)


# --------------------------------------------------------------------------
# Report rendering
# --------------------------------------------------------------------------

def _format_entry(c, with_snippet=True):
    # Everything on one line: Markdown collapses single '\n's inside a list
    # item into a space, so multi-line bullets (heading / url / snippet on
    # separate lines) render as one run-on jumble in most renderers. Putting
    # the link on the heading itself also guarantees every item mentioned
    # carries its own clickable link, rather than relying on a separate URL
    # line that's easy to omit or detach.
    b = c["breakdown"]
    hours = b["hours_since_last_comment"]
    hours_str = "{}h ago".format(hours) if hours is not None else "no timestamp found"
    line = "- **[{score}] [{source}] [{heading}]({url})** ({page}) -- {comments}c/{participants}p, {hours}".format(
        score=b["score"], source=c["source"], heading=c["heading"], url=c["url"],
        page=c["page"], comments=b["comments"], participants=b["participants"], hours=hours_str,
    )
    if with_snippet and c["snippet"]:
        line += " -- " + c["snippet"]
    return line


def render_report(all_candidates, per_source, top_n, summary_text, summary_error):
    now = now_utc()
    lines = []
    lines.append("# WikiZenBot report -- {} UTC".format(now.strftime("%Y-%m-%d %H:%M")))
    lines.append("")
    if summary_text:
        lines.append("## Editorial summary (Claude)")
        lines.append("")
        lines.append(summary_text.strip())
        lines.append("")
    elif summary_error:
        lines.append("_LLM summary skipped: {}_".format(summary_error))
        lines.append("")

    lines.append("## Top discussions overall")
    lines.append("")
    top_overall = sorted(all_candidates, key=lambda c: c["score"], reverse=True)[:top_n]
    for c in top_overall:
        lines.append(_format_entry(c, with_snippet=True))
    lines.append("")

    lines.append("## By source")
    lines.append("")
    by_source = {}
    for c in all_candidates:
        by_source.setdefault(c["source"], []).append(c)
    for source, items in by_source.items():
        items.sort(key=lambda c: c["score"], reverse=True)
        lines.append("### {} ({} threads seen)".format(source, len(items)))
        lines.append("")
        for c in items[:per_source]:
            lines.append(_format_entry(c, with_snippet=False))
        lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-source", type=int, default=8, help="Threads to list per source (default: 8)")
    parser.add_argument("--top", type=int, default=20, help="Threads to list in the overall top section (default: 20)")
    parser.add_argument("--afd-days", type=int, default=2, help="How many days of AfD logs to scan (default: 2)")
    parser.add_argument("--rfc-max-pages", type=int, default=150, help="Max pages to scan for open RfCs (default: 150)")
    parser.add_argument("--summarize", action="store_true", help="Also generate an editorial summary via Claude (needs ANTHROPIC_API_KEY)")
    parser.add_argument("--out", default=None, help="Write the Markdown report to this file (default: print to stdout)")
    parser.add_argument("--sources", default="villagepump,noticeboards,rfc,afd",
                         help="Comma-separated subset of: villagepump,noticeboards,rfc,afd")
    args = parser.parse_args()

    now = now_utc()
    wanted = set(s.strip() for s in args.sources.split(","))

    all_candidates = []

    if "villagepump" in wanted:
        print("Fetching Village Pump...", file=sys.stderr)
        all_candidates += gather_village_pump(now)

    if "noticeboards" in wanted:
        print("Fetching noticeboards (ANI/AN)...", file=sys.stderr)
        all_candidates += gather_noticeboards(now)

    if "rfc" in wanted:
        print("Fetching open RfCs...", file=sys.stderr)
        all_candidates += gather_rfcs(now, max_pages=args.rfc_max_pages)

    if "afd" in wanted:
        print("Fetching AfD logs...", file=sys.stderr)
        all_candidates += gather_afd(now, days_back=args.afd_days)

    print("Scored {} threads.".format(len(all_candidates)), file=sys.stderr)

    summary_text, summary_error = None, None
    if args.summarize:
        top_for_llm = sorted(all_candidates, key=lambda c: c["score"], reverse=True)[:15]
        print("Summarizing top candidates with Claude...", file=sys.stderr)
        summary_text, summary_error = summarize_with_claude(top_for_llm)
        if summary_error:
            print(summary_error, file=sys.stderr)

    report = render_report(all_candidates, args.per_source, args.top, summary_text, summary_error)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(report)
        print("Wrote report to {}".format(args.out), file=sys.stderr)
    else:
        print(report)


if __name__ == "__main__":
    main()
