#!/usr/bin/env python3
"""
Copies today's WikiZenBot report into the wikizenbot-reports GitHub Pages
repo, rebuilds the index page listing all past reports, and pushes.

Usage: python3 publish_to_pages.py <report_path> <report_date YYYY-MM-DD>
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

PAGES_REPO = Path(os.environ.get("WIKIZENBOT_PAGES_REPO", Path.home() / "wikizenbot-reports"))
DOCS_DIR = PAGES_REPO / "docs"
REPORTS_DIR = DOCS_DIR / "reports"


def run(cmd, cwd):
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        print("Command failed: {}\n{}\n{}".format(" ".join(cmd), result.stdout, result.stderr), file=sys.stderr)
    return result


def add_front_matter(text, title):
    return "---\nlayout: default\ntitle: \"{}\"\n---\n\n{}".format(title, text)


def build_index():
    dates = sorted((p.stem for p in REPORTS_DIR.glob("*.md")), reverse=True)
    lines = [
        "---",
        "layout: default",
        "title: \"WikiZenBot\"",
        "---",
        "",
        "<p align=\"center\"><img src=\"assets/logo.webp\" alt=\"WikiZenBot\" width=\"220\"></p>",
        "",
        "# WikiZenBot -- Wikipedia discussion digest",
        "",
        "Daily scan of Village Pump, admin noticeboards, open RfCs, and AfD logs, ranked by activity and summarized by Claude.",
        "",
        "## Reports",
        "",
    ]
    for d in dates:
        lines.append("- [{}](reports/{}.html)".format(d, d))
    (DOCS_DIR / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    if len(sys.argv) != 3:
        print("Usage: publish_to_pages.py <report_path> <report_date>", file=sys.stderr)
        sys.exit(1)
    report_path, report_date = Path(sys.argv[1]), sys.argv[2]

    if not PAGES_REPO.exists():
        print("Pages repo not found at {}; skipping publish.".format(PAGES_REPO), file=sys.stderr)
        sys.exit(1)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    text = report_path.read_text(encoding="utf-8")
    dated_page = REPORTS_DIR / "{}.md".format(report_date)
    dated_page.write_text(add_front_matter(text, "WikiZenBot -- {}".format(report_date)), encoding="utf-8")

    build_index()

    run(["git", "add", "-A"], cwd=PAGES_REPO)
    commit = run(["git", "commit", "-m", "Report for {}".format(report_date)], cwd=PAGES_REPO)
    if commit.returncode != 0 and "nothing to commit" in (commit.stdout + commit.stderr):
        print("Nothing new to publish.", file=sys.stderr)
        return
    push = run(["git", "push"], cwd=PAGES_REPO)
    if push.returncode == 0:
        print("Published.", file=sys.stderr)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
