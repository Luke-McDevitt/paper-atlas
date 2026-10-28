# Paper Atlas kit -- turn your papers into a searchable, browsable atlas
<!-- doc: area=infra; status=current; title="Paper Atlas kit (2026-10-15): the standalone lab-mate kit -- install, add papers (PDFs, Zotero / BibTeX, lists), update (text, OCR, figures, citations, glossary, lookup, zip), customize atlas.toml; exported by tools/atlas.py export-kit" -->

This kit builds an **atlas** of the papers you work with: one web page you open in any browser, plus Excel and Word
copies and a portable `.zip` holding every PDF. You give it papers -- a folder of PDFs, a Zotero export, or a list of
DOIs -- and one command does the rest: it identifies each paper, checks it against arXiv / Crossref / OpenAlex, files
the PDF under a clean name, reads every page, pulls out every figure, maps who cites whom, builds a glossary from the
papers' own words, and lets you find papers by describing a task in plain words.

It is the machinery behind the RoboPI lab's BlueBoat world-model atlas (2,813 papers), packaged so you can run it on
your own papers. Nothing here phones home: the only network calls are to the paper registries (arXiv, Crossref,
DataCite, OpenAlex) and, for `add` with a paper list, to open-access copies of the papers.

---

## What you get

| | |
|---|---|
| **The atlas page** (`index/browser.html`; `atlas open`) | **Find** (type a task or a few terms: papers ranked, each with *why* it is listed), **Lexicon** (terms with verbatim quotes), **Papers** (status, PDF held or where to get it, code links, citations), **Citations** (who cites whom inside your atlas; click a paper), **Figures** (search every figure caption, open the image), Labs, Trends |
| **Office copies** | `index/ATLAS.xlsx` (every paper, term, fact), `index/LEXICON.docx`, `index/PAPERS.docx` (needs [pandoc](https://pandoc.org)) |
| **The zip** | `<short_name>_<date>.zip` beside your project: unzip anywhere, open `atlas.html`, click a paper to open its PDF |
| **Search from the command line** | `atlas find "sonar odometry without a DVL"` -- or from Python, `from paper_find import find` |

---

## Install (once)

1. **Python 3.11 or newer** ([python.org](https://www.python.org/downloads/); on Windows tick *Add python.exe to PATH*).
2. Unzip this kit anywhere (e.g. `Documents\atlaskit`) -- or get the newest version from GitHub:
   `git clone https://github.com/Luke-McDevitt/paper-atlas`.
3. In that folder run **`setup.cmd`** (Windows) or **`./setup.sh`** (macOS / Linux). It makes a private Python
   environment in `.venv` and installs what the kit needs. Add the optional parts on the same line:
   * `setup.cmd gpu` -- dense search and figure detection on an NVIDIA GPU. It installs PyTorch's CUDA build itself
     (the newest your NVIDIA driver runs, from download.pytorch.org -- a CUDA torch you installed yourself is kept),
     then the rest held to that torch, and ends with one line saying whether a CUDA operation ran. Without an NVIDIA
     GPU it installs the CPU build, and the kit uses a small CPU model and a model-free figure finder. The first
     `update` with a GPU downloads the two models from Hugging Face (about 2.3 GB, once); on an 8 GB RTX 4060 it
     used at most about 3.5 GB of GPU memory.
   * `setup.cmd ocr` -- reads scanned PDFs (RapidOCR, CPU).
4. Check it: **`atlas selftest`** (under a minute; builds a three-paper atlas offline -- text, figures, the page, the
   zip -- and checks each).

Optional, recommended:
* **An OpenAlex key** (free: openalex.org -> sign in -> settings -> API key): set the environment variable
  `OPENALEX_API_KEY`. Without one OpenAlex answers about 1,000 requests a day, enough for a few hundred papers.
* **Your e-mail for the registries' polite pool**: set `ATLAS_CONTACT_EMAIL` (it goes in the User-Agent of registry
  requests; the kit sends none if you do not set it).
* **pandoc** on your PATH for the Word copies.

In the commands below `atlas` means `atlas.cmd` (Windows) or `./atlas.sh` (macOS / Linux) in the kit folder -- and
`atlas new` puts the same pair in each project folder, so inside a project `atlas ...` runs the kit on THAT project
(moved the kit? run `atlas new <project>` from its new place to re-point them; atlas.toml is never touched).
PowerShell runs a command in the current folder only as `.\atlas`.

---

## The app: buttons instead of commands

Never opened a terminal? **Double-click `atlas-app.cmd`** (Windows) in the kit folder -- or run `./atlas-app.sh`
(macOS / Linux). The first time on a computer it runs the setup above for you (a few minutes); after that it opens
your atlas in a window of its own (Edge or Chrome; else your default browser), with a toolbar:

| button | what it does (the command it runs for you) |
|---|---|
| *the project's name* | open another atlas project (your recent ones are listed), or make a new one -- a folder and a title (`atlas new`) |
| **Add papers** | a folder of PDFs (the project's `inbox` is one click), files or a Zotero / BibTeX export, or a pasted list of DOIs / arXiv ids / titles; optionally a topic (`atlas add`) |
| **Update** | builds or refreshes the atlas; *force* re-runs every stage, *offline* skips the network ones (`atlas update`) |
| **Status** | what the project holds and what is out of date (`atlas status`) |
| **Discover** | describe a task: the papers you have for it and the ones out there; then read the report, the closed papers' links, and bring the new ones in -- optionally as a section (`atlas discover`) |
| **Make zip** | the portable copy, with or without the PDFs; *Show the folder* opens where it is (`atlas zip`) |
| **Settings** | your **OpenAlex API key**, your **contact e-mail** and **GPU** on / off |

One job runs at a time; its output streams into a panel at the bottom (**Cancel** stops it), and the atlas reloads
when it ends. Each *Browse...* button opens your system's own file dialog (on Linux it needs `python3-tk`; without it,
paste the path). `atlas app` in a project folder opens the app on that project.

**Your key and address stay on your computer**, outside every project: in `%APPDATA%\paper-atlas\settings.json`
(Windows), `~/Library/Application Support/paper-atlas/` (macOS) or `~/.config/paper-atlas/` (Linux), readable by
you only. They reach the kit's tools only as environment variables while a job runs -- never in a command line, a
link or a log -- and the app shows them again only masked ("set (ends ...abcd)"). On Windows a key you stored
yourself with `setx OPENALEX_API_KEY` wins over the app's; Settings tells you when there is one.

The app is a small web server on your own computer only (127.0.0.1): it refuses requests from other machines and
from other web pages. It stops by itself 15 minutes after you close its window.

---

## Quick start

```
atlas new C:\Research\reef_atlas --title "Reef Mapping Atlas"      (in the kit folder)
cd C:\Research\reef_atlas                                          (the project has its own `atlas` now)
    (put PDFs, a Zotero export or a list of DOIs into the inbox\ folder)
atlas add inbox
atlas update
atlas open
```

`add` brings papers in; `update` does everything else (the first run on a few hundred papers takes from minutes to
an hour, mostly reading PDFs and finding figures; later runs only do what changed). `atlas status` shows what the
project holds and what is out of date; `atlas coverage` shows, PER PAPER, which step each paper is still missing
on this computer (`atlas coverage --missing figures` lists them).

---

## Adding papers -- three ways (mix them freely)

**1. A folder of PDFs.** Put them in `inbox\` -- in a **subfolder to give them a topic** (`inbox\sonar\*.pdf` ->
topic `sonar`; loose PDFs go to `unsorted`). Each PDF is identified from what it prints: the arXiv stamp, a DOI, or
its title (looked up on OpenAlex). It is filed only when its own first page prints that paper's title **and** an
author's name or its DOI -- so a citing paper, a correction notice or a supplement is never filed as the paper.

**2. A Zotero / BibTeX export.** In Zotero: *File -> Export Library...* (or right-click a collection -> *Export
Collection...*), format **BibTeX**, tick **Export Files**; put the `.bib` and its `files` folder in `inbox\`. Each
entry is checked against the registry by its DOI / arXiv id (or a near-exact title search), and its attached PDF is
checked to be that paper. Zotero's **CSL JSON** export works too (it carries no PDFs: put them beside it).

**3. A list.** A `.txt` file, one paper per line: a DOI (`10.1109/...`), an arXiv id (`2301.04104`) or a title.
Add ` | topic` after it to set its topic. A `.csv` with columns `doi`, `arxiv_id`, `title`, `topic` works too.
For papers on the list, `add` downloads open-access copies (arXiv, and the open copies OpenAlex knows) and checks
each one; the rest appear in the atlas with links to get them.

**4. Another atlas.** `atlas import <its project folder>` copies another atlas project's topics into yours -- each
paper's row as it is there (ratings, notes, tags), its verified records, and every PDF it holds that yours does not --
so its topics can become a section of yours (above: Sections). Run it again to re-sync with that project. It refuses
a topic name your project already uses for its own papers, never edits your `atlas.toml`, and prints the labels,
tasks and a `[[sections]]` block to paste in. If that project's PDFs live outside it, add `--from-pdfs <folder>`.
Then `atlas update`.

**What `add` prints.** Each item is `added`, `present` (already in the atlas), or **NEEDS A LOOK** -- with what it
found ("closest: '...' (title similarity 0.71)"). It never guesses. If you know what a PDF is, say so:

```
atlas add inbox\odd_scan.pdf --as odd_scan.pdf=10.1016/j.oceaneng.2020.107000
```

The full report is in `pdfs_atlas\intake\<time>.md`. Your inbox files are never moved or changed; the filed copies
are in `pdfs\` as `Surname_Year_Title.pdf`.

---

## What `update` does

| stage | what | needs |
|---|---|---|
| scan | which PDFs the project holds | |
| text | every page's text + a search index (new PDFs only) | |
| ocr | scanned pages (RapidOCR), then their text | `setup ocr` |
| html | arXiv's LaTeX HTML -- equations, sections (one page per 15 s, so **off** by default) | network |
| facts | code links, datasets, metrics, hardware -- each with the sentence it came from | |
| embed | dense vectors for search | torch (`setup gpu`; a CPU works for small atlases) |
| figures | every figure and its caption | YOLO with a GPU; a model-free finder without |
| meta | venue, dates, citation counts, reference lists (OpenAlex) | network |
| citemap | the citation map's layout and figures | |
| figindex | the Figures tab | |
| terms | the automatic glossary: abbreviations the papers define, with their own words | |
| build | the atlas page, ATLAS.xlsx, the Word copies | |
| lookup | each term's and task's full-text ranking (for the Find tab) | torch |
| rebuild | the page again, with the lookup tables | |
| zip | the portable bundle | |

A stage runs only when its inputs changed since its last good run. `atlas update --only text,build` runs just those;
`--skip zip` leaves one out; `--force` reruns everything; `--offline` skips the network stages. Logs are in
`pdfs_atlas\logs\`; a failed stage prints its last lines. A failed optional stage does not stop the others.

---

## Make it yours: `atlas.toml`

Every project has an `atlas.toml` (made by `atlas new`, all optional, commented):

```toml
[atlas]
title = "Reef Mapping Atlas"        # the page's title
short_name = "reef_atlas"           # the zip: reef_atlas_<date>.zip

[topics.sonar]                      # a readable label per topic
label = "Sonar navigation"

[[groups]]                          # the citation map colours papers by group (at most 7)
name = "Perception"
topics = ["sonar", "vision"]

[[tasks]]                           # one-click searches on the Find tab
id = "slam"
title = "Underwater SLAM"
query = "simultaneous localization and mapping for underwater vehicles with sonar and an IMU"

[pipeline]
gpu = "auto"                        # auto | yes | no
figures = "auto"                    # auto | yolo | heuristic | off
arxiv_html = false                  # true: equations and sections from arXiv (slow)
bundle_pdfs = true                  # false: a small zip without the PDFs
```

After changing tasks or topics, run `atlas update` (the lookup and the page are rebuilt).

**Sections: one atlas in several parts.** Name parts of the atlas -- each a set of topics, plus any extra papers by
key -- and the page gets a row of banners, one per section (a paper can sit in several):

```toml
[atlas]
sections_mode = "banners"           # or "atlases" (below)

[[sections]]
id = "marine-wm"
title = "Marine world models"
subtitle = "World-model navigation for a survey boat"
topics = ["marine_wm", "wm_navigation"]
tasks = ["usv-dynamics-wm"]         # its Find tab's one-click tasks (ids of [[tasks]])

[[sections]]
id = "mining"
title = "Paper mining techniques"
color = "#9b2a64"                   # its banner colour (optional)
topics = ["paper_mining"]
papers = ["smith2024parsing"]       # extra papers, by key, from other topics
```

* `banners` (the default): one page. Picking a banner shows only that section in every tab -- Find, Lexicon (terms
  its papers define, with their quotes), Papers, Citations, Figures, Labs, Trends -- and a link keeps it:
  `atlas.html#in=mining&find=layout detection`.
* `atlases`: each section is also an atlas of its own -- `index\atlas_<id>.html` with only its papers, terms, groups,
  citations, figures and tasks -- and the same banner row links the pages. `atlas update` (or `atlas zip`) also writes
  one zip per section, `<short_name>_<id>_<date>.zip`, to hand a section to someone by itself.
* A section with no papers yet is greyed out (and gets no page): add papers to one of its topics
  (`atlas add <folder> --topic <topic>`), then `atlas update`.

**Titles and examples.** The files `update` writes take their titles from `title`: `index\PAPERS.md` is "<title>:
master paper index", `LEXICON.md` / `LEXICON.docx` "<title>: lexicon", `LABS.md` "<title>: labs", `PAPERS.docx`
"<title>: papers", the README in the PDF folder "<title>: papers". To word one yourself, set it in a `[documents]`
table -- `papers`, `lexicon`, `labs`, `workbook`, `papers_docx`, `papers_docx_subtitle`, `lexicon_docx`,
`lexicon_docx_subtitle`, `pdfs`, `pdfs_what` (a misspelt key is refused, by name). Under `[atlas]`, `find_example` and
`figures_example` put an example into the Find and Figures search boxes.

**Your own terms.** Add a file `lexicon\<topic>.json`: a list of `{"term": ..., "aliases": [...], "definition":
"...", "citations": [{"key": "<paper key>", "quote": "<= 15 words, verbatim", "where": "p. 3", "source":
"fulltext"}]}`. Check the quotes against the papers with `python tools\term_mine.py verify lexicon\<topic>.json`
(each must be verbatim on the page it names). The automatic glossary (`lexicon\auto_glossary.json`) is rewritten by
each update -- to keep an entry you edited, set its `"auto": false`.

**Your own words about a paper.** Each paper's row in `papers\<topic>.json` has `summary` (the abstract, to start
with), `why_for_us` (empty: say why it matters to you), `relevance` (`core`, `high`, `medium`, `low`) and `tags`.
Edit them, then `atlas update --only build`.

---

## Finding papers

* On the page: the **Find** tab (it opens first). Links can be shared: `atlas.html#find=sonar%20odometry`.
* Command line:
  ```
  atlas find "train a world model of a boat from logged data"      # a task in plain words
  atlas find --term "SLAM"                                          # a term: its definition, then its papers
  atlas find "side-scan sonar" "IMU" --all                          # papers that contain EVERY term
  atlas find "..." --held --relevance high --keys                   # filters; --keys / --tsv / --json for scripts
  ```
* Python (from the kit's `tools` folder on `sys.path`): `from paper_find import find; find("...", k=10)` -- each hit
  says *why* it is listed.

How it ranks: it fuses the full text of every page (BM25 plus a dense model), the glossary / lexicon terms your query
names, each term's precomputed full-text ranking and the papers' own descriptions. On the BlueBoat atlas (a benchmark
built from its literature reviews, 715 held-out task queries) this found more of the papers a review cites in the top
ten than full-text search alone (0.849 vs 0.780). Your atlas is a different corpus, so treat that as a reason to
trust the method, not a number for your papers.

---

## Papers for a new task: `atlas discover`

`find` searches the papers you have. `discover` also searches the ones you do not have yet -- OpenAlex, Semantic
Scholar and arXiv -- for a task in plain words, and sorts everything into: already in your atlas, new and open, new
and closed.

```
atlas discover "follow a pipeline with a sonar on a small AUV and map its condition" --title "Pipeline following"
atlas discover pipeline_following --add --section      # bring the new ones in, as a section of your atlas
atlas update
```

The first command writes `discover\pipeline_following\`:

* `DISCOVER.md` -- what you already have, the best starting points (surveys, papers with code, new datasets), the new
  open papers and the new closed ones, each with WHY it is listed (which search found it, which of your best matches
  it cites or is cited by);
* `GET_THESE.md` -- the closed papers with their links: the DOI, the publisher's page, Google Scholar (a link for you
  to click -- the tool never searches Scholar: it has no API and its terms forbid scripts) and your library's proxy
  when `atlas.toml` names one (`[discover] proxy`). Download what you want into `inbox\`, then `atlas add inbox`;
* `add_list.txt` -- the top new papers (`--top 25`) as an `atlas add` list: delete a line to skip a paper, add DOIs;
* `plan.json` -- how it searched: the task's key phrases, the words your best matches share, the facets (each an OR
  of words) and the keyword queries. If a facet is wrong, fix it there and run `atlas discover pipeline_following`
  again -- everything already asked is cached.

How it searches, three ways at once because none finds enough alone: **by meaning** (the task's own words to
OpenAlex's semantic search and Semantic Scholar's search), **by keywords** (queries built from the task: its key
phrases, your glossary's synonyms for them, and the words your best matching papers use -- frequent in them, rare in
the rest of your atlas -- on OpenAlex, arXiv and Semantic Scholar) and **by citations** (the papers your best matches
cite, and the papers citing them). It ranks the lot by the task's meaning (your atlas's embedding model, with a GPU),
its words, how many searches found a paper, its citation links to your best matches, and its citation count.

`--add` brings `add_list.txt` in as `atlas add` would: each paper checked against its registry, open copies downloaded
and checked, closed ones listed in the atlas with where to get them. `--section` makes the task a section of your
atlas (its own banner: the new papers plus the ones you already had) and a one-click task in the Find tab.

Limits: OpenAlex allows a small daily budget without a key (about 100 searches); a free key (openalex.org, your
account, API key; then `setx OPENALEX_API_KEY <key>` on Windows, `export OPENALEX_API_KEY=<key>` in your shell profile
elsewhere) gives ten times that. Semantic Scholar's keyless service is shared and often busy. A source that could not
be asked is named in `DISCOVER.md` under COULD NOT ASK -- its papers are missing, not absent -- and a re-run asks it
again.

---

## Sharing

* **The zip** (`atlas zip`, or the last stage of `update`): unzip, open `atlas.html` -- every held paper opens its
  PDF, the Figures tab shows the images. `[pipeline] bundle_pdfs = false` makes a small one (PDFs are often
  copyrighted: share the full zip inside your group only).
* **The page alone**: `index\browser.html` needs the two files beside it (`atlas_citations.js`, `atlas_figures.js`).

---

## Troubleshooting

| you see | do |
|---|---|
| NEEDS A LOOK for a PDF | read the note: often a retitled version or a scan. If you know the paper: `atlas add FILE --as FILE=DOI` |
| "no text layer (a scan?)" | `setup ocr`, then `atlas update` (OCR runs), then add it again |
| HTTP 429 / "could not query" | the registry's daily limit: set `OPENALEX_API_KEY`, or try again tomorrow; nothing is lost |
| "no GPU" | fine: search uses a small CPU model, figures the model-free finder |
| "GPU no" (or a note about it) on a machine WITH an NVIDIA GPU | run `setup.cmd gpu` again: it swaps a CPU torch for the CUDA build. `.venv\Scripts\python tools\atlas_setup.py check` says what torch sees |
| `setup gpu` warns that torch cannot use the GPU | update the NVIDIA driver, or name a PyTorch index: `set ATLAS_CUDA=cu126`, then `setup.cmd gpu` |
| no Word copies | install pandoc and run `atlas update --only build` |
| a stage keeps failing | `atlas status`, then read its log in `pdfs_atlas\logs\` |
| some papers have no figures / search vectors | `atlas coverage` names them and the step each misses; `atlas update --only <step>` fills them |
| start a stage over | `atlas update --only <stage> --force` |

---

## What is checked, and what is automatic

* **Checked:** every paper's identity (registry record + the PDF's own title block), every glossary quote (verbatim on
  the page it names), every citation link (OpenAlex reference lists, both ends in your atlas).
* **Automatic, read before relying on it:** facts (pattern matches with their sentences), the glossary's definitions
  (they say what the abbreviation stands for, nothing more), figure detection (about 9 in 10 figures found), status
  (published / preprint, from the registry's record type).
* **Not judged at all:** whether a paper is relevant or good -- that is yours (`relevance`, `why_for_us`).

---

## Folders

```
<your project>\
  atlas.toml          settings
  inbox\              drop papers here (never changed)
  papers\<topic>.json one row per paper (edit summaries, relevance, tags here)
  refs\               each paper's registry record
  lexicon\            terms (yours + auto_glossary.json)
  index\              GENERATED: the atlas page, papers.json, ATLAS.xlsx, Word copies
  figures\            GENERATED: citation maps
  pdfs\               your PDFs, filed as Surname_Year_Title.pdf
  pdfs_text\ pdfs_ocr\ pdfs_figures\ pdfs_embed\    caches (rebuildable)
  pdfs_atlas\         the pipeline's state, logs, intake reports, the open view
```

Back up `atlas.toml`, `papers\`, `refs\`, `lexicon\` and `pdfs\`; everything else is rebuilt by `atlas update`.

## Licences

The kit's code is the RoboPI lab's, released under the GNU Affero General Public License v3.0 (`LICENSE`), the
licence of PyMuPDF and DocLayout-YOLO, which it uses. It uses pypdfium2 (Apache-2.0 / BSD-3), pdfminer.six
(MIT), PyMuPDF (AGPL-3.0), pypdf, matplotlib, numpy, openpyxl, python-docx, Pillow, lxml; optionally RapidOCR (Apache-2.0), sentence-transformers
models (each model's licence is on its Hugging Face page) and DocLayout-YOLO (code AGPL-3.0, weights Apache-2.0).
`KIT.json` records the commit this kit was exported from.
