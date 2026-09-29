---
marp: true
theme: default
paginate: true
size: 16:9
---

<!--
Research-seminar version (~25 min). Two archives:

1. Cursor workspace export (22 Apr 2026): how the *analysis library* was
   built, Nov 2025 – Mar 2026. File:
   chat_history_exports/cursor_chat_history_workspace_20260422_105555.md
   plus the global dump cursor_chat_history_20260422_104901.md
2. Agent-transcript jsonl (this folder, 5 Mar – 18 Sep 2026): 83 parent
   chats, 480 researcher turns, 4,429 agent messages, 8,039 tool calls.

Student intro: docs/mdchat-intro-talk.md
Charts: canvas beside the chat.
-->

# LLM at work in research

**What six months of agent transcripts actually look like**

A coding-agent collaboration on molecular dynamics analysis · MDChat + GSA nanocubes

`[your name]` · `[seminar / date]`

<!-- (~0:30) This is not a product demo. It is a lab notebook of how one researcher and a coding agent built a platform and then used it on real trajectories. -->

---

## The claim

A coding agent is not a co-author and not an intern who “knows Python.”

It is a **very fast lab technician for code**.

You stay the PI of *meaning*:

- what question is being asked
- which paper’s definition is the spec
- when a plot is chemically wrong even if it runs

<!-- (~0:45) Plant this before any numbers. If they remember one sentence, this is it. -->

---

## The evidence, not a vibe

| | This project, Mar–Sep 2026 |
|---|---|
| Parent conversations | **83** |
| Researcher turns | **480** |
| Agent messages | **4,429** (~9 per your turn) |
| Tool calls | **8,039** (~17 per your turn) |
| Git commits | **90** |
| Subagent runs | **20** |

Every number below is counted from Cursor transcripts in this repo, not estimated.

<!-- (~0:45) Point at the ratio: you speak once, the agent goes and does a lot. That is the new unit of work. -->

---

## Two ledgers of the same lab

The jsonl transcripts in `agent-transcripts/` start **5 March 2026**.

They are **not** the start of the project.

Git’s first commit is **13 November 2025**. An April 22 Cursor export still holds the earlier chats: `endpoints_finder.py` → classes → one-pass iterator → guest/volume scoring.

MDChat sits on top of a library that was already being pair-programmed.

<!-- (~0:30) Correct the timeline. People will otherwise think MDChat was day one. -->

---

## Part A · How we first built it

### From a script to a class, then to a question

The earliest MD chats are not “build a chatbot.” They are chemistry plus a file:

> *Turn `endpoints_finder.py` into a class.*

> *Fuse `endpoints_finder.py` and `trajectory_deformation_workflow.py`. Use the endpoints between residues to calculate the metrics.*

> *Plot the distance between endpoints through frames and find the key pair that represents the expansion of the gas nanocube.*

The LLM’s first job was **to grow a scientific script into an API.**

<!-- (~0:50) This is the origin they asked for. Read the three quotes slowly. -->

---

## Part A · The design that everything later depends on

> *Some of our class functions need to iterate through all the frames. I want to write a gather class to run the iteration only at once.*

Then, after a too-eager in-memory gatherer:

> *FrameGatherer seems only iterate one time to store into memory which is not necessary. I would like to rewrite the whole codebase with Single-Pass Architecture.*

That became `TrajectoryIterator` + `FrameObserver`.  
MDChat’s later rule — “do not walk the trajectory twice” — is this sentence, productized.

<!-- (~0:50) Draw iterator + observers on the board if you can. -->

---

## Part A · What the LLM was good and bad at here

| You asked | What happened |
|---|---|
| Split the 2400-line workflow into `src/` modules | Agent mapped imports, you approved the plan, then “implement the plan” |
| Make `TrajectoryIterator` parallel / Dask | It ran. **4 Dask workers took 16 min; one process took 10.** |
| Align by copying the whole Amber mdcrd three times | HPC job **Killed** — you asked why three universes exist |
| Score simulations so we can pick the good ones | `FrameSelection` + `rescore_simulations.py` |

The agent writes the parallel code. **You keep the wall-clock and the OOM killer.**

<!-- (~0:50) Empirical CS, not vibe. 10 vs 16 minutes is a lecture-ready number. -->

---

## Part A · Then wrap the library as a product

> *Help me conclude this project into a research proposal. The vision is to develop a platform to utilize the LLM and create skills to analyze the MD trajectory without much coding for chemist researchers.*
>
> — 5 March 2026, `RESEARCH_PROPOSAL.md`

The proposal does not invent modules. It **lists what already existed**: EndpointAnalyzer, VolumeAnalyzer, GSAnalyzer, TrajectoryIterator, guest tracking, scoring.

Three weeks later, same day as a repo cleanup:

> *Let's start to build the MDchat.*
>
> — 31 March 2026

<!-- (~0:40) Proposal = wrapping, not a greenfield app. -->

---

## Part A · MDChat’s first architectural bets

Plan mode, then implement. Choices on the record:

- **Anthropic tool-use**, not free-form codegen
- **Rich CLI** first, not a web app
- **Five pilot skills** wrapping *existing* APIs: `load_trajectory`, `compute_rmsd`, `find_endpoints`, `plot_timeseries`, `score_simulation`
- Skills fail as `SkillResult(success=False)` so the **chat does not crash**

Design line from the plan: *“No code generation. The LLM only calls pre-defined skills.”*

<!-- (~0:45) This is the safety slide for a methods audience. -->

---

## Part A · Two doors into the same lab

Same week:

> *I want it also compatible to Cursor or VS Code. The user only needs to clone the repo and ready to use.*

> *Create a CLAUDE.md so that we can also use MDChat in the Cursor IDE for more beginner friendly.*

Two interfaces, one skill registry:

| Door | Who it is for |
|---|---|
| `mdchat` CLI + API key | Chemist in a terminal |
| Cursor agent + `CLAUDE.md` | Same chemist, no extra API |

You were already designing **how a beginner meets the tool**, not only how the tool computes.

<!-- (~0:40) Dual interface is a product decision, lecture-worthy. -->

---

## Part A · First science through the new door

| Date | Ask | What it forced |
|---|---|---|
| 6 Apr | Iodine intake in `traj/` | Guest skills on a real folder, not a demo |
| 8 Apr | NGL visualize skill | 3D, then a traceback, then a fix |
| 8 Apr | PhosT protein trajectory | “Wrap existing `src/` as skills — don’t rewrite” |
| 9 Apr | WT protein, ATP / Mg²⁺ | “Why didn’t you use the RMSF skill we already have?” |
| 13 Apr | Movie around a key frame | Browser `file://` security → skill launches `http.server` |

The first non-nanocube trajectories taught **generality**: hardcoded protein selections had to become `main_selection`.

<!-- (~1:00) This table is the ‘how we interact’ of the first month. -->

---

## Part A · Close the loop: make MDChat obey the iterator

20 April, back to the original idea:

> *In our design of Observer we first collect the observer then register to the iterator so that we don’t iterate the same trajectory multiple times. Is our MDChat follow this concept?*

21 April, when the LLM kept forgetting:

> *Set `run_trajectory_observer_pass` as the must-do process. The MDChat LLM behavior always forgets.*

The platform is finished when **the chat cannot skip the expensive physics.**

<!-- (~0:40) Land this, then go to the later paper work. -->

---

## Four phases (same person, changing job)

| Phase | When | What you were doing |
|---|---|---|
| **Library** | Nov 2025 – Mar 2026 | Scripts → `src/` modules, single-pass iterator, guest/volume/score |
| **Platform** | Mar–May 2026 | MDChat skills, `CLAUDE.md`, movies, observer-pass policy |
| **Science** | Jun–Jul 2026 | GSA features, changepoints, Imamura MSM |
| **Rigor / paper** | Aug–Sep 2026 | Pipeline integration, Jaccard, Murata G0–G5 |

Later jsonl chats get **heavier per turn** (~41 tool calls/chat in spring → ~295 in Aug–Sep). The chemistry got harder; you did not type more.

<!-- (~0:50) Now the three-phase slide from before, with Library restored. -->

---

## What the agent actually did with its hands

Of 8,039 tool calls:

| Action | Calls | Lab analogue |
|---|---:|---|
| Read files | 2,775 | Open the notebook the PI pointed at |
| Edit code | ~2,240 | Write the next version of the script |
| Search the repo | ~1,245 | “Did we already have this function?” |
| Run the shell | ~1,142 | Execute, test, fail, execute again |

It spent **more time reading this project than writing it.**

That is why a coding agent on *your* repo is different from ChatGPT in a blank window.

<!-- (~0:50) The pie is on the canvas. Verbally: read > edit. -->

---

## How we actually talked

Five recurring moves — this is the interaction pattern:

1. **Point at a file, then state the science**  
   `@plan.md` How does the table generate in G5?
2. **Paste the paper as a spec**  
   Imamura’s preprocessing paragraph; Murata’s A / B / C1 / C2 rules
3. **Paste the failure**  
   traceback, terminal dump, empty reply, `cp950` codec
4. **Correct the chemistry or the math**  
   (next slides)
5. **Approve a plan, then let it run**  
   “yes switch to agent mode and start to code”

<!-- (~1:00) This is the “how we interact” slide they asked for. Walk 1–5 with your hands. -->

---

## Architecture was a scientific decision

April, about not wasting trajectory I/O:

> *In our design of Observer we will first collect the observer then register to iterator so that we don't iterate the same trajectory multiple times. Is our MDChat follow this concept?*

April, about not letting the LLM forget:

> *I want to set `run_trajectory_observer_pass` the must-do process in workflow. The MDChat LLM behavior always forget or just register only one or two.*

The researcher specified the **computational contract**. The agent implemented it.

<!-- (~0:50) This is “LLM at work” as software design, not just codegen. Chemists in the room: one pass through a  GB trajectory is a real constraint. -->

---

## Catch 1 — the clustering would have been wrong

> *Important correction: Ward clustering. … In SciPy, the cleanest use of Ward is `linkage(X_scaled, method="ward")` not `linkage(squareform(distance_matrix), method="ward")`.*
>
> — 16 June 2026

The pipeline ran. The dendrogram would have looked like science.

It would not have been the Ward merge Imamura used.

<!-- (~0:40) Silent statistical error. Only a person who knows the method sees it. -->

---

## Catch 2 — the similarity metric was not Jaccard

> *Jaccard mixes fuzzy and exact matching. The numerator (shared) counts matches within ±tolerance, but the denominator (`union_size`) is the exact set union. Those don't compose into a valid Jaccard.*
>
> — 1 September 2026, pointing at `detection.py`

A comparison table for the paper, built on a broken formula, is worse than no table.

The researcher read the code as a method. The agent then repaired it.

<!-- (~0:40) Pair with the existing Jaccard canvas if you want a live figure. -->

---

## Catch 3 — a sort that destroys the feature

> *Why do we sort the distance? We should always keep the order of the distance pair consistent throughout all our experiment.*
>
> — 21 June 2026

Sorting unique pairwise distances is the kind of “cleanup” an LLM offers for free.

It makes trajectories **incomparable**. The PI has to say no.

<!-- (~0:35) Short and punchy. Good for a laugh of recognition. -->

---

## Catch 4 — the plot answered the wrong paper

> *I see the difference on how we calculate the RMSD. We use the whole GSA but Murata only calculate the cation–π interaction (Py+, Ph, Py+) or equator (Py+, R2, R3, Py+).*
>
> — September 2026, G0 thread

A whole-cube RMSD plot can be beautiful and still not be Murata Figure S11.

G0 in `plan.md` exists because of this sentence.

<!-- (~0:50) This is the paper-methods punchline. Show Fig. 2 from the student talk if you have it up. -->

---

## One conversation as a day in the lab

**4–28 August 2026** — integrate four changepoint scripts into `src/`

| | |
|---|---|
| Researcher turns | 77 |
| Agent messages | 561 |
| Tool calls | 1,046 |
| Then it kept going | Murata metastructures, ring COM not atom–atom, docs |

That is not “prompt once, receive a repo.”  
It is a **month of directed iteration**, compressed.

<!-- (~0:45) If they ask “does it save time?” — yes, but the unit is still a collaboration, not a vending machine. -->

---

## Division of labor (the slide to leave up)

| Researcher | Agent |
|---|---|
| Name the chemical question | Find the existing function |
| Paste the literature definition | Implement the loop / observer / PELT |
| Reject silent math or geometry errors | Run, fail, fix from the traceback |
| Decide when a plot is usable | Draw the plot again |
| Keep G0–G5 honest | Write the docs after the method exists |

**Domain knowledge got more valuable, not less.**

<!-- (~1:00) Slow down. This is the take-home. -->

---

## What this is *not*

- Not “the AI did the research”
- Not unsupervised analysis of a trajectory
- Not a replacement for knowing what 6.5 Å on a cation–π contact means
- Not cheaper than thinking — cheaper than **re-deriving the same NumPy loop**

MDChat’s own rule, written into the agent: **do not fabricate results.**

Every number still has to come from a real computation on a real file.

<!-- (~0:40) Credibility beat. Do not skip this in a research room. -->

---

## Advice if you start this in your group

1. **Put the paper in the prompt** as a spec, not a citation.
2. **Point at files** (`@plan.md`, `@detection.py`) — the agent’s advantage is the repo.
3. **Paste failures**, don’t rephrase them.
4. **Read the formula** the agent wrote. Jaccard, Ward, RMSD selection, pair order.
5. **Write the rule down** (`CLAUDE.md`, observer-pass policy) so the next session cannot forget.

The 22 April 2026 chat already asked for this lecture. The transcripts are now the data.

<!-- (~0:50) Practical close. Mention CLAUDE.md as “the PI’s standing orders.” -->

---

## Three things to take with you

1. The library came first (scripts → iterator → scoring). MDChat is an **adapter** on that library, not a replacement for it.
2. A coding agent **acts in your repository** — 83 later conversations, 8,039 tool calls, and a pipeline that now compares nanocube metastructures to Murata.
3. The skill that mattered was not typing Python. It was **catching when the plausible code was the wrong experiment** — including a Dask run that was slower, and an RMSD that answered the wrong paper.

<!-- (~0:40) Match the student talk’s closing shape, at a higher level. -->

---

# Questions?

Transcript summary (interactive): the canvas beside this chat  
Student version of the talk: `docs/mdchat-intro-talk.md`  
This outline: `docs/llm-at-work-research-talk.md`

`[repo / contact]`
