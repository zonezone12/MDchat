---
marp: true
theme: default
paginate: true
size: 16:9
---

<!--
MDChat intro talk — ~15 minutes, pitched at freshman/sophomore students.
Renders as slides with Marp (VS Code "Marp for VS Code" extension, or `marp docs/mdchat-intro-talk.md -o talk.pdf`).
Without Marp it still reads fine as a plain outline — each "---" is one slide.
HTML comments like this one are Marp *presenter notes*: hidden when presenting, visible in the editor.
-->

# From Chat to Coding Agent

**How modern LLMs changed the way we do research — a working example from molecular dynamics simulation**

`[your name]` · `[course / date]`

<!-- (~0:30) Welcome the room. One line: this talk is about how the tool you chat with became a tool that can also do your coding for you — and what that means once it points at real lab data. -->

---

## Warm-up: two questions before we start

**Q1.** Who has asked ChatGPT (or similar) a question this week?

**Q2.** Who has written code to analyze data for a class or project?

Hold both hands up in your mind — by the end of this talk, they're the same tool.

<!-- (~0:45) Ask both questions for real, get a show of hands. Don't answer yet — just plant the idea that "chatting with AI" and "writing code" are about to become the same activity. -->

---

## Part 1 · How we got here

### The short history of talking to a machine

| When | What changed | In practice |
|---|---|---|
| **2018** | Autocomplete | GPT-1 predicts the next word. No chat window — just a research model. |
| **2020** | Fluent text | GPT-3 writes convincing paragraphs — but you still need code and an API key to reach it. |
| **Nov 2022** | Everyone can chat | ChatGPT puts a plain text box in front of anyone. Input: words. Output: words. |
| **2023–24** | It can see and act | Images and files come in; the first tool calls appear — the model can trigger a search or a calculation. |
| **2024–26** | It can build | Coding agents read a whole project, write files, run programs, check the result, and try again — with no human typing the code. |

<!-- (~1:45) Walk the timeline briskly. The point isn't the dates — it's the shape: each step adds a capability (fluency, then an interface, then sight, then hands). By 2026 the model doesn't just describe an answer, it goes and produces one. -->

---

## Part 1 · What changed

### A chatbot answers. An agent acts.

| Chat | Coding agent |
|---|---|
| Input: a question in words. | Input: a goal, in words. |
| Output: an answer in words. | Output: files edited, code run, plots produced. |
| One turn — it doesn't remember your files. | Many turns — it keeps state across a session. |
| No way to check if the answer is actually correct. | It can inspect its own output and try again. |

The tool in this talk, **MDChat**, lives entirely in the right-hand column.

<!-- (~1:00) This is the conceptual pivot of the talk. A chatbot is a vending machine for text. An agent has a loop: it can inspect what happened and decide what to do next. -->

---

## Part 2 · Why it matters for research

### The bottleneck was never the idea

- Most researchers can describe the analysis they want in one sentence.
- Turning that sentence into correct code — the right library call, the right array shape, a readable plot — is a separate skill, usually learned the hard way.
- That gap costs time: hours spent debugging a script instead of thinking about the actual question.

> "The science was ready. The code wasn't."

<!-- (~1:00) Reframe the bottleneck: it was rarely the science idea. It was translating that idea into correct, working code. Ask: how many hours have you spent debugging a plot instead of thinking about your actual question? -->

---

## Part 2 · The case study

### Molecular dynamics, in one slide

- A molecular dynamics (MD) simulation moves every atom forward in femtosecond-sized time steps and records a *trajectory* — essentially a movie of atoms.
- One trajectory file can hold gigabytes of coordinates across thousands of frames.
- Answering "did it stay folded?" or "when did the guest molecule leave the cage?" means writing numerical code against that movie: selections, alignment, statistics, plotting.

> Exactly the kind of task LLM coding agents are now good at.

<!-- (~1:00) Give just enough MD background for non-chemists to follow the rest: atoms move in tiny time steps, we record positions as a "trajectory," and the file is huge. Land on the punchline: this is exactly the shape of task agents are good at. -->

---

## Part 2 · Meet the tool

### MDChat: ask the trajectory a question

```
> /load complex.prmtop complex.nc

MDChat: Loaded 5,000 frames, 12,340 atoms.
Suggested selection for alignment: resname MOL and not name H*. Use this?

> yes — and compute RMSD and Rg together

MDChat: Running one trajectory pass (RMSD + Rg)… done.
RMSD plateaus around 2.1 Å after 3 ns — the assembly is stable.
Plot saved to output/session/rmsd_rg.png.
```

*representative example — illustrates the interaction pattern, not a captured live run*

<!-- (~1:00) Walk through the mock transcript line by line. Emphasize that MDChat asks before assuming — it proposes a selection and waits for confirmation rather than guessing silently. Say clearly this is a representative example, not a captured run. -->

---

## Part 3 · Under the hood

### One sentence in, real science out

1. **You ask, in plain English** — "Compute the RMSD and plot it."
2. **Agent picks a skill** — a vetted Python function from a registry (RMSD, Rg, contacts, clustering, cage geometry, …).
3. **Skill runs real code** — MDAnalysis / NumPy / SciPy on the actual trajectory. No shortcuts, no invented numbers.
4. **Results, explained** — data + plots + files come back in chemistry language, with a path to the artifact.

![width:320px](../test/find_endpoints.png)

*one skill's internals: automatic ring/endpoint detection on a molecular graph (`test/find_endpoints.png`). Nobody hand-writes this per project anymore.*

<!-- (~1:15) This is the mechanism slide. A "skill" is just a real, tested Python function — MDChat's job is picking the right one and calling it correctly, not inventing numbers. The endpoint-detection figure is real repo output, included to show the code being replaced is genuinely nontrivial. -->

---

## Part 3 · A real result

### Did the assembly hold together?

![width:520px](../output/BHHpH_motif_rmsd/plots/rmsd_vs_time_assembly.png)

*Fig. 1 — whole-cube RMSD to the initial structure, across an ensemble of replicate runs · `output/BHHpH_motif_rmsd/plots/rmsd_vs_time_assembly.png`*

- Each faint blue line is one independent simulation replicate; black is the mean.
- Rising RMSD over nanoseconds shows how far the assembly drifts from where it started.
- The dashed guide lines mark reference distances used to classify structures.

> No manual NumPy loop required to get here.

<!-- (~1:00) Explain the plot simply: each faint line is one replicate, black is the mean, x-axis is nanoseconds. Rising RMSD = drifting from the starting structure. Land on: one sentence produced this figure. -->

---

## Part 3 · A real result

### Turning numbers into chemistry

![width:520px](../output/BHHpH_motif_rmsd/plots/murata_assembly_overlay.png)

*Fig. 2 — density of whole-cube RMSD across all frames, with literature thresholds marking closed (~1.0 Å) vs. open (~2.7 Å) states · `output/BHHpH_motif_rmsd/plots/murata_assembly_overlay.png`*

- The agent doesn't just plot numbers — it applies a domain-specific classification rule from the literature (the Murata criteria).
- Frames get labeled "closed," "partially open," or "open" automatically.

> The judgment is chemistry. The bookkeeping is automated.

<!-- (~0:45) Second example — show the agent applies a domain rule (Murata criteria) from the literature to label frames as closed/open. Judgment stays chemistry, bookkeeping is automated. -->

---

## Part 3 · A real result

### And it's not just 2D plots

MDChat's clustering & cage-geometry skills also export interactive, rotatable 3D structures of representative conformations.

> **◇ [ INSERT INTERACTIVE 3D STRUCTURE VIEWER HERE ] ◇**
>
> e.g. embed an NGL.js / 3Dmol.js viewer, or link/iframe one of MDChat's exported representative-structure HTML files, before you present.
>
> Example source: `output/changepoints/clusters/.../cluster_02_structures.html`

<!-- (~1:00) This box is intentionally empty. Drop in your NGL.js or 3Dmol.js viewer, or an <iframe> pointed at one of MDChat's exported representative-structure HTML files, before you present. -->

---

## Part 4 · What actually changed

### Before and after, for the researcher

| Before | After |
|---|---|
| Learn MDAnalysis / NumPy / Matplotlib syntax first. | Describe the question in plain English. |
| Write and debug a new script per question. | Get a checked, reproducible analysis in minutes. |
| Re-derive the same RMSD loop for the tenth project. | Spend saved time interpreting chemistry, not indexing arrays. |
| Wait on a labmate who "knows Python." | Still need to know what a good answer looks like. |

The barrier is lower. It isn't gone — **and that's the point.**

<!-- (~1:00) Be honest here: the barrier is lower, not gone. You still need to know what a sensible RMSD value looks like. This is the credibility moment of the talk — don't oversell it. -->

---

## Part 4 · The bigger picture

### This isn't just chemistry

- **Biology** — agents drafting and running variant-calling and sequencing pipelines.
- **Physics & astronomy** — agents wrangling detector and telescope data into usable form.
- **Social science** — agents cleaning, coding, and modeling survey and interview data.
- **Every field** — the common thread: natural language in, verified computation out.

Whatever you study, this pattern will show up in your research.

<!-- (~0:45) Zoom out from chemistry to every field in the room. Whatever major they're in, the same pattern — natural language in, verified computation out — is already showing up or about to. -->

---

## Part 4 · Use it responsibly

### An agent is a collaborator, not an oracle

- Rule #1 written into MDChat itself: *"do not fabricate results"* — every number must come from a real computation on real data.
- Agents can still misread a request, pick the wrong selection, or hit a bug — you have to be able to tell.
- The fundamentals you learn in class — statistics, what RMSD even means — are exactly what let you check the agent's work.

> Learning to verify is the new learning to code.

<!-- (~0:45) Important ethical/practical beat. "Do not fabricate results" is a literal written rule inside MDChat's own instructions — not a vague hope. The skill you need now is verification, not just syntax. -->

---

## Closing

### Three things to take with you

1. LLMs moved from answering in words to acting with tools — that's the chat → agent shift.
2. MDChat shows what that shift looks like inside a real research workflow: plain-English questions, real computed answers.
3. The more precisely you understand your field, the better you can direct — and check — an agent working in it.

<!-- (~0:45) Slow down for the summary — this is what they should remember tomorrow. Point 3 is the one to linger on: domain knowledge becomes more valuable, not less, once an agent can execute for you. -->

---

# Questions?

MDChat — an LLM coding agent for molecular dynamics analysis.

`[repo link / contact]`

<!-- (~0:15) Open the floor. If someone asks for the repo link, give it verbally rather than reading a guessed URL off the slide. -->
