# MDChat Analysis Playbook

Use this as a quick guide for running MD analysis sessions with MDChat.

**Runtime:** MDChat can append this file to the model system prompt when `MDCHAT_INCLUDE_PLAYBOOK=1` is set in `.env` (optional `MDCHAT_PLAYBOOK_PATH` for a custom file). See `src/mdchat/engine_common.py`.

## 1) Start a Session

1. Provide topology and trajectory paths.
2. State your analysis goal (stability, pore opening, guest motion, etc.).
3. Confirm the main atom selection if needed.

Suggested prompt:
- `Load <topology_file> and <trajectory_file>. My goal is <goal>.`

## 2) Recommended Analysis Flow

1. **Load** trajectory.
2. **Quick stability scan** (RMSD + Rg).
3. **Targeted dynamics** (RMSF, PCA, contacts, endpoints, volume/GSA) based on your goal.
4. **Plot + export** artifacts (CSV/plots).
5. **Interpretation summary** (key events and likely structural meaning).

## 3) Metric Routing Tips

- If you request **2+ metrics together** (RMSD/Rg/contacts/RMSF/PCA), ask for a **single observer pass**.
- If you need **plateau detection** for RMSD or Rg, ask for standalone `compute_rmsd` / `compute_rg` with plateau options.
- If RMSF/PCA is included with `n_jobs > 1`, execution is expected to run effectively with `n_jobs=1` for accumulation safety.

## 4) Copy-Paste Prompt Templates

- **Single-pass multi-metric**
  - `Run RMSD, Rg, RMSF, and PCA in one pass on selection "<selection>" (PCA components=3).`

- **Plateau-focused RMSD**
  - `Compute RMSD for "<selection>" with plateau detection (window=80, rel_std_threshold=0.10).`

- **Contacts**
  - `Compute contacts between "<selection_a>" and "<selection_b>" with label "<label>".`

- **Endpoint analysis**
  - `Find endpoints and compute endpoint distances for: <residue_selection_list>.`

- **Session summary**
  - `Summarize the key structural events and list all generated files.`

## 5) What Good Output Looks Like

- A concise interpretation (not only raw numbers).
- Clear mention of where artifacts are saved.
- Follow-up suggestion based on results (e.g., “RMSD plateau starts near frame X; next check PCA mode shifts.”).
