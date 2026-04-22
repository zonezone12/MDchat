"""Interactive trajectory clip + browser-side GIF export (NGL + gifenc)."""

from __future__ import annotations

import base64
import functools
import json
import os
import socketserver
import threading
from typing import Any, TYPE_CHECKING
from urllib.parse import quote

from ..skill import Parameter, ParamType, Skill, SkillResult
from ..registry import get_default_registry
from .visualization import _DEFAULT_HIGHLIGHT_COLORS, _NGL_REPR_CHOICES

if TYPE_CHECKING:
    from ..context import AnalysisContext

# Keep TCPServer instances alive for daemon threads serving the viewer.
_http_servers: list[socketserver.TCPServer] = []


def _find_repo_root(path: str) -> str | None:
    """Walk upward from *path* until a directory containing pyproject.toml."""
    d = os.path.dirname(os.path.abspath(path))
    for _ in range(32):
        if os.path.isfile(os.path.join(d, "pyproject.toml")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return None


def _start_local_http_server(
    root: str,
    port_start: int,
    port_attempts: int,
) -> tuple[int | None, str | None]:
    """
    Serve *root* at http://127.0.0.1:<port>/ on a daemon thread.
    Returns (port, None) on success, or (None, error reason).
    """
    import http.server

    root = os.path.abspath(root)
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=root)

    for port in range(port_start, port_start + port_attempts):
        try:
            httpd = socketserver.TCPServer(("127.0.0.1", port), handler)
            break
        except OSError:
            continue
    else:
        return None, f"no free port in [{port_start}, {port_start + port_attempts - 1}]"

    def _run() -> None:
        httpd.serve_forever()

    threading.Thread(target=_run, daemon=True, name="mdchat-trajectory-movie-http").start()
    _http_servers.append(httpd)
    return port, None


def _viewer_http_url(repo_root: str, html_abs_path: str, port: int) -> str:
    rel = os.path.relpath(html_abs_path, repo_root)
    rel_posix = rel.replace(os.sep, "/")
    path_q = quote(rel_posix, safe="/")
    return f"http://127.0.0.1:{port}/{path_q}"


def _build_ngl_layer_indices(
    atoms, idx_to_pos: dict[int, int]
) -> dict[str, list[int]]:
    """
    Split exported atoms into per-species NGL atom-index lists (0..N-1).
    Prefer residue-name buckets so each chemical species gets its own style control.
    Falls back to broad structural buckets when residue names are unavailable.
    """
    n = len(atoms)
    if n == 0:
        return {}
    layers: dict[str, list[int]] = {}

    # Primary path: one layer per residue name (species-like control in typical systems).
    try:
        by_resname: dict[str, list[int]] = {}
        for atom in atoms:
            key = str(getattr(atom, "resname", "") or "").strip() or "UNSPEC"
            pos = idx_to_pos.get(atom.index)
            if pos is None:
                continue
            by_resname.setdefault(key, []).append(pos)
        if len(by_resname) > 1:
            for rn in sorted(by_resname.keys()):
                layers[rn] = sorted(by_resname[rn])
            return layers
    except Exception:
        pass

    # Fallback path: broad mutually exclusive buckets.
    all_pos = set(range(n))
    assigned: set[int] = set()

    def take(sel_str: str, name: str) -> None:
        try:
            ag = atoms.select_atoms(sel_str)
        except Exception:
            return
        pos = sorted(
            idx_to_pos[a.index]
            for a in ag
            if a.index in idx_to_pos and idx_to_pos[a.index] not in assigned
        )
        if pos:
            layers[name] = pos
            assigned.update(pos)

    take("protein", "protein")
    take("nucleic", "nucleic")
    water_sel = (
        "resname HOH or resname WAT or resname SOL or resname TIP3 or "
        "resname OPC or resname SPC or resname TIP4 or resname TIP4P or resname TIP5"
    )
    take(water_sel, "water")
    other = sorted(all_pos - assigned)
    if other:
        layers["other"] = other
    if not layers:
        layers["all"] = list(range(n))
    return layers


def _build_layers_from_residue_catalog(
    u, idx_to_pos: dict[int, int], residue_catalog: list[dict[str, Any]] | None
) -> dict[str, list[int]]:
    """
    Build per-species layers from context residue_catalog selections.
    Returns empty dict when no valid per-species layers are available.
    """
    if not residue_catalog:
        return {}

    layers: dict[str, list[int]] = {}
    for row in residue_catalog:
        sel = str(row.get("selection", "") or "").strip()
        if not sel:
            continue
        segid = str(row.get("segid", "") or "").strip()
        resname = str(row.get("resname", "") or "").strip() or "UNSPEC"
        key = f"{segid}:{resname}" if segid else resname
        try:
            ag = u.select_atoms(sel)
        except Exception:
            continue
        pos = sorted(idx_to_pos[a.index] for a in ag if a.index in idx_to_pos)
        if pos:
            layers[key] = pos
    return layers


def _build_multiframe_pdb(u, atoms, frame_indices: list[int]) -> str:
    """Write selected trajectory frames as a multi-MODEL PDB string."""
    import tempfile

    import MDAnalysis as mda

    fd, tmp_path = tempfile.mkstemp(suffix=".pdb")
    os.close(fd)
    try:
        with mda.Writer(tmp_path, multiframe=True) as w:
            for fi in frame_indices:
                u.trajectory[fi]
                w.write(atoms)
        with open(tmp_path, encoding="utf-8") as fh:
            return fh.read()
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


# Embedded HTML: NGL loads multi-model PDB as trajectory (asTrajectory: true).
# User sets the view, then "Download GIF" captures each frame at the current camera.
TRAJECTORY_MOVIE_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  * {{ box-sizing: border-box; }}
  html, body {{ margin: 0; height: 100%; font-family: system-ui, sans-serif; background: #12121a; color: #e8e8e8; }}
  #viewport {{ width: 100vw; height: calc(100vh - 200px); }}
  #bar {{
    position: absolute; bottom: 0; left: 0; right: 0; min-height: 200px;
    background: rgba(0,0,0,0.75); padding: 10px 14px 14px;
    display: flex; flex-wrap: wrap; gap: 10px; align-items: center;
    border-top: 1px solid #333;
  }}
  #bar label {{ font-size: 12px; color: #aaa; display: block; margin-bottom: 4px; }}
  #bar button, #bar input[type="range"] {{ vertical-align: middle; }}
  button {{
    background: #3a5bc7; color: #fff; border: none; padding: 8px 14px;
    border-radius: 6px; cursor: pointer; font-size: 13px;
  }}
  button:disabled {{ opacity: 0.45; cursor: not-allowed; }}
  button.secondary {{ background: #444; }}
  #info {{
    position: absolute; top: 10px; left: 10px; font-size: 12px;
    background: rgba(0,0,0,0.55); padding: 8px 12px; border-radius: 8px;
    max-width: min(520px, 92vw); pointer-events: none; line-height: 1.45;
  }}
  #status {{ font-size: 12px; color: #9cf; min-width: 200px; flex: 1 1 200px; }}
  .row {{ display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }}
  input[type="range"] {{ width: 160px; }}
  select {{
    background: #2a2a38; color: #e8e8e8; border: 1px solid #444;
    padding: 6px 10px; border-radius: 6px; font-size: 13px; max-width: 200px;
  }}
  #layerControls {{ display: flex; flex-wrap: wrap; gap: 10px 16px; align-items: center; }}
  #layerControls label {{ display: inline; margin-bottom: 0; font-size: 12px; color: #ccc; cursor: pointer; }}
  #layerControls input {{ vertical-align: middle; margin-right: 4px; }}
  .layer-item {{ display: flex; align-items: center; gap: 8px; background: #1f1f2b; border: 1px solid #333; border-radius: 8px; padding: 6px 8px; }}
  .layer-item select {{ max-width: 150px; padding: 4px 8px; font-size: 12px; }}
  #appearanceRow {{ flex: 1 1 100%; border-top: 1px solid #333; padding-top: 10px; margin-top: 4px; }}
</style>
<script src="https://unpkg.com/ngl@2.3.1/dist/ngl.js"></script>
</head>
<body>
<div id="viewport"></div>
<div id="info">{info_text}</div>
<div id="bar">
  <div>
    <label>Frame in clip</label>
    <div class="row">
      <input type="range" id="fslider" min="0" max="0" value="0" />
      <span id="flab">0</span>
    </div>
  </div>
  <div>
    <label>&nbsp;</label>
    <div class="row">
      <button type="button" id="btnPlay">Play</button>
      <button type="button" class="secondary" id="btnStop">Stop</button>
    </div>
  </div>
  <div>
    <label>Playback (ms / frame)</label>
    <div class="row">
      <input type="number" id="playMs" value="80" min="20" max="2000" step="10" style="width:72px" />
    </div>
  </div>
  <div>
    <label>GIF delay (ms per frame)</label>
    <div class="row">
      <input type="number" id="gifDelay" value="{gif_delay_ms}" min="20" max="500" step="5" style="width:72px" />
    </div>
  </div>
  <div id="appearanceRow">
    <label>Species appearance (atoms in this clip only)</label>
    <div id="layerControls" class="row"></div>
  </div>
  <div style="flex:1 1 100%">
    <label>After you set the view (rotate / zoom / pan), capture the animation</label>
    <div class="row">
      <button type="button" id="btnGif">Download GIF</button>
      <span id="status">Ready.</span>
    </div>
  </div>
</div>
<script>
/**
 * Classic script (not type="module"): Chrome blocks top-level ES module imports on
 * file:// pages. We load gifenc via dynamic import() after DOM ready; that path
 * fetches HTTPS and works when opened as a local file (needs network once).
 */
const PDB_B64 = "{pdb_b64}";
const FRAME_LABELS = {frame_labels_json};
const GIF_DELAY_DEFAULT = {gif_delay_ms};
const DEFAULT_LAYER_REPR = {default_main_repr_json};
const LAYER_INDICES = __MDCHAT_LAYER_JSON__;
const HIGHLIGHTS_DATA = __MDCHAT_HIGHLIGHTS_JSON__;
const REPR_CHOICES = __MDCHAT_REPR_CHOICES_JSON__;
const GIFENC_MODULE = "https://unpkg.com/gifenc@1.0.3/dist/gifenc.esm.js";

function layerLabel(key) {{
  var m = {{
    protein: "Protein",
    nucleic: "Nucleic / RNA",
    water: "Water",
    other: "Other (ligands, ions, …)",
    all: "All atoms",
    structure: "Structure"
  }};
  return m[key] || key;
}}

function layerDomId(prefix, key) {{
  return prefix + "_" + String(key).replace(/[^A-Za-z0-9_-]/g, "_");
}}

function seleFromIndices(arr) {{
  if (!arr || arr.length === 0) return null;
  return "@" + arr.join(",");
}}

function pdbTextFromB64(b64) {{
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new TextDecoder("ascii").decode(bytes);
}}

function blobToImageData(blob) {{
  return new Promise((resolve, reject) => {{
    const url = URL.createObjectURL(blob);
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {{
      const c = document.createElement("canvas");
      c.width = img.naturalWidth;
      c.height = img.naturalHeight;
      const ctx = c.getContext("2d");
      ctx.drawImage(img, 0, 0);
      URL.revokeObjectURL(url);
      resolve(ctx.getImageData(0, 0, c.width, c.height));
    }};
    img.onerror = () => {{ URL.revokeObjectURL(url); reject(new Error("Image load failed")); }};
    img.src = url;
  }});
}}

/**
 * Extract x,y,z for each MODEL block (MDAnalysis multi-frame PDB). NGL 2.x often does not
 * expose these as structure.frames when loading from Blob, so we drive frames manually.
 */
function parseMultiModelPdbCoords(pdbText) {{
  const lines = pdbText.split(/\\r?\\n/);
  const frames = [];
  let bucket = null;
  for (let li = 0; li < lines.length; li++) {{
    const line = lines[li];
    if (line.startsWith("MODEL")) {{
      bucket = [];
      frames.push(bucket);
    }} else if (line.startsWith("ENDMDL")) {{
      bucket = null;
    }} else if (bucket !== null && (line.startsWith("ATOM") || line.startsWith("HETATM"))) {{
      if (line.length < 54) {{
        throw new Error("ATOM/HETATM line too short at line " + (li + 1));
      }}
      const x = parseFloat(line.substr(30, 8));
      const y = parseFloat(line.substr(38, 8));
      const z = parseFloat(line.substr(46, 8));
      bucket.push(x, y, z);
    }}
  }}
  const out = [];
  const n = frames.length > 0 ? frames[0].length : 0;
  for (let f = 0; f < frames.length; f++) {{
    if (frames[f].length !== n) {{
      throw new Error("Atom count differs in MODEL " + (f + 1));
    }}
    out.push(new Float32Array(frames[f]));
  }}
  return out;
}}

/** First MODEL … ENDMDL only — enough for NGL topology + initial coords. */
function firstModelPdbText(pdbText) {{
  const lines = pdbText.split(/\\r?\\n/);
  const acc = [];
  for (let i = 0; i < lines.length; i++) {{
    acc.push(lines[i]);
    if (lines[i].startsWith("ENDMDL")) {{
      break;
    }}
  }}
  return acc.join("\\n");
}}

function runTrajectoryViewer(gifenc) {{
  var GIFEncoder = gifenc.GIFEncoder;
  var quantize = gifenc.quantize;
  var applyPalette = gifenc.applyPalette;

  const pdbText = pdbTextFromB64(PDB_B64);
  var statusEarly = document.getElementById("status");
  statusEarly.textContent = "Parsing multi-model PDB…";

  var frameCoords;
  try {{
    frameCoords = parseMultiModelPdbCoords(pdbText);
  }} catch (e) {{
    statusEarly.textContent = "Parse error: " + (e && e.message ? e.message : e);
    return;
  }}
  if (frameCoords.length < 2) {{
    statusEarly.textContent = "Expected at least two MODEL blocks in the embedded PDB.";
    return;
  }}

  const firstPdb = firstModelPdbText(pdbText);
  const pdbBlob = new Blob([firstPdb], {{ type: "text/plain" }});

  const stage = new NGL.Stage("viewport", {{
    backgroundColor: "{bg_color}",
    ambientIntensity: 0.35,
    quality: "high"
  }});
  window.addEventListener("resize", function () {{ stage.handleResize(); }});

  const fslider = document.getElementById("fslider");
  const flab = document.getElementById("flab");
  const btnPlay = document.getElementById("btnPlay");
  const btnStop = document.getElementById("btnStop");
  const btnGif = document.getElementById("btnGif");
  const statusEl = document.getElementById("status");
  const playMs = document.getElementById("playMs");
  const gifDelayInput = document.getElementById("gifDelay");
  gifDelayInput.value = String(GIF_DELAY_DEFAULT);

  let structComp = null;
  let nFrames = 0;
  let playTimer = null;
  var layerVisibility = {{}};
  var layerRepr = {{}};

  function initLayerVisibility() {{
    Object.keys(LAYER_INDICES).forEach(function (k) {{
      layerVisibility[k] = true;
      layerRepr[k] = DEFAULT_LAYER_REPR;
    }});
  }}

  function rebuildRepresentations() {{
    if (!structComp) return;
    structComp.removeAllRepresentations();
    var keys = Object.keys(LAYER_INDICES);
    if (keys.length === 0) {{
      structComp.addRepresentation(DEFAULT_LAYER_REPR, {{ color: "element", radiusScale: 1.5 }});
    }} else {{
      keys.forEach(function (k) {{
        if (!layerVisibility[k]) return;
        var reprName = layerRepr[k] || DEFAULT_LAYER_REPR;
        var idx = LAYER_INDICES[k];
        var sele = seleFromIndices(idx);
        if (!sele) return;
        structComp.addRepresentation(reprName, {{
          sele: sele,
          color: "element",
          radiusScale: 1.5
        }});
      }});
    }}
    (HIGHLIGHTS_DATA || []).forEach(function (h) {{
      var sele = seleFromIndices(h.indices);
      if (!sele) return;
      structComp.addRepresentation(h.repr, {{
        sele: sele,
        color: h.color,
        radiusScale: 2.5
      }});
    }});
    stage.viewer.requestRender();
  }}

  function setupAppearanceControls() {{
    initLayerVisibility();
    var defaultRepr =
      REPR_CHOICES.indexOf(DEFAULT_LAYER_REPR) >= 0 ? DEFAULT_LAYER_REPR : REPR_CHOICES[0];
    Object.keys(layerRepr).forEach(function (k) {{ layerRepr[k] = defaultRepr; }});
    var lc = document.getElementById("layerControls");
    var lk = Object.keys(LAYER_INDICES);
    if (lc) {{
      lk.forEach(function (k) {{
        var wrap = document.createElement("div");
        wrap.className = "layer-item";
        var id = layerDomId("layer", k);
        var lab = document.createElement("label");
        var chk = document.createElement("input");
        chk.type = "checkbox";
        chk.id = id;
        chk.checked = true;
        chk.addEventListener("change", function () {{
          layerVisibility[k] = chk.checked;
          rebuildRepresentations();
        }});
        lab.appendChild(chk);
        lab.appendChild(document.createTextNode(" " + layerLabel(k)));
        wrap.appendChild(lab);

        var reprSel = document.createElement("select");
        reprSel.id = layerDomId("repr", k);
        reprSel.title = "Representation for " + layerLabel(k);
        if (REPR_CHOICES && REPR_CHOICES.length) {{
          REPR_CHOICES.forEach(function (r) {{
            var opt = document.createElement("option");
            opt.value = r;
            opt.textContent = r;
            reprSel.appendChild(opt);
          }});
          reprSel.value = defaultRepr;
        }}
        reprSel.addEventListener("change", function () {{
          layerRepr[k] = reprSel.value || defaultRepr;
          rebuildRepresentations();
        }});
        wrap.appendChild(reprSel);
        lc.appendChild(wrap);
      }});
    }}
  }}

  function setFrameCoords(i, done) {{
    if (!structComp || !frameCoords[i]) return;
    structComp.structure.updatePosition(frameCoords[i], true);
    if (typeof done === "function") done();
  }}

  function updateSliderLabel(idx) {{
    const real = FRAME_LABELS[idx] !== undefined ? FRAME_LABELS[idx] : idx;
    flab.textContent = idx + " / " + (nFrames - 1) + " (traj " + real + ")";
  }}

  stage.loadFile(pdbBlob, {{ ext: "pdb", defaultRepresentation: false }})
    .then(function (comp) {{
      structComp = comp;
      nFrames = frameCoords.length;
      var nAtom = structComp.structure.atomStore.count;
      var expect = nAtom * 3;
      if (frameCoords[0].length !== expect) {{
        statusEl.textContent =
          "Coordinate/atom mismatch: got " + frameCoords[0].length / 3 + " atoms, structure has " + nAtom;
        return;
      }}

      setupAppearanceControls();
      rebuildRepresentations();

      structComp.autoView();
      statusEl.textContent = "Ready.";
      fslider.max = String(nFrames - 1);
      fslider.value = "0";
      updateSliderLabel(0);
      setFrameCoords(0, function () {{}});

      fslider.addEventListener("input", function () {{
        const i = parseInt(fslider.value, 10);
        setFrameCoords(i, function () {{ updateSliderLabel(i); }});
      }});

      btnPlay.addEventListener("click", function () {{
        if (playTimer) return;
        let i = parseInt(fslider.value, 10);
        const ms = Math.max(20, parseInt(playMs.value, 10) || 80);
        playTimer = setInterval(function () {{
          i = (i + 1) % nFrames;
          fslider.value = String(i);
          setFrameCoords(i, function () {{ updateSliderLabel(i); }});
        }}, ms);
      }});

      btnStop.addEventListener("click", function () {{
        if (playTimer) {{ clearInterval(playTimer); playTimer = null; }}
      }});

      btnGif.addEventListener("click", async function () {{
        const delayMs = Math.max(20, parseInt(gifDelayInput.value, 10) || GIF_DELAY_DEFAULT);
        btnGif.disabled = true;
        btnPlay.disabled = true;
        if (playTimer) {{ clearInterval(playTimer); playTimer = null; }}

        const gif = GIFEncoder({{ initialCapacity: 4096 * 256, auto: true }});
        try {{
          for (let i = 0; i < nFrames; i++) {{
            statusEl.textContent = "Rendering frame " + (i + 1) + " / " + nFrames + "…";
            fslider.value = String(i);
            await new Promise(function (resolve) {{
              setFrameCoords(i, resolve);
            }});
            await new Promise(function (r) {{ requestAnimationFrame(function () {{ requestAnimationFrame(r); }}); }});
            await new Promise(function (r) {{ setTimeout(r, 40); }});

            const imgBlob = await stage.makeImage({{
              factor: 2,
              antialias: true,
              trim: false,
              transparent: false
            }});
            const imageData = await blobToImageData(imgBlob);
            const w = imageData.width;
            const h = imageData.height;
            const rgba = new Uint8Array(imageData.data.buffer);
            const palette = quantize(rgba, 256);
            const index = applyPalette(rgba, palette);
            gif.writeFrame(index, w, h, {{
              palette: palette,
              delay: delayMs,
              repeat: 0
            }});
          }}
          gif.finish();
          const bytes = gif.bytes();
          const outBlob = new Blob([bytes], {{ type: "image/gif" }});
          const a = document.createElement("a");
          a.href = URL.createObjectURL(outBlob);
          a.download = "{download_name}";
          a.click();
          URL.revokeObjectURL(a.href);
          statusEl.textContent = "GIF saved (" + nFrames + " frames).";
        }} catch (e) {{
          console.error(e);
          statusEl.textContent = "GIF failed: " + (e && e.message ? e.message : e);
        }} finally {{
          btnGif.disabled = false;
          btnPlay.disabled = false;
        }}
      }});
    }})
    .catch(function (e) {{
      console.error(e);
      document.getElementById("status").textContent = "Load failed: " + e;
    }});
}}

document.addEventListener("DOMContentLoaded", function () {{
  var statusEl = document.getElementById("status");
  import(GIFENC_MODULE)
    .then(function (mod) {{
      runTrajectoryViewer(mod);
    }})
    .catch(function (e) {{
      console.error(e);
      statusEl.textContent =
        "Could not load GIF encoder from CDN. Offline or blocked. " +
        "Try: python -m http.server 8765 in the repo folder, then open " +
        "http://127.0.0.1:8765/output/.../this-file.html — Error: " + e;
    }});
}});
</script>
</body>
</html>
"""


class TrajectoryMovieSkill(Skill):
    name = "trajectory_movie"
    description = (
        "Build a self-contained HTML viewer with a trajectory clip around a key frame. "
        "Rotate/zoom in the browser, then download an animated GIF. Uses NGL.js and "
        "gifenc (HTTPS CDN). By default starts a local http.server from the repo root "
        "and can open the viewer URL so you do not need to run python -m http.server "
        "manually."
    )
    category = "visualization"
    parameters = [
        Parameter(
            "center_frame",
            ParamType.INTEGER,
            "Trajectory frame index (0-based) to center the clip on.",
            required=True,
            min_value=0,
        ),
        Parameter(
            "window",
            ParamType.INTEGER,
            "Include this many frames before and after the center (total up to "
            "2*window+1, after stride). Default 50 gives a longer clip than before.",
            required=False,
            default=50,
            min_value=1,
            max_value=500,
        ),
        Parameter(
            "stride",
            ParamType.INTEGER,
            "Use every Nth frame within the window.",
            required=False,
            default=1,
            min_value=1,
            max_value=100,
        ),
        Parameter(
            "selection",
            ParamType.ATOM_SELECTION,
            "Optional atom selection to include in the movie. If omitted, uses all atoms.",
            required=False,
            default=None,
        ),
        Parameter(
            "representation",
            ParamType.STRING,
            "NGL representation for the main selection.",
            required=False,
            default="licorice",
            enum_values=_NGL_REPR_CHOICES,
        ),
        Parameter(
            "highlight_selections",
            ParamType.ARRAY,
            "Optional highlight selections (same as visualize_structure).",
            required=False,
            default=[],
            items_type=ParamType.STRING,
        ),
        Parameter(
            "highlight_colors",
            ParamType.ARRAY,
            "Colors for highlights.",
            required=False,
            default=_DEFAULT_HIGHLIGHT_COLORS,
            items_type=ParamType.STRING,
        ),
        Parameter(
            "highlight_repr",
            ParamType.STRING,
            "Representation for highlighted atoms.",
            required=False,
            default="ball+stick",
            enum_values=_NGL_REPR_CHOICES,
        ),
        Parameter(
            "background",
            ParamType.STRING,
            "Viewer background.",
            required=False,
            default="white",
            enum_values=["white", "black", "#1a1a2e"],
        ),
        Parameter(
            "gif_delay_ms",
            ParamType.INTEGER,
            "Default delay between GIF frames in milliseconds (browser default).",
            required=False,
            default=125,
            min_value=20,
            max_value=500,
        ),
        Parameter(
            "max_frames",
            ParamType.INTEGER,
            "Refuse if the clip would exceed this many frames (safety cap).",
            required=False,
            default=250,
            min_value=10,
            max_value=2000,
        ),
        Parameter(
            "serve_http",
            ParamType.BOOLEAN,
            "After writing the HTML, start a background http.server bound to "
            "127.0.0.1 (repo root = web root) so the viewer opens over http:// "
            "without running python -m http.server yourself.",
            required=False,
            default=True,
        ),
        Parameter(
            "open_browser",
            ParamType.BOOLEAN,
            "If serve_http succeeds, open the default browser to the viewer URL.",
            required=False,
            default=True,
        ),
        Parameter(
            "http_port",
            ParamType.INTEGER,
            "First TCP port to try for the local server (tries following ports if busy).",
            required=False,
            default=8765,
            min_value=1024,
            max_value=65535,
        ),
        Parameter(
            "filename",
            ParamType.STRING,
            "Output HTML filename (saved in session output directory).",
            required=False,
            default="trajectory_movie.html",
        ),
    ]
    requires = ["universe"]
    produces = []

    def execute(self, context: AnalysisContext, **params: Any) -> SkillResult:
        u = context.universe
        center = int(params["center_frame"])
        window = int(params.get("window", 50))
        stride = int(params.get("stride", 1))
        selection = params.get("selection")
        main_repr = params.get("representation", "licorice")
        highlight_sels = params.get("highlight_selections", []) or []
        highlight_colors = params.get("highlight_colors", _DEFAULT_HIGHLIGHT_COLORS)
        highlight_repr_name = params.get("highlight_repr", "ball+stick")
        bg_key = params.get("background", "white")
        gif_delay_ms = int(params.get("gif_delay_ms", 125))
        max_frames = int(params.get("max_frames", 250))
        filename = params.get("filename", "trajectory_movie.html")
        serve_http = bool(params.get("serve_http", True))
        open_browser = bool(params.get("open_browser", True))
        http_port = int(params.get("http_port", 8765))

        bg_map = {"white": "#ffffff", "black": "#000000", "#1a1a2e": "#1a1a2e"}
        bg_color = bg_map.get(bg_key, "#ffffff")

        n_traj = u.trajectory.n_frames
        if center < 0 or center >= n_traj:
            return SkillResult(
                success=False,
                error=f"center_frame {center} out of range (0–{n_traj - 1}).",
                summary=f"Invalid center_frame. Trajectory has {n_traj} frames.",
            )

        lo = max(0, center - window)
        hi = min(n_traj - 1, center + window)
        frame_indices = list(range(lo, hi + 1, stride))
        if not frame_indices:
            return SkillResult(
                success=False,
                error="No frames in selected window.",
                summary="Empty frame list; widen window or check stride.",
            )

        if len(frame_indices) > max_frames:
            return SkillResult(
                success=False,
                error=f"Clip has {len(frame_indices)} frames; max_frames is {max_frames}.",
                summary=(
                    f"Too many frames ({len(frame_indices)}). "
                    f"Increase stride, reduce window, or raise max_frames."
                ),
            )

        if selection:
            try:
                atoms = u.select_atoms(selection)
            except Exception as exc:
                return SkillResult(
                    success=False,
                    error=f"Invalid selection '{selection}': {exc}",
                    summary=f"Atom selection failed: {exc}",
                )
        else:
            atoms = u.atoms

        if len(atoms) == 0:
            return SkillResult(
                success=False,
                error=(
                    f"Selection '{selection}' matched 0 atoms."
                    if selection
                    else "No atoms available in the universe."
                ),
                summary="No atoms matched the selection.",
            )

        idx_to_pos = {atom.index: pos for pos, atom in enumerate(atoms)}

        layers = _build_layers_from_residue_catalog(
            u, idx_to_pos, context.get("residue_catalog")
        ) or _build_ngl_layer_indices(atoms, idx_to_pos)

        highlight_summaries: list[str] = []
        highlights_payload: list[dict[str, Any]] = []
        for i, hl_sel in enumerate(highlight_sels):
            color = highlight_colors[i % len(highlight_colors)]
            try:
                hl_atoms = u.select_atoms(hl_sel)
            except Exception:
                highlight_summaries.append(f"  - '{hl_sel}': invalid selection, skipped")
                continue
            ngl_positions = sorted(
                idx_to_pos[a.index] for a in hl_atoms if a.index in idx_to_pos
            )
            if not ngl_positions:
                highlight_summaries.append(
                    f"  - '{hl_sel}': no overlap with exported movie atoms, skipped"
                )
                continue
            highlights_payload.append(
                {
                    "indices": ngl_positions,
                    "color": color,
                    "repr": highlight_repr_name,
                }
            )
            highlight_summaries.append(f"  - '{hl_sel}': {len(ngl_positions)} atoms in {color}")

        pdb_text = _build_multiframe_pdb(u, atoms, frame_indices)
        pdb_b64 = base64.b64encode(pdb_text.encode("ascii", errors="replace")).decode("ascii")

        frame_labels_json = json.dumps(frame_indices)
        title = f"MDChat trajectory clip — frames {frame_indices[0]}–{frame_indices[-1]}"
        info_text = (
            f"{len(frame_indices)} frames &bull; stride {stride} &bull; "
            f"traj indices {frame_indices[0]}–{frame_indices[-1]} &bull; "
            f"{len(atoms)} atoms &bull; "
            f"{selection if selection else 'all atoms'}<br/>"
            f"Set each species appearance and visibility (embedded atoms only), "
            f"then <b>Download GIF</b>."
        )

        base_dl = filename.rsplit(".", 1)[0] if "." in filename else filename
        download_name = base_dl + ".gif"

        html = TRAJECTORY_MOVIE_HTML.format(
            title=title,
            info_text=info_text,
            pdb_b64=pdb_b64,
            frame_labels_json=frame_labels_json,
            gif_delay_ms=gif_delay_ms,
            bg_color=bg_color,
            default_main_repr_json=json.dumps(main_repr),
            download_name=download_name,
        )
        html = html.replace("__MDCHAT_LAYER_JSON__", json.dumps(layers))
        html = html.replace("__MDCHAT_HIGHLIGHTS_JSON__", json.dumps(highlights_payload))
        html = html.replace("__MDCHAT_REPR_CHOICES_JSON__", json.dumps(_NGL_REPR_CHOICES))

        out_path = os.path.join(context.output_dir, filename)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html)

        artifacts: dict[str, str] = {"trajectory_movie_html": out_path}
        summary_parts = [
            f"Trajectory movie viewer: {out_path}",
            f"Clip: {len(frame_indices)} frames (trajectory indices "
            f"{frame_indices[0]}–{frame_indices[-1]}), center {center}.",
            (
                f"Movie atoms: '{selection}' ({len(atoms)} atoms)."
                if selection
                else f"Movie atoms: all atoms ({len(atoms)} atoms)."
            ),
            "In the browser, set per-species styles and visibility (atoms in your "
            "movie export), rotate/zoom, then **Download GIF**.",
        ]

        viewer_url: str | None = None
        if serve_http:
            repo_root = _find_repo_root(out_path)
            if repo_root:
                port, err = _start_local_http_server(
                    repo_root, http_port, port_attempts=20,
                )
                if port is not None:
                    viewer_url = _viewer_http_url(repo_root, out_path, port)
                    artifacts["viewer_url"] = viewer_url
                    summary_parts.append(
                        f"Local server: http://127.0.0.1:{port}/ (repo root). "
                        f"Viewer URL: {viewer_url}"
                    )
                    if open_browser:
                        import webbrowser

                        webbrowser.open(viewer_url)
                        summary_parts.append("Opened the viewer URL in your default browser.")
                else:
                    summary_parts.append(
                        f"Could not start a local http.server ({err}). "
                        f"Open the HTML file manually or run: "
                        f"python -m http.server {http_port}"
                    )
            else:
                summary_parts.append(
                    "Could not find repo root (pyproject.toml) to start http.server; "
                    "open the HTML path above or run python -m http.server from the repo root."
                )

        if highlight_summaries:
            summary_parts.append("Highlights:\n" + "\n".join(highlight_summaries))

        return SkillResult(
            success=True,
            artifacts=artifacts,
            summary="\n".join(summary_parts),
        )


get_default_registry().register(TrajectoryMovieSkill())
