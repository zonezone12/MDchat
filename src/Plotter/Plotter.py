from __future__ import annotations

import ast
import io
import os
import re
import warnings
from typing import Dict, List, Optional, Tuple, Union

# Try to import multiprocessing for parallel frame generation
try:
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from multiprocessing import cpu_count
    PARALLEL_AVAILABLE = True
except ImportError:
    PARALLEL_AVAILABLE = False
    ProcessPoolExecutor = None
    as_completed = None
    cpu_count = None

import numpy as np
import pandas as pd
from rdkit.Chem import Draw

try:
    import MDAnalysis as mda
except ImportError:
    import sys
    sys.stderr.write("MDAnalysis is required. pip install MDAnalysis\n")
    raise

# Optional matplotlib (kept for molecule visualization and RDKit integration)
try:
    import matplotlib.pyplot as plt  # type: ignore
    from matplotlib.animation import FuncAnimation  # type: ignore
    from matplotlib.colors import ListedColormap
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None  # type: ignore
    FuncAnimation = None  # type: ignore
    ListedColormap = None  # type: ignore
    warnings.warn("matplotlib not available. Molecule visualization will be disabled.")

# Datashader for high-performance 2D plotting
try:
    import datashader as ds
    from datashader import transfer_functions as tf
    from datashader.colors import viridis, inferno
    HAS_DATASHADER = True
except ImportError:
    HAS_DATASHADER = False
    ds = None  # type: ignore
    tf = None  # type: ignore
    viridis = None  # type: ignore
    inferno = None  # type: ignore
    warnings.warn("datashader not available. Will fall back to matplotlib if available.")

# PIL for image composition and text overlays
try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    Image = None  # type: ignore
    ImageDraw = None  # type: ignore
    ImageFont = None  # type: ignore
    warnings.warn("PIL/Pillow not available. Text overlays on datashader plots will be limited.")

# Optional GIF support
try:
    import imageio.v2 as imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False
    imageio = None  # type: ignore
    warnings.warn("imageio not available. GIF generation will be disabled.")

# Endpoint finder (now in EndpointAnalyzer module)
from src.EndpointAnalyzer import EndpointsFinder, EndpointAnalyzer  # type: ignore
from src.EndpointAnalyzer import EndpointAnalyzerObserver  # type: ignore

# Optional plotly for 3D visualization
try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    go = None  # type: ignore
    warnings.warn("plotly not available. Interactive 3D visualization will be disabled.")

# Optional scipy for image processing
try:
    from scipy import ndimage
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    ndimage = None  # type: ignore
    warnings.warn("scipy not available. Some volume visualization features will be disabled.")

# Optional scikit-image for mesh volume computation
try:
    from skimage import measure
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    measure = None  # type: ignore
    warnings.warn("scikit-image not available. Mesh volume computation will be disabled.")

try:
    from src.utils.plotly_molecule import make_molecule_components
except ImportError:
    make_molecule_components = None
    warnings.warn("plotly_molecule not available. Interactive 3D visualization will be limited.")

# Import rdkit utils for SMILES support
try:
    from src.utils.rdkit_utils import get_3d_coordinates_from_smiles
except ImportError:
    get_3d_coordinates_from_smiles = None
    warnings.warn("rdkit_utils not available. SMILES input will be disabled.")


def _parse_numpy_int_string_guest(s) -> Optional[int]:
    """Extract an int from raw CSV values or strings like ``np.int64(708)``."""
    if isinstance(s, (int, np.integer)):
        return int(s)
    if isinstance(s, str):
        match = re.search(r"np\.int\d+\((\d+)\)", s)
        if match:
            return int(match.group(1))
        try:
            return int(float(s))
        except Exception:
            return None
    try:
        return int(float(s))
    except Exception:
        return None


def _pair_guest_entry_exit_intervals(guest_residence_stats: dict) -> List[Tuple[int, int]]:
    """
    Match each exit to the latest unmatched entry for the same guest index
    (same logic as the guest timeline GIF). Falls back to index-aligned pairing
    when guest index lists are missing or empty.
    """
    entry_frames = guest_residence_stats.get("entry_frames") or []
    exit_frames = guest_residence_stats.get("exit_frames") or []
    entry_guest_indices = guest_residence_stats.get("entry_guest_indices") or []
    exit_guest_indices = guest_residence_stats.get("exit_guest_indices") or []

    pairs: List[Tuple[int, int]] = []

    use_guest_ids = (
        len(entry_guest_indices) == len(entry_frames)
        and len(exit_guest_indices) == len(exit_frames)
        and any(len(x or []) > 0 for x in entry_guest_indices)
        and any(len(x or []) > 0 for x in exit_guest_indices)
    )

    if use_guest_ids:
        guest_timelines: Dict[int, List[dict]] = {}
        for ef, gidxs in zip(entry_frames, entry_guest_indices):
            for g in gidxs or []:
                guest_timelines.setdefault(g, []).append({"entry": ef, "exit": None})
        for xf, gidxs in zip(exit_frames, exit_guest_indices):
            for g in gidxs or []:
                if g not in guest_timelines:
                    continue
                for period in reversed(guest_timelines[g]):
                    if period["exit"] is None:
                        period["exit"] = xf
                        break
        for periods in guest_timelines.values():
            for p in periods:
                ex = p["exit"]
                if ex is not None:
                    ef, xf = p["entry"], ex
                    if xf >= ef:
                        pairs.append((ef, xf))
    else:
        max_frame_val = max(entry_frames + exit_frames) if (entry_frames or exit_frames) else 0
        for i, ef in enumerate(entry_frames):
            if i < len(exit_frames):
                xf = exit_frames[i]
            else:
                xf = max_frame_val
            if xf >= ef:
                pairs.append((ef, xf))

    pairs.sort(key=lambda t: (t[0], t[1]))
    return pairs


def _merge_frame_intervals(pairs: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Merge overlapping or touching frame intervals (avoids stacked alpha turning into a solid block)."""
    if not pairs:
        return []
    sorted_pairs = sorted(pairs, key=lambda t: (t[0], t[1]))
    merged: List[List[int]] = [[sorted_pairs[0][0], sorted_pairs[0][1]]]
    for a, b in sorted_pairs[1:]:
        ma, mb = merged[-1]
        if a <= mb:
            merged[-1][1] = max(mb, b)
        else:
            merged.append([a, b])
    return [(int(u), int(v)) for u, v in merged]


def guest_residence_stats_from_entering_events_csv(
    events_csv_path: str,
    stats_csv_path: Optional[str] = None,
) -> dict:
    """
    Build a ``guest_residence_stats`` dict from ``*_guest_entering_events.csv``
    (and optional ``*_guest_entering_stats.csv``), for plotting without re-running analysis.
    """
    entry_frames: List[int] = []
    exit_frames: List[int] = []
    entry_times: List[float] = []
    exit_times: List[float] = []
    entry_guest_indices: List[List[int]] = []
    exit_guest_indices: List[List[int]] = []

    if not os.path.exists(events_csv_path):
        warnings.warn(f"Guest entering events CSV not found: {events_csv_path}")
        return {}

    try:
        events_df = pd.read_csv(events_csv_path)
        for _, row in events_df.iterrows():
            event_type = str(row["event_type"]).strip().lower()

            frame = _parse_numpy_int_string_guest(row["frame"])
            if frame is None:
                continue

            try:
                time = float(row["time"])
            except Exception:
                continue

            guest_indices_str = row.get("guest_indices", "[]")
            guest_indices: List[int] = []

            if pd.isna(guest_indices_str) or guest_indices_str == "":
                guest_indices = []
            elif isinstance(guest_indices_str, str):
                guest_indices_str = guest_indices_str.strip("\"'")
                matches = re.findall(r"np\.int\d+\((\d+)\)", guest_indices_str)
                if matches:
                    guest_indices = [int(m) for m in matches]
                else:
                    try:
                        parsed = ast.literal_eval(guest_indices_str)
                        if isinstance(parsed, list):
                            guest_indices = [
                                x for x in (_parse_numpy_int_string_guest(v) for v in parsed)
                                if x is not None
                            ]
                        else:
                            idx = _parse_numpy_int_string_guest(parsed)
                            if idx is not None:
                                guest_indices = [idx]
                    except Exception:
                        parts = guest_indices_str.strip("[]()").split(",")
                        for part in parts:
                            idx = _parse_numpy_int_string_guest(part.strip())
                            if idx is not None:
                                guest_indices.append(idx)
            elif isinstance(guest_indices_str, list):
                guest_indices = [
                    x for x in (_parse_numpy_int_string_guest(v) for v in guest_indices_str)
                    if x is not None
                ]
            else:
                idx = _parse_numpy_int_string_guest(guest_indices_str)
                if idx is not None:
                    guest_indices = [idx]

            if event_type == "entry":
                entry_frames.append(frame)
                entry_times.append(time)
                entry_guest_indices.append(guest_indices)
            elif event_type == "exit":
                exit_frames.append(frame)
                exit_times.append(time)
                exit_guest_indices.append(guest_indices)
    except Exception as e:
        warnings.warn(f"Failed to read guest entering events CSV {events_csv_path}: {e}")
        return {}

    guest_residence_stats: dict = {}
    if stats_csv_path and os.path.exists(stats_csv_path):
        try:
            stats_df = pd.read_csv(stats_csv_path)
            for _, row in stats_df.iterrows():
                metric = row["metric"]
                value = row["value"]
                if pd.notna(value):
                    if metric in ["first_entry_frame", "n_entries", "n_exits"]:
                        guest_residence_stats[metric] = int(value)
                    elif metric in [
                        "first_entry_time",
                        "total_time_inside",
                        "total_time_outside",
                        "avg_stay_duration",
                        "max_stay_duration",
                        "min_stay_duration",
                    ]:
                        guest_residence_stats[metric] = float(value)
                    else:
                        guest_residence_stats[metric] = value
        except Exception as e:
            warnings.warn(f"Failed to read stats CSV {stats_csv_path}: {e}")

    if "n_entries" not in guest_residence_stats:
        guest_residence_stats["n_entries"] = len(entry_frames)
    if "n_exits" not in guest_residence_stats:
        guest_residence_stats["n_exits"] = len(exit_frames)
    if "first_entry_frame" not in guest_residence_stats and entry_frames:
        guest_residence_stats["first_entry_frame"] = min(entry_frames)

    guest_residence_stats["entry_frames"] = entry_frames
    guest_residence_stats["exit_frames"] = exit_frames
    guest_residence_stats["entry_times"] = entry_times
    guest_residence_stats["exit_times"] = exit_times
    guest_residence_stats["entry_guest_indices"] = entry_guest_indices
    guest_residence_stats["exit_guest_indices"] = exit_guest_indices

    return guest_residence_stats


# ============================================================================
# Datashader Helper Functions
# ============================================================================

def _add_text_annotations(
    img: "Image.Image",
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    stats_text: str = "",
    legend_items: List[Tuple[str, str]] = None,
    x_range: Tuple[float, float] = (0, 1),
    y_range: Tuple[float, float] = (0, 1),
    margin: dict = None,
) -> "Image.Image":
    """
    Add text annotations to a PIL Image (for datashader plots).
    
    Args:
        img: PIL Image to annotate
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        stats_text: Statistics text box content
        legend_items: List of (label, color) tuples for legend
        x_range: Tuple of (min, max) for x-axis tick labels
        y_range: Tuple of (min, max) for y-axis tick labels
        margin: Dict with 'top', 'bottom', 'left', 'right' margins
    
    Returns:
        Annotated PIL Image
    """
    if not HAS_PIL or Image is None:
        return img
    
    if margin is None:
        margin = {'top': 60, 'bottom': 50, 'left': 80, 'right': 150}
    
    # Create new image with margins for annotations
    orig_width, orig_height = img.size
    new_width = orig_width + margin['left'] + margin['right']
    new_height = orig_height + margin['top'] + margin['bottom']
    
    # Create white background
    annotated = Image.new('RGBA', (new_width, new_height), (255, 255, 255, 255))
    
    # Paste the original image
    annotated.paste(img, (margin['left'], margin['top']))
    
    draw = ImageDraw.Draw(annotated)
    
    # Try to get a font, fall back to default
    try:
        font_title = ImageFont.truetype("arial.ttf", 16)
        font_label = ImageFont.truetype("arial.ttf", 12)
        font_tick = ImageFont.truetype("arial.ttf", 10)
    except (OSError, IOError):
        try:
            font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
            font_label = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
            font_tick = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
        except (OSError, IOError):
            font_title = ImageFont.load_default()
            font_label = font_title
            font_tick = font_title
    
    # Add title
    if title:
        title_bbox = draw.textbbox((0, 0), title, font=font_title)
        title_width = title_bbox[2] - title_bbox[0]
        title_x = margin['left'] + (orig_width - title_width) // 2
        draw.text((title_x, 10), title, fill='black', font=font_title)
    
    # Add x-axis label
    if xlabel:
        xlabel_bbox = draw.textbbox((0, 0), xlabel, font=font_label)
        xlabel_width = xlabel_bbox[2] - xlabel_bbox[0]
        xlabel_x = margin['left'] + (orig_width - xlabel_width) // 2
        xlabel_y = new_height - 25
        draw.text((xlabel_x, xlabel_y), xlabel, fill='black', font=font_label)
    
    # Add y-axis label (rotated)
    if ylabel:
        # Create a temporary image for rotated text
        ylabel_img = Image.new('RGBA', (200, 20), (255, 255, 255, 0))
        ylabel_draw = ImageDraw.Draw(ylabel_img)
        ylabel_draw.text((0, 0), ylabel, fill='black', font=font_label)
        ylabel_img = ylabel_img.rotate(90, expand=True)
        # Paste rotated label
        ylabel_y = margin['top'] + (orig_height - ylabel_img.size[1]) // 2
        annotated.paste(ylabel_img, (5, ylabel_y), ylabel_img)
    
    # Add tick labels
    # X-axis ticks
    num_x_ticks = 5
    for i in range(num_x_ticks):
        x_val = x_range[0] + (x_range[1] - x_range[0]) * i / (num_x_ticks - 1)
        x_pos = margin['left'] + int(orig_width * i / (num_x_ticks - 1))
        tick_text = f"{x_val:.0f}"
        draw.text((x_pos - 10, margin['top'] + orig_height + 5), tick_text, fill='black', font=font_tick)
    
    # Y-axis ticks
    num_y_ticks = 5
    for i in range(num_y_ticks):
        y_val = y_range[0] + (y_range[1] - y_range[0]) * i / (num_y_ticks - 1)
        y_pos = margin['top'] + orig_height - int(orig_height * i / (num_y_ticks - 1))
        tick_text = f"{y_val:.1f}"
        draw.text((margin['left'] - 50, y_pos - 5), tick_text, fill='black', font=font_tick)
    
    # Add stats text box
    if stats_text:
        # Draw background rectangle
        stats_lines = stats_text.split('\n')
        max_line_width = max(draw.textbbox((0, 0), line, font=font_tick)[2] for line in stats_lines)
        stats_height = len(stats_lines) * 15
        stats_x = margin['left'] + 10
        stats_y = margin['top'] + 10
        draw.rectangle(
            [stats_x - 5, stats_y - 5, stats_x + max_line_width + 10, stats_y + stats_height + 5],
            fill=(255, 248, 220, 200),  # wheat color with alpha
            outline=(180, 160, 120)
        )
        for idx, line in enumerate(stats_lines):
            draw.text((stats_x, stats_y + idx * 15), line, fill='black', font=font_tick)
    
    # Add legend
    if legend_items:
        legend_x = margin['left'] + orig_width + 10
        legend_y = margin['top'] + 10
        for idx, (label, color) in enumerate(legend_items):
            # Draw color box
            draw.rectangle(
                [legend_x, legend_y + idx * 20, legend_x + 15, legend_y + idx * 20 + 10],
                fill=color, outline='black'
            )
            draw.text((legend_x + 20, legend_y + idx * 20 - 2), label, fill='black', font=font_tick)
    
    return annotated


def _datashader_line_plot(
    x_values: np.ndarray,
    y_values: np.ndarray,
    pixel_width: int = 800,
    pixel_height: int = 600,
    line_color: str = 'blue',
    x_range: Tuple[float, float] = None,
    y_range: Tuple[float, float] = None,
) -> "Image.Image":
    """
    Create a line plot using datashader.
    
    Args:
        x_values: X coordinates
        y_values: Y coordinates
        pixel_width: Output image width in pixels
        pixel_height: Output image height in pixels
        line_color: Color for the line
        x_range: Optional (min, max) for x-axis
        y_range: Optional (min, max) for y-axis
    
    Returns:
        PIL Image of the plot
    """
    if not HAS_DATASHADER or ds is None:
        raise RuntimeError("datashader not available")
    
    # Create DataFrame for datashader
    df = pd.DataFrame({'x': x_values, 'y': y_values})
    
    # Remove NaN values
    df = df.dropna()
    
    if len(df) < 2:
        # Return blank image if not enough data
        if HAS_PIL and Image is not None:
            return Image.new('RGBA', (pixel_width, pixel_height), (255, 255, 255, 255))
        return None
    
    # Determine ranges
    if x_range is None:
        x_range = (float(df['x'].min()), float(df['x'].max()))
    if y_range is None:
        y_min, y_max = float(df['y'].min()), float(df['y'].max())
        y_padding = (y_max - y_min) * 0.1 if y_max > y_min else 1.0
        y_range = (y_min - y_padding, y_max + y_padding)
    
    # Create canvas
    canvas = ds.Canvas(plot_width=pixel_width, plot_height=pixel_height,
                       x_range=x_range, y_range=y_range)
    
    # Aggregate line
    agg = canvas.line(df, 'x', 'y', agg=ds.count())
    
    # Apply color
    color_map = {
        'blue': (0, 100, 200),
        'red': (200, 50, 50),
        'green': (50, 150, 50),
        'orange': (255, 165, 0),
        'black': (0, 0, 0),
    }
    rgb = color_map.get(line_color, (0, 100, 200))
    
    # Create image
    img = tf.shade(agg, cmap=[f'rgb{rgb}'], how='linear')
    img = tf.set_background(img, 'white')
    
    # Convert to PIL Image
    return img.to_pil()


def _datashader_multi_line_plot(
    line_data: List[Tuple[np.ndarray, np.ndarray, str, str]],
    pixel_width: int = 800,
    pixel_height: int = 600,
    x_range: Tuple[float, float] = None,
    y_range: Tuple[float, float] = None,
) -> "Image.Image":
    """
    Create a multi-line plot using datashader with different colors.
    
    Args:
        line_data: List of (x_values, y_values, color, label) tuples
        pixel_width: Output image width in pixels
        pixel_height: Output image height in pixels
        x_range: Optional (min, max) for x-axis
        y_range: Optional (min, max) for y-axis
    
    Returns:
        PIL Image of the plot
    """
    if not HAS_DATASHADER or ds is None:
        raise RuntimeError("datashader not available")
    
    if not line_data:
        if HAS_PIL and Image is not None:
            return Image.new('RGBA', (pixel_width, pixel_height), (255, 255, 255, 255))
        return None
    
    # Build combined DataFrame with category column
    dfs = []
    for idx, (x_vals, y_vals, color, label) in enumerate(line_data):
        df = pd.DataFrame({'x': x_vals, 'y': y_vals})
        df = df.dropna()
        if len(df) > 0:
            df['category'] = label
            dfs.append(df)
    
    if not dfs:
        if HAS_PIL and Image is not None:
            return Image.new('RGBA', (pixel_width, pixel_height), (255, 255, 255, 255))
        return None
    
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df['category'] = combined_df['category'].astype('category')
    
    # Determine ranges
    if x_range is None:
        x_range = (float(combined_df['x'].min()), float(combined_df['x'].max()))
    if y_range is None:
        y_min, y_max = float(combined_df['y'].min()), float(combined_df['y'].max())
        y_padding = (y_max - y_min) * 0.1 if y_max > y_min else 1.0
        y_range = (y_min - y_padding, y_max + y_padding)
    
    # Create canvas
    canvas = ds.Canvas(plot_width=pixel_width, plot_height=pixel_height,
                       x_range=x_range, y_range=y_range)
    
    # Create color mapping
    color_map = {
        'blue': '#0064C8',
        'red': '#C83232',
        'green': '#329632',
        'orange': '#FFA500',
        'purple': '#800080',
        'cyan': '#00FFFF',
        'magenta': '#FF00FF',
        'yellow': '#FFD700',
        'brown': '#8B4513',
        'pink': '#FF69B4',
    }
    
    # Assign colors to categories
    categories = combined_df['category'].cat.categories.tolist()
    available_colors = list(color_map.values())
    category_colors = {}
    for idx, cat in enumerate(categories):
        # Find the color from line_data
        for x_vals, y_vals, color, label in line_data:
            if label == cat:
                category_colors[cat] = color_map.get(color, available_colors[idx % len(available_colors)])
                break
        else:
            category_colors[cat] = available_colors[idx % len(available_colors)]
    
    # Aggregate with category
    agg = canvas.line(combined_df, 'x', 'y', agg=ds.count_cat('category'))
    
    # Shade with category colors
    img = tf.shade(agg, color_key=category_colors, how='linear')
    img = tf.set_background(img, 'white')
    
    return img.to_pil()


def _add_horizontal_lines(
    img: "Image.Image",
    h_lines: List[Tuple[float, str, str]],
    y_range: Tuple[float, float],
    margin: dict = None,
) -> "Image.Image":
    """
    Add horizontal reference lines to an image.
    
    Args:
        img: PIL Image
        h_lines: List of (y_value, color, style) tuples
        y_range: (min, max) for y-axis to calculate position
        margin: Margins dict
    
    Returns:
        Modified PIL Image
    """
    if not HAS_PIL or ImageDraw is None:
        return img
    
    if margin is None:
        margin = {'top': 60, 'bottom': 50, 'left': 80, 'right': 150}
    
    draw = ImageDraw.Draw(img)
    
    plot_height = img.size[1] - margin['top'] - margin['bottom']
    y_min, y_max = y_range
    
    for y_val, color, style in h_lines:
        if y_min <= y_val <= y_max:
            # Calculate pixel position
            y_frac = (y_val - y_min) / (y_max - y_min) if y_max > y_min else 0.5
            y_pixel = margin['top'] + int(plot_height * (1 - y_frac))
            
            # Get color
            color_map = {
                'red': (200, 50, 50),
                'orange': (255, 165, 0),
                'blue': (0, 100, 200),
                'green': (50, 150, 50),
                'black': (0, 0, 0),
            }
            rgb = color_map.get(color, (128, 128, 128))
            
            # Draw line (dashed or solid)
            x_start = margin['left']
            x_end = img.size[0] - margin['right']
            
            if style == 'dashed':
                dash_len = 10
                gap_len = 5
                x = x_start
                while x < x_end:
                    draw.line([(x, y_pixel), (min(x + dash_len, x_end), y_pixel)], fill=rgb, width=1)
                    x += dash_len + gap_len
            elif style == 'dotted':
                x = x_start
                while x < x_end:
                    draw.point((x, y_pixel), fill=rgb)
                    x += 3
            else:
                draw.line([(x_start, y_pixel), (x_end, y_pixel)], fill=rgb, width=1)
    
    return img


def _datashader_scatter_plot(
    scatter_data: List[Tuple[np.ndarray, np.ndarray, str, str]],
    pixel_width: int = 800,
    pixel_height: int = 600,
    x_range: Tuple[float, float] = None,
    y_range: Tuple[float, float] = None,
    point_size: int = 5,
) -> "Image.Image":
    """
    Create a scatter plot using datashader with different colors for categories.
    
    Args:
        scatter_data: List of (x_values, y_values, color, label) tuples
        pixel_width: Output image width in pixels
        pixel_height: Output image height in pixels
        x_range: Optional (min, max) for x-axis
        y_range: Optional (min, max) for y-axis
        point_size: Size of points in pixels
    
    Returns:
        PIL Image of the plot
    """
    if not HAS_DATASHADER or ds is None:
        raise RuntimeError("datashader not available")
    
    if not scatter_data:
        if HAS_PIL and Image is not None:
            return Image.new('RGBA', (pixel_width, pixel_height), (255, 255, 255, 255))
        return None
    
    # Build combined DataFrame with category column
    dfs = []
    for idx, (x_vals, y_vals, color, label) in enumerate(scatter_data):
        if len(x_vals) == 0:
            continue
        df = pd.DataFrame({'x': np.array(x_vals, dtype=float), 'y': np.array(y_vals, dtype=float)})
        df = df.dropna()
        if len(df) > 0:
            df['category'] = label
            dfs.append(df)
    
    if not dfs:
        if HAS_PIL and Image is not None:
            return Image.new('RGBA', (pixel_width, pixel_height), (255, 255, 255, 255))
        return None
    
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df['category'] = combined_df['category'].astype('category')
    
    # Determine ranges
    if x_range is None:
        x_min, x_max = float(combined_df['x'].min()), float(combined_df['x'].max())
        x_padding = (x_max - x_min) * 0.05 if x_max > x_min else 1.0
        x_range = (x_min - x_padding, x_max + x_padding)
    if y_range is None:
        y_min, y_max = float(combined_df['y'].min()), float(combined_df['y'].max())
        y_padding = (y_max - y_min) * 0.1 if y_max > y_min else 1.0
        y_range = (y_min - y_padding, y_max + y_padding)
    
    # Create canvas
    canvas = ds.Canvas(plot_width=pixel_width, plot_height=pixel_height,
                       x_range=x_range, y_range=y_range)
    
    # Create color mapping
    color_map = {
        'blue': '#0064C8',
        'red': '#C83232',
        'green': '#329632',
        'orange': '#FFA500',
        'purple': '#800080',
        'cyan': '#00FFFF',
        'magenta': '#FF00FF',
        'yellow': '#FFD700',
        'brown': '#8B4513',
        'pink': '#FF69B4',
        'lightblue': '#87CEEB',
        'darkgreen': '#006400',
        'darkred': '#8B0000',
    }
    
    # Assign colors to categories
    categories = combined_df['category'].cat.categories.tolist()
    category_colors = {}
    available_colors = list(color_map.values())
    for idx, cat in enumerate(categories):
        # Find the color from scatter_data
        for x_vals, y_vals, color, label in scatter_data:
            if label == cat:
                category_colors[cat] = color_map.get(color, available_colors[idx % len(available_colors)])
                break
        else:
            category_colors[cat] = available_colors[idx % len(available_colors)]
    
    # Aggregate with category
    agg = canvas.points(combined_df, 'x', 'y', agg=ds.count_cat('category'))
    
    # Shade with category colors
    img = tf.shade(agg, color_key=category_colors, how='linear')
    
    # Spread points to make them more visible
    img = tf.spread(img, px=point_size)
    img = tf.set_background(img, 'white')
    
    return img.to_pil()


def _draw_scatter_markers_pil(
    img: "Image.Image",
    scatter_data: List[Tuple[List, List, str, str, str]],
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    margin: dict = None,
    marker_size: int = 8,
) -> "Image.Image":
    """
    Draw scatter markers using PIL for better control over marker shapes.
    
    Args:
        img: PIL Image to draw on
        scatter_data: List of (x_values, y_values, color, label, marker) tuples
                     marker can be 'triangle_up', 'triangle_down', 'circle', 'square'
        x_range: (min, max) for x-axis
        y_range: (min, max) for y-axis
        margin: Margins dict
        marker_size: Size of markers
    
    Returns:
        Modified PIL Image
    """
    if not HAS_PIL or ImageDraw is None:
        return img
    
    if margin is None:
        margin = {'top': 60, 'bottom': 50, 'left': 80, 'right': 150}
    
    draw = ImageDraw.Draw(img)
    
    plot_width = img.size[0] - margin['left'] - margin['right']
    plot_height = img.size[1] - margin['top'] - margin['bottom']
    x_min, x_max = x_range
    y_min, y_max = y_range
    
    color_map = {
        'blue': (0, 100, 200),
        'red': (200, 50, 50),
        'green': (50, 150, 50),
        'orange': (255, 165, 0),
        'purple': (128, 0, 128),
        'lightblue': (135, 206, 235),
        'darkgreen': (0, 100, 0),
        'darkred': (139, 0, 0),
    }
    
    for x_vals, y_vals, color, label, marker in scatter_data:
        rgb = color_map.get(color, (128, 128, 128))
        edge_color = tuple(max(0, c - 50) for c in rgb)  # Darker edge
        
        for x_val, y_val in zip(x_vals, y_vals):
            # Calculate pixel position
            x_frac = (x_val - x_min) / (x_max - x_min) if x_max > x_min else 0.5
            y_frac = (y_val - y_min) / (y_max - y_min) if y_max > y_min else 0.5
            x_pixel = margin['left'] + int(plot_width * x_frac)
            y_pixel = margin['top'] + int(plot_height * (1 - y_frac))
            
            s = marker_size
            if marker == 'triangle_up':
                points = [
                    (x_pixel, y_pixel - s),
                    (x_pixel - s, y_pixel + s),
                    (x_pixel + s, y_pixel + s),
                ]
                draw.polygon(points, fill=rgb, outline=edge_color)
            elif marker == 'triangle_down':
                points = [
                    (x_pixel, y_pixel + s),
                    (x_pixel - s, y_pixel - s),
                    (x_pixel + s, y_pixel - s),
                ]
                draw.polygon(points, fill=rgb, outline=edge_color)
            elif marker == 'square':
                draw.rectangle(
                    [x_pixel - s, y_pixel - s, x_pixel + s, y_pixel + s],
                    fill=rgb, outline=edge_color
                )
            else:  # circle
                draw.ellipse(
                    [x_pixel - s, y_pixel - s, x_pixel + s, y_pixel + s],
                    fill=rgb, outline=edge_color
                )
    
    return img


def _add_vertical_line(
    img: "Image.Image",
    x_val: float,
    x_range: Tuple[float, float],
    color: str = 'black',
    style: str = 'dashed',
    margin: dict = None,
) -> "Image.Image":
    """Add a vertical line to the image."""
    if not HAS_PIL or ImageDraw is None:
        return img
    
    if margin is None:
        margin = {'top': 60, 'bottom': 50, 'left': 80, 'right': 150}
    
    draw = ImageDraw.Draw(img)
    
    plot_width = img.size[0] - margin['left'] - margin['right']
    x_min, x_max = x_range
    
    if x_min <= x_val <= x_max:
        x_frac = (x_val - x_min) / (x_max - x_min) if x_max > x_min else 0.5
        x_pixel = margin['left'] + int(plot_width * x_frac)
        
        color_map = {
            'red': (200, 50, 50),
            'orange': (255, 165, 0),
            'blue': (0, 100, 200),
            'green': (50, 150, 50),
            'black': (0, 0, 0),
        }
        rgb = color_map.get(color, (0, 0, 0))
        
        y_start = margin['top']
        y_end = img.size[1] - margin['bottom']
        
        if style == 'dashed':
            dash_len = 10
            gap_len = 5
            y = y_start
            while y < y_end:
                draw.line([(x_pixel, y), (x_pixel, min(y + dash_len, y_end))], fill=rgb, width=2)
                y += dash_len + gap_len
        else:
            draw.line([(x_pixel, y_start), (x_pixel, y_end)], fill=rgb, width=2)
    
    return img


def _generate_timeline_frame(
    current_frame: int,
    entry_frames: List[int],
    exit_frames: List[int],
    entry_guest_indices: List[List[int]],
    exit_guest_indices: List[List[int]],
    sorted_guest_indices: List[int],
    guest_index_to_y: dict,
    guest_residence_stats: dict,
    x_min: float,
    x_max: float,
    figure_size: Tuple[int, int],
    dpi: int,
    show_connections: bool,
) -> np.ndarray:
    """
    Helper function to generate a single frame for the timeline GIF.
    This is a module-level function to enable multiprocessing.
    Uses datashader + PIL for rendering (no matplotlib dependency).
    """
    # Import PIL (needed in worker processes)
    try:
        from PIL import Image as PILImage, ImageDraw as PILImageDraw, ImageFont as PILImageFont
    except ImportError:
        # Fall back to matplotlib if PIL not available
        return _generate_timeline_frame_matplotlib(
            current_frame, entry_frames, exit_frames, entry_guest_indices,
            exit_guest_indices, sorted_guest_indices, guest_index_to_y,
            guest_residence_stats, x_min, x_max, figure_size, dpi, show_connections
        )
    
    # Calculate pixel dimensions
    pixel_width = int(figure_size[0] * dpi / 4)  # Reduced for reasonable file size
    pixel_height = int(figure_size[1] * dpi / 4)
    
    # Y range based on number of guests
    y_min = -0.5
    y_max = len(sorted_guest_indices) - 0.5
    
    margin = {'top': 60, 'bottom': 50, 'left': 80, 'right': 180}
    new_width = pixel_width + margin['left'] + margin['right']
    new_height = pixel_height + margin['top'] + margin['bottom']
    
    # Create final image
    final_img = PILImage.new('RGBA', (new_width, new_height), (255, 255, 255, 255))
    draw = PILImageDraw.Draw(final_img)
    
    # Draw connections if requested
    if show_connections:
        guest_timelines = {gidx: [] for gidx in sorted_guest_indices}
        
        for entry_frame, guest_indices in zip(entry_frames, entry_guest_indices):
            if entry_frame <= current_frame:
                for gidx in guest_indices:
                    if gidx in guest_timelines:
                        guest_timelines[gidx].append({'entry': entry_frame, 'exit': None})
        
        for exit_frame, guest_indices in zip(exit_frames, exit_guest_indices):
            if exit_frame <= current_frame:
                for gidx in guest_indices:
                    if gidx in guest_timelines:
                        for timeline_entry in reversed(guest_timelines[gidx]):
                            if timeline_entry['exit'] is None:
                                timeline_entry['exit'] = exit_frame
                                break
        
        # Draw horizontal lines for connections
        for gidx, timeline in guest_timelines.items():
            y_pos = guest_index_to_y[gidx]
            for period in timeline:
                ef = period['entry']
                ex_f = period['exit'] if period['exit'] is not None else current_frame
                
                # Calculate pixel positions
                x1_frac = (ef - x_min) / (x_max - x_min) if x_max > x_min else 0
                x2_frac = (ex_f - x_min) / (x_max - x_min) if x_max > x_min else 0
                y_frac = (y_pos - y_min) / (y_max - y_min) if y_max > y_min else 0.5
                
                x1_pixel = margin['left'] + int(pixel_width * x1_frac)
                x2_pixel = margin['left'] + int(pixel_width * x2_frac)
                y_pixel = margin['top'] + int(pixel_height * (1 - y_frac))
                
                draw.line([(x1_pixel, y_pixel), (x2_pixel, y_pixel)],
                         fill=(135, 206, 235, 180), width=4)

    # Build scatter data
    entry_x = []
    entry_y = []
    for entry_frame, guest_indices in zip(entry_frames, entry_guest_indices):
        if entry_frame <= current_frame:
            for gidx in guest_indices:
                if gidx in guest_index_to_y:
                    entry_x.append(entry_frame)
                    entry_y.append(guest_index_to_y[gidx])

    exit_x = []
    exit_y = []
    for exit_frame, guest_indices in zip(exit_frames, exit_guest_indices):
        if exit_frame <= current_frame:
            for gidx in guest_indices:
                if gidx in guest_index_to_y:
                    exit_x.append(exit_frame)
                    exit_y.append(guest_index_to_y[gidx])

    # Color mapping
    color_map = {
        'green': (50, 150, 50),
        'red': (200, 50, 50),
        'black': (0, 0, 0),
    }
    
    # Draw scatter markers
    marker_size = 8
    
    # Entry markers (triangle up)
    for x_val, y_val in zip(entry_x, entry_y):
        x_frac = (x_val - x_min) / (x_max - x_min) if x_max > x_min else 0.5
        y_frac = (y_val - y_min) / (y_max - y_min) if y_max > y_min else 0.5
        x_pixel = margin['left'] + int(pixel_width * x_frac)
        y_pixel = margin['top'] + int(pixel_height * (1 - y_frac))
        
        s = marker_size
        points = [(x_pixel, y_pixel - s), (x_pixel - s, y_pixel + s), (x_pixel + s, y_pixel + s)]
        draw.polygon(points, fill=color_map['green'], outline=(0, 100, 0))
    
    # Exit markers (triangle down)
    for x_val, y_val in zip(exit_x, exit_y):
        x_frac = (x_val - x_min) / (x_max - x_min) if x_max > x_min else 0.5
        y_frac = (y_val - y_min) / (y_max - y_min) if y_max > y_min else 0.5
        x_pixel = margin['left'] + int(pixel_width * x_frac)
        y_pixel = margin['top'] + int(pixel_height * (1 - y_frac))
        
        s = marker_size
        points = [(x_pixel, y_pixel + s), (x_pixel - s, y_pixel - s), (x_pixel + s, y_pixel - s)]
        draw.polygon(points, fill=color_map['red'], outline=(139, 0, 0))
    
    # Draw vertical line for current frame
    x_frac = (current_frame - x_min) / (x_max - x_min) if x_max > x_min else 0.5
    x_pixel = margin['left'] + int(pixel_width * x_frac)
    
    # Dashed vertical line
    dash_len = 10
    gap_len = 5
    y = margin['top']
    y_end = margin['top'] + pixel_height
    while y < y_end:
        draw.line([(x_pixel, y), (x_pixel, min(y + dash_len, y_end))], fill=(0, 0, 0), width=2)
        y += dash_len + gap_len
    
    # Try to get fonts
    try:
        font_title = PILImageFont.truetype("arial.ttf", 16)
        font_label = PILImageFont.truetype("arial.ttf", 12)
        font_tick = PILImageFont.truetype("arial.ttf", 10)
    except (OSError, IOError):
        try:
            font_title = PILImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
            font_label = PILImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
            font_tick = PILImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
        except (OSError, IOError):
            font_title = PILImageFont.load_default()
            font_label = font_title
            font_tick = font_title
    
    # Add title
    title = f"Guest Entry/Exit Timeline (Frame {current_frame})"
    title_bbox = draw.textbbox((0, 0), title, font=font_title)
    title_width = title_bbox[2] - title_bbox[0]
    title_x = margin['left'] + (pixel_width - title_width) // 2
    draw.text((title_x, 10), title, fill='black', font=font_title)
    
    # Add x-axis label
    xlabel = "Frame"
    xlabel_bbox = draw.textbbox((0, 0), xlabel, font=font_label)
    xlabel_width = xlabel_bbox[2] - xlabel_bbox[0]
    xlabel_x = margin['left'] + (pixel_width - xlabel_width) // 2
    draw.text((xlabel_x, new_height - 25), xlabel, fill='black', font=font_label)
    
    # Add y-axis label (rotated)
    ylabel = "Guest Index"
    ylabel_img = PILImage.new('RGBA', (200, 20), (255, 255, 255, 0))
    ylabel_draw = PILImageDraw.Draw(ylabel_img)
    ylabel_draw.text((0, 0), ylabel, fill='black', font=font_label)
    ylabel_img = ylabel_img.rotate(90, expand=True)
    ylabel_y = margin['top'] + (pixel_height - ylabel_img.size[1]) // 2
    final_img.paste(ylabel_img, (5, ylabel_y), ylabel_img)
    
    # X-axis ticks
    num_x_ticks = 5
    for i in range(num_x_ticks):
        x_val = x_min + (x_max - x_min) * i / (num_x_ticks - 1)
        x_pos = margin['left'] + int(pixel_width * i / (num_x_ticks - 1))
        tick_text = f"{x_val:.0f}"
        draw.text((x_pos - 10, margin['top'] + pixel_height + 5), tick_text, fill='black', font=font_tick)
    
    # Y-axis ticks (guest indices)
    for i, gidx in enumerate(sorted_guest_indices):
        y_frac = (i - y_min) / (y_max - y_min) if y_max > y_min else 0.5
        y_pos = margin['top'] + int(pixel_height * (1 - y_frac))
        draw.text((margin['left'] - 30, y_pos - 5), str(gidx), fill='black', font=font_tick)
    
    # Legend
    legend_x = margin['left'] + pixel_width + 10
    legend_y = margin['top'] + 10
    legend_items = [
        (f'Entry (n={len(entry_x)})', (50, 150, 50)),
        (f'Exit (n={len(exit_x)})', (200, 50, 50)),
        ('Current frame', (0, 0, 0)),
    ]
    for idx, (label, color) in enumerate(legend_items):
        draw.rectangle([legend_x, legend_y + idx * 20, legend_x + 15, legend_y + idx * 20 + 10],
                      fill=color, outline='black')
        draw.text((legend_x + 20, legend_y + idx * 20 - 2), label, fill='black', font=font_tick)
    
    # Statistics text box
    stats_lines = []
    if guest_residence_stats.get('n_entries', 0) > 0:
        stats_lines.append(f"Total entries: {guest_residence_stats['n_entries']}")
        stats_lines.append(f"Total exits: {guest_residence_stats['n_exits']}")
        stats_lines.append(f"Unique guests: {len(sorted_guest_indices)}")
        if guest_residence_stats.get('first_entry_frame') is not None:
            stats_lines.append(f"First entry: Frame {guest_residence_stats['first_entry_frame']}")
    
    if stats_lines:
        stats_text = '\n'.join(stats_lines)
        # Calculate position (bottom right of plot area)
        stats_x = margin['left'] + pixel_width - 150
        stats_y = margin['top'] + pixel_height - len(stats_lines) * 15 - 15
        max_line_width = max(draw.textbbox((0, 0), line, font=font_tick)[2] for line in stats_lines)
        
        draw.rectangle(
            [stats_x - 5, stats_y - 5, stats_x + max_line_width + 10, stats_y + len(stats_lines) * 15 + 5],
            fill=(255, 248, 220, 200),
            outline=(180, 160, 120)
        )
        for idx, line in enumerate(stats_lines):
            draw.text((stats_x, stats_y + idx * 15), line, fill='black', font=font_tick)
    
    # Convert to numpy array
    return np.array(final_img)


def _generate_timeline_frame_matplotlib(
    current_frame: int,
    entry_frames: List[int],
    exit_frames: List[int],
    entry_guest_indices: List[List[int]],
    exit_guest_indices: List[List[int]],
    sorted_guest_indices: List[int],
    guest_index_to_y: dict,
    guest_residence_stats: dict,
    x_min: float,
    x_max: float,
    figure_size: Tuple[int, int],
    dpi: int,
    show_connections: bool,
) -> np.ndarray:
    """
    Matplotlib fallback for timeline frame generation.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    try:
        import imageio.v2 as imageio
    except ImportError:
        import imageio
    
    fig, ax = plt.subplots(figsize=figure_size)
    
    if show_connections:
        guest_timelines = {gidx: [] for gidx in sorted_guest_indices}
        for entry_frame, guest_indices in zip(entry_frames, entry_guest_indices):
            if entry_frame <= current_frame:
                for gidx in guest_indices:
                    if gidx in guest_timelines:
                        guest_timelines[gidx].append({'entry': entry_frame, 'exit': None})
        for exit_frame, guest_indices in zip(exit_frames, exit_guest_indices):
            if exit_frame <= current_frame:
                for gidx in guest_indices:
                    if gidx in guest_timelines:
                        for timeline_entry in reversed(guest_timelines[gidx]):
                            if timeline_entry['exit'] is None:
                                timeline_entry['exit'] = exit_frame
                                break
        for gidx, timeline in guest_timelines.items():
            y_pos = guest_index_to_y[gidx]
            for period in timeline:
                ef = period['entry']
                ex_f = period['exit'] if period['exit'] is not None else current_frame
                ax.plot([ef, ex_f], [y_pos, y_pos], color='lightblue', linewidth=3, alpha=0.5, zorder=1)

    entry_x, entry_y = [], []
    for entry_frame, guest_indices in zip(entry_frames, entry_guest_indices):
        if entry_frame <= current_frame:
            for gidx in guest_indices:
                if gidx in guest_index_to_y:
                    entry_x.append(entry_frame)
                    entry_y.append(guest_index_to_y[gidx])
    if entry_x:
        ax.scatter(entry_x, entry_y, color='green', marker='^', s=150, zorder=3,
                  label=f'Entry events (n={len(entry_x)})', edgecolors='darkgreen', linewidths=1.5)

    exit_x, exit_y = [], []
    for exit_frame, guest_indices in zip(exit_frames, exit_guest_indices):
        if exit_frame <= current_frame:
            for gidx in guest_indices:
                if gidx in guest_index_to_y:
                    exit_x.append(exit_frame)
                    exit_y.append(guest_index_to_y[gidx])
    if exit_x:
        ax.scatter(exit_x, exit_y, color='red', marker='v', s=150, zorder=3,
                  label=f'Exit events (n={len(exit_x)})', edgecolors='darkred', linewidths=1.5)

    ax.axvline(x=current_frame, color='black', linestyle='--', linewidth=2, alpha=0.5, zorder=2, label='Current frame')
    ax.set_xlabel("Frame", fontsize=12)
    ax.set_ylabel("Guest Index", fontsize=12)
    ax.set_title(f"Guest Entry/Exit Timeline (Frame {current_frame})", fontsize=14, fontweight="bold")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.5, len(sorted_guest_indices) - 0.5)
    ax.set_yticks(range(len(sorted_guest_indices)))
    ax.set_yticklabels([str(gidx) for gidx in sorted_guest_indices])
    ax.grid(True, alpha=0.3, axis='x')
    ax.legend(loc='upper left', fontsize=9)

    stats_text = []
    if guest_residence_stats.get('n_entries', 0) > 0:
        stats_text.append(f"Total entries: {guest_residence_stats['n_entries']}")
        stats_text.append(f"Total exits: {guest_residence_stats['n_exits']}")
        stats_text.append(f"Unique guests: {len(sorted_guest_indices)}")
        if guest_residence_stats.get('first_entry_frame') is not None:
            stats_text.append(f"First entry: Frame {guest_residence_stats['first_entry_frame']}")
    if stats_text:
        ax.text(0.98, 0.02, '\n'.join(stats_text), transform=ax.transAxes, fontsize=10,
               verticalalignment='bottom', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    img = imageio.imread(buf)
    return img


class Plotter:
    """Handle plotting operations for trajectory analysis."""

    def __init__(self) -> None:
        self.output_prefix: Optional[str] = None
        self.plots_generated: List[str] = []
        self.figure_size: Tuple[int, int] = (12, 8)
        self.dpi: int = 300

    def plot_endpoint_distances(
        self,
        endpoint_dists_array: Union[np.ndarray, dict],
        residue_sel_list: List[str],
        out_prefix: str,
    ) -> None:
        """Plot distances between endpoint pairs over frames using datashader."""
        if not HAS_DATASHADER or ds is None:
            warnings.warn("datashader not available. Skipping endpoint distance plots.")
            return

        if endpoint_dists_array is None:
            warnings.warn("Endpoint distances array is None. Skipping plots.")
            return

        if isinstance(endpoint_dists_array, dict):
            all_pairs = endpoint_dists_array.get("all_pairs", {})
            T = endpoint_dists_array.get("n_frames", 0)
            n_res = endpoint_dists_array.get("n_residues", 0)

            if not all_pairs:
                warnings.warn("Endpoint distances array is empty. Skipping plots.")
                return

            frames = np.arange(T)
            
            # Collect line data for multi-line plot
            line_data = []
            colors = ['blue', 'red', 'green', 'orange', 'purple', 'cyan', 'magenta', 'brown', 'pink', 'yellow']
            color_idx = 0
            
            for i in range(n_res):
                for j in range(i + 1, n_res):
                    if (i, j) in all_pairs:
                        pair_array = all_pairs[(i, j)]
                        mean_dists = []
                        for t in range(T):
                            frame_dists = pair_array[t]
                            valid_dists = frame_dists[~np.isnan(frame_dists)]
                            if len(valid_dists) > 0:
                                mean_dists.append(np.mean(valid_dists))
                            else:
                                mean_dists.append(np.nan)
                        mean_dists = np.array(mean_dists)
                        if not np.all(np.isnan(mean_dists)):
                            label = f"Res {i}-{j}"
                            color = colors[color_idx % len(colors)]
                            line_data.append((frames.astype(float), mean_dists, color, label))
                            color_idx += 1

            if not line_data:
                warnings.warn("No valid endpoint distance data to plot.")
                return

            # Calculate pixel dimensions
            pixel_width = int(self.figure_size[0] * self.dpi / 4)
            pixel_height = int(self.figure_size[1] * self.dpi / 4)

            try:
                # Create multi-line plot
                img = _datashader_multi_line_plot(
                    line_data,
                    pixel_width=pixel_width,
                    pixel_height=pixel_height,
                )
                
                if img is None:
                    warnings.warn("Failed to create datashader plot.")
                    return

                # Calculate y_range
                all_y = np.concatenate([y for _, y, _, _ in line_data])
                valid_y = all_y[~np.isnan(all_y)]
                if len(valid_y) > 0:
                    y_min, y_max = float(valid_y.min()), float(valid_y.max())
                    y_padding = (y_max - y_min) * 0.1 if y_max > y_min else 1.0
                    y_range = (y_min - y_padding, y_max + y_padding)
                else:
                    y_range = (0, 1)
                x_range = (0, float(T - 1)) if T > 1 else (0, 1)

                # Build legend items
                legend_items = [(label, color) for _, _, color, label in line_data]

                # Add text annotations
                margin = {'top': 60, 'bottom': 50, 'left': 80, 'right': 180}
                img = _add_text_annotations(
                    img,
                    title="Endpoint Pair Distances Over Frames (Mean)",
                    xlabel="Frame",
                    ylabel="Endpoint Distance (Å)",
                    stats_text="",
                    legend_items=legend_items,
                    x_range=x_range,
                    y_range=y_range,
                    margin=margin,
                )

                # Save image
                plot_path = f"{out_prefix}_endpoint_distances.png"
                img.save(plot_path)
                self.output_prefix = out_prefix
                self.plots_generated.append(plot_path)
                print(f"Endpoint distance plot saved to {plot_path}")
                
            except Exception as e:
                warnings.warn(f"Failed to create datashader plot: {e}. Trying matplotlib fallback.")
                if HAS_MATPLOTLIB and plt is not None:
                    self._plot_endpoint_distances_matplotlib(endpoint_dists_array, residue_sel_list, out_prefix)
    
    def _plot_endpoint_distances_matplotlib(
        self,
        endpoint_dists_array: Union[np.ndarray, dict],
        residue_sel_list: List[str],
        out_prefix: str,
    ) -> None:
        """Matplotlib fallback for endpoint distances plot."""
        if not HAS_MATPLOTLIB or plt is None:
            return

        if isinstance(endpoint_dists_array, dict):
            all_pairs = endpoint_dists_array.get("all_pairs", {})
            T = endpoint_dists_array.get("n_frames", 0)
            n_res = endpoint_dists_array.get("n_residues", 0)

            if not all_pairs:
                return

            frames = np.arange(T)
            fig, ax = plt.subplots(figsize=self.figure_size)

            for i in range(n_res):
                for j in range(i + 1, n_res):
                    if (i, j) in all_pairs:
                        pair_array = all_pairs[(i, j)]
                        mean_dists = []
                        for t in range(T):
                            frame_dists = pair_array[t]
                            valid_dists = frame_dists[~np.isnan(frame_dists)]
                            if len(valid_dists) > 0:
                                mean_dists.append(np.mean(valid_dists))
                            else:
                                mean_dists.append(np.nan)
                        mean_dists = np.array(mean_dists)
                        if not np.all(np.isnan(mean_dists)):
                            label = f"Res {i}-{j} (mean)"
                            ax.plot(frames, mean_dists, label=label, alpha=0.7, linewidth=1.5)

            ax.set_xlabel("Frame", fontsize=12)
            ax.set_ylabel("Endpoint Distance (Å)", fontsize=12)
            ax.set_title("Endpoint Pair Distances Over Frames (Mean)", fontsize=14, fontweight="bold")
            ax.grid(True, alpha=0.3)
            handles, labels = ax.get_legend_handles_labels()
            if labels:
                ax.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8, ncol=1)
            plt.tight_layout()
            plot_path = f"{out_prefix}_endpoint_distances.png"
            plt.savefig(plot_path, dpi=self.dpi, bbox_inches="tight")
            plt.close()
            self.output_prefix = out_prefix
            self.plots_generated.append(plot_path)
            print(f"Endpoint distance plot saved to {plot_path}")

    def plot_endpoint_volume_correlation(
        self,
        endpoint_dists_array: Union[np.ndarray, dict],
        volume: np.ndarray,
        residue_sel_list: List[str],
        correlation_df: pd.DataFrame,
        out_prefix: str,
        top_n: int = 5,
    ) -> None:
        """Plot the top N endpoint pairs with highest correlation to volume using datashader."""
        if not HAS_DATASHADER or ds is None or not HAS_PIL:
            warnings.warn("datashader or PIL not available. Skipping correlation plots.")
            return

        if endpoint_dists_array is None or correlation_df.empty:
            warnings.warn("No correlation data to plot.")
            return

        top_pairs = correlation_df.head(top_n)
        n_plots = min(top_n, len(top_pairs))
        if n_plots == 0:
            return

        # Calculate dimensions for each subplot
        subplot_width = int(10 * self.dpi / 4)
        subplot_height = int(3 * self.dpi / 4)
        margin = {'top': 40, 'bottom': 40, 'left': 80, 'right': 100}
        
        frames = np.arange(len(volume))
        is_dict_format = isinstance(endpoint_dists_array, dict)
        if is_dict_format:
            all_pairs = endpoint_dists_array.get("all_pairs", {})

        # Create list of subplot images
        subplot_images = []

        for idx, (_, row) in enumerate(top_pairs.iterrows()):
            i, j = int(row["residue_i"]), int(row["residue_j"])
            corr = row["correlation"]
            p_val = row["p_value"]
            ep_i_idx = row.get("ep_i_idx", None)
            ep_j_idx = row.get("ep_j_idx", None)

            if (
                is_dict_format
                and ep_i_idx is not None
                and ep_j_idx is not None
                and (i, j) in all_pairs
            ):
                pair_array = all_pairs[(i, j)]
                dists = pair_array[:, int(ep_i_idx), int(ep_j_idx)]
                ep_label = f"ep{int(ep_i_idx)}-ep{int(ep_j_idx)}"
            elif is_dict_format:
                if (i, j) in all_pairs:
                    pair_array = all_pairs[(i, j)]
                    mean_dists = []
                    for t in range(len(volume)):
                        frame_dists = pair_array[t]
                        valid_dists = frame_dists[~np.isnan(frame_dists)]
                        if len(valid_dists) > 0:
                            mean_dists.append(np.mean(valid_dists))
                        else:
                            mean_dists.append(np.nan)
                    dists = np.array(mean_dists)
                    ep_label = "mean"
                else:
                    warnings.warn(f"Cannot extract distances for pair ({i}, {j})")
                    continue
            else:
                dists = endpoint_dists_array[:, i, j]
                ep_label = "min"

            try:
                # Create subplot image
                full_width = subplot_width + margin['left'] + margin['right']
                full_height = subplot_height + margin['top'] + margin['bottom']
                subplot_img = Image.new('RGBA', (full_width, full_height), (255, 255, 255, 255))
                
                # Calculate ranges
                x_range = (0, float(len(volume) - 1)) if len(volume) > 1 else (0, 1)
                
                valid_dists = dists[~np.isnan(dists)]
                if len(valid_dists) > 0:
                    dist_min, dist_max = float(valid_dists.min()), float(valid_dists.max())
                    dist_padding = (dist_max - dist_min) * 0.1 if dist_max > dist_min else 1.0
                    dist_range = (dist_min - dist_padding, dist_max + dist_padding)
                else:
                    dist_range = (0, 1)
                
                valid_vol = volume[~np.isnan(volume)]
                if len(valid_vol) > 0:
                    vol_min, vol_max = float(valid_vol.min()), float(valid_vol.max())
                    vol_padding = (vol_max - vol_min) * 0.1 if vol_max > vol_min else 1.0
                    vol_range = (vol_min - vol_padding, vol_max + vol_padding)
                else:
                    vol_range = (0, 1)

                # Create line plots using datashader
                line_data = [
                    (frames.astype(float), dists, 'blue', 'Endpoint Distance'),
                    (frames.astype(float), volume, 'red', 'Cube Volume'),
                ]
                
                # Render combined lines (normalize both to same range for overlay)
                # We'll draw them as separate colored lines
                plot_img = _datashader_multi_line_plot(
                    line_data,
                    pixel_width=subplot_width,
                    pixel_height=subplot_height,
                )
                
                if plot_img is not None:
                    subplot_img.paste(plot_img, (margin['left'], margin['top']))
                
                draw = ImageDraw.Draw(subplot_img)
                
                # Try to get fonts
                try:
                    font_title = ImageFont.truetype("arial.ttf", 12)
                    font_label = ImageFont.truetype("arial.ttf", 10)
                    font_tick = ImageFont.truetype("arial.ttf", 9)
                except (OSError, IOError):
                    try:
                        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
                        font_label = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
                        font_tick = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 9)
                    except (OSError, IOError):
                        font_title = ImageFont.load_default()
                        font_label = font_title
                        font_tick = font_title

                # Add title
                title = f"Pair {i}-{j}"
                if ep_i_idx is not None and ep_j_idx is not None:
                    title += f" (ep{int(ep_i_idx)}-ep{int(ep_j_idx)})"
                title += f": r={corr:.3f}, p={p_val:.3e}"
                title_bbox = draw.textbbox((0, 0), title, font=font_title)
                title_x = margin['left'] + (subplot_width - (title_bbox[2] - title_bbox[0])) // 2
                draw.text((title_x, 5), title, fill='black', font=font_title)

                # Add x-axis label
                xlabel = "Frame"
                xlabel_bbox = draw.textbbox((0, 0), xlabel, font=font_label)
                xlabel_x = margin['left'] + (subplot_width - (xlabel_bbox[2] - xlabel_bbox[0])) // 2
                draw.text((xlabel_x, full_height - 20), xlabel, fill='black', font=font_label)

                # Add y-axis labels (left: distance in blue, right: volume in red)
                ylabel_dist = "Distance (Å)"
                ylabel_vol = "Volume (Å³)"
                
                # Left y-axis label
                ylabel_img = Image.new('RGBA', (150, 15), (255, 255, 255, 0))
                ylabel_draw = ImageDraw.Draw(ylabel_img)
                ylabel_draw.text((0, 0), ylabel_dist, fill=(0, 100, 200), font=font_label)
                ylabel_img = ylabel_img.rotate(90, expand=True)
                subplot_img.paste(ylabel_img, (5, margin['top'] + (subplot_height - ylabel_img.size[1]) // 2), ylabel_img)
                
                # Right y-axis label
                ylabel_img2 = Image.new('RGBA', (150, 15), (255, 255, 255, 0))
                ylabel_draw2 = ImageDraw.Draw(ylabel_img2)
                ylabel_draw2.text((0, 0), ylabel_vol, fill=(200, 50, 50), font=font_label)
                ylabel_img2 = ylabel_img2.rotate(-90, expand=True)
                subplot_img.paste(ylabel_img2, (full_width - 20, margin['top'] + (subplot_height - ylabel_img2.size[1]) // 2), ylabel_img2)

                # X-axis ticks
                num_x_ticks = 5
                for k in range(num_x_ticks):
                    x_val = x_range[0] + (x_range[1] - x_range[0]) * k / (num_x_ticks - 1)
                    x_pos = margin['left'] + int(subplot_width * k / (num_x_ticks - 1))
                    draw.text((x_pos - 10, margin['top'] + subplot_height + 5), f"{x_val:.0f}", fill='black', font=font_tick)

                # Legend
                legend_items = [
                    (f'Distance ({row.get("residue_i_sel", "")})', (0, 100, 200)),
                    ('Volume', (200, 50, 50)),
                ]
                legend_x = margin['left'] + 10
                legend_y = margin['top'] + 10
                for k, (label, color) in enumerate(legend_items):
                    draw.rectangle([legend_x, legend_y + k * 15, legend_x + 12, legend_y + k * 15 + 8],
                                  fill=color, outline='black')
                    draw.text((legend_x + 15, legend_y + k * 15 - 2), label[:30], fill='black', font=font_tick)

                subplot_images.append(subplot_img)
                
            except Exception as e:
                warnings.warn(f"Failed to create subplot {idx}: {e}")
                continue

        if not subplot_images:
            warnings.warn("No subplots created. Trying matplotlib fallback.")
            if HAS_MATPLOTLIB and plt is not None:
                self._plot_endpoint_volume_correlation_matplotlib(
                    endpoint_dists_array, volume, residue_sel_list, correlation_df, out_prefix, top_n
                )
            return

        # Combine all subplot images vertically
        total_height = sum(img.size[1] for img in subplot_images)
        max_width = max(img.size[0] for img in subplot_images)
        combined_img = Image.new('RGBA', (max_width, total_height), (255, 255, 255, 255))
        
        y_offset = 0
        for subplot_img in subplot_images:
            combined_img.paste(subplot_img, (0, y_offset))
            y_offset += subplot_img.size[1]

        # Save image
        plot_path = f"{out_prefix}_endpoint_volume_correlation.png"
        combined_img.save(plot_path)
        self.output_prefix = out_prefix
        self.plots_generated.append(plot_path)
        print(f"Endpoint-volume correlation plot saved to {plot_path}")

    def _plot_endpoint_volume_correlation_matplotlib(
        self,
        endpoint_dists_array: Union[np.ndarray, dict],
        volume: np.ndarray,
        residue_sel_list: List[str],
        correlation_df: pd.DataFrame,
        out_prefix: str,
        top_n: int = 5,
    ) -> None:
        """Matplotlib fallback for endpoint volume correlation plot."""
        if not HAS_MATPLOTLIB or plt is None:
            return

        top_pairs = correlation_df.head(top_n)
        n_plots = min(top_n, len(top_pairs))
        if n_plots == 0:
            return

        fig, axes = plt.subplots(n_plots, 1, figsize=(10, 3 * n_plots))
        if n_plots == 1:
            axes = [axes]

        frames = np.arange(len(volume))
        is_dict_format = isinstance(endpoint_dists_array, dict)
        if is_dict_format:
            all_pairs = endpoint_dists_array.get("all_pairs", {})

        for idx, (_, row) in enumerate(top_pairs.iterrows()):
            i, j = int(row["residue_i"]), int(row["residue_j"])
            corr = row["correlation"]
            p_val = row["p_value"]
            ep_i_idx = row.get("ep_i_idx", None)
            ep_j_idx = row.get("ep_j_idx", None)

            ax = axes[idx]
            ax2 = ax.twinx()

            if is_dict_format and ep_i_idx is not None and ep_j_idx is not None and (i, j) in all_pairs:
                pair_array = all_pairs[(i, j)]
                dists = pair_array[:, int(ep_i_idx), int(ep_j_idx)]
                ep_label = f"ep{int(ep_i_idx)}-ep{int(ep_j_idx)}"
            elif is_dict_format and (i, j) in all_pairs:
                pair_array = all_pairs[(i, j)]
                mean_dists = []
                for t in range(len(volume)):
                    frame_dists = pair_array[t]
                    valid_dists = frame_dists[~np.isnan(frame_dists)]
                    mean_dists.append(np.mean(valid_dists) if len(valid_dists) > 0 else np.nan)
                dists = np.array(mean_dists)
                ep_label = "mean"
            else:
                dists = endpoint_dists_array[:, i, j]
                ep_label = "min"

            ax.plot(frames, dists, "b-", label=f"Endpoint Distance ({ep_label})", linewidth=2, alpha=0.8)
            ax2.plot(frames, volume, "r-", label="Cube Volume", linewidth=2, alpha=0.8)

            ax.set_xlabel("Frame", fontsize=11)
            ax.set_ylabel("Endpoint Distance (Å)", fontsize=11, color="b")
            ax2.set_ylabel("Volume (Å³)", fontsize=11, color="r")
            ax.tick_params(axis="y", labelcolor="b")
            ax2.tick_params(axis="y", labelcolor="r")

            title = f"Pair {i}-{j}"
            if ep_i_idx is not None and ep_j_idx is not None:
                title += f" (ep{int(ep_i_idx)}-ep{int(ep_j_idx)})"
            title += f": r={corr:.3f}, p={p_val:.3e}"
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper left", fontsize=9)

        plt.tight_layout()
        plot_path = f"{out_prefix}_endpoint_volume_correlation.png"
        plt.savefig(plot_path, dpi=self.dpi, bbox_inches="tight")
        plt.close()
        self.output_prefix = out_prefix
        self.plots_generated.append(plot_path)
        print(f"Endpoint-volume correlation plot saved to {plot_path}")

    def plot_volume_change(
        self,
        volume: np.ndarray,
        out_prefix: str,
        frame_indices: Optional[np.ndarray] = None,
        times: Optional[np.ndarray] = None,
        xlabel: str = "Frame",
        ylabel: str = "Volume (Å³)",
        title: Optional[str] = None,
        show_stats: bool = True,
    ) -> None:
        """Plot volume change through frames using datashader."""
        if not HAS_DATASHADER or ds is None:
            warnings.warn("datashader not available. Skipping volume change plot.")
            return

        if volume is None or len(volume) == 0:
            warnings.warn("Volume array is None or empty. Skipping plot.")
            return

        if times is not None and len(times) == len(volume):
            x_values = np.array(times)
            xlabel = "Time (ps)"
        elif frame_indices is not None and len(frame_indices) == len(volume):
            x_values = np.array(frame_indices)
        else:
            x_values = np.arange(len(volume))

        # Calculate pixel dimensions
        pixel_width = int(self.figure_size[0] * self.dpi / 4)  # Reduced for reasonable file size
        pixel_height = int(self.figure_size[1] * self.dpi / 4)

        # Calculate statistics
        valid_volume = volume[~np.isnan(volume)]
        stats_text = ""
        h_lines = []
        legend_items = [("Volume", "blue")]
        
        if show_stats and len(valid_volume) > 0:
            mean_vol = np.mean(valid_volume)
            std_vol = np.std(valid_volume)
            min_vol = np.min(valid_volume)
            max_vol = np.max(valid_volume)

            stats_text = (
                f"Mean: {mean_vol:.2f} Å³\n"
                f"Std: {std_vol:.2f} Å³\n"
                f"Min: {min_vol:.2f} Å³\n"
                f"Max: {max_vol:.2f} Å³"
            )
            
            h_lines = [
                (mean_vol, 'red', 'dashed'),
                (mean_vol + std_vol, 'orange', 'dotted'),
                (mean_vol - std_vol, 'orange', 'dotted'),
            ]
            legend_items.extend([
                (f"Mean: {mean_vol:.2f}", "red"),
                (f"Mean ± Std", "orange"),
            ])

        # Determine y_range with padding
        y_min, y_max = float(np.nanmin(volume)), float(np.nanmax(volume))
        y_padding = (y_max - y_min) * 0.1 if y_max > y_min else 1.0
        y_range = (y_min - y_padding, y_max + y_padding)
        x_range = (float(x_values.min()), float(x_values.max()))

        # Create datashader line plot
        try:
            img = _datashader_line_plot(
                x_values, volume,
                pixel_width=pixel_width,
                pixel_height=pixel_height,
                line_color='blue',
                x_range=x_range,
                y_range=y_range,
            )
            
            if img is None:
                warnings.warn("Failed to create datashader plot.")
                return

            # Add text annotations
            if title is None:
                title = "Volume Change Through Frames"
            
            margin = {'top': 60, 'bottom': 50, 'left': 80, 'right': 150}
            img = _add_text_annotations(
                img,
                title=title,
                xlabel=xlabel,
                ylabel=ylabel,
                stats_text=stats_text,
                legend_items=legend_items,
                x_range=x_range,
                y_range=y_range,
                margin=margin,
            )
            
            # Add horizontal lines for statistics
            if h_lines:
                img = _add_horizontal_lines(img, h_lines, y_range, margin)
            
            # Save image
            plot_path = f"{out_prefix}_volume_change.png"
            img.save(plot_path)
            self.output_prefix = out_prefix
            self.plots_generated.append(plot_path)
            print(f"Volume change plot saved to {plot_path}")
            
        except Exception as e:
            warnings.warn(f"Failed to create datashader plot: {e}. Trying matplotlib fallback.")
            if HAS_MATPLOTLIB and plt is not None:
                self._plot_volume_change_matplotlib(
                    volume, out_prefix, frame_indices, times, xlabel, ylabel, title, show_stats
                )
    
    def _plot_volume_change_matplotlib(
        self,
        volume: np.ndarray,
        out_prefix: str,
        frame_indices: Optional[np.ndarray] = None,
        times: Optional[np.ndarray] = None,
        xlabel: str = "Frame",
        ylabel: str = "Volume (Å³)",
        title: Optional[str] = None,
        show_stats: bool = True,
    ) -> None:
        """Matplotlib fallback for volume change plot."""
        if not HAS_MATPLOTLIB or plt is None:
            return

        if times is not None and len(times) == len(volume):
            x_values = times
            xlabel = "Time (ps)"
        elif frame_indices is not None and len(frame_indices) == len(volume):
            x_values = frame_indices
        else:
            x_values = np.arange(len(volume))

        fig, ax = plt.subplots(figsize=self.figure_size)
        ax.plot(x_values, volume, "b-", linewidth=2, alpha=0.8, label="Volume")

        valid_volume = volume[~np.isnan(volume)]
        if show_stats and len(valid_volume) > 0:
            mean_vol = np.mean(valid_volume)
            std_vol = np.std(valid_volume)
            min_vol = np.min(valid_volume)
            max_vol = np.max(valid_volume)

            ax.axhline(mean_vol, color="r", linestyle="--", alpha=0.5, linewidth=1,
                      label=f"Mean: {mean_vol:.2f} Å³")
            ax.axhline(mean_vol + std_vol, color="orange", linestyle=":", alpha=0.5, linewidth=1,
                      label=f"Mean ± Std: {mean_vol:.2f} ± {std_vol:.2f} Å³")
            ax.axhline(mean_vol - std_vol, color="orange", linestyle=":", alpha=0.5, linewidth=1)

            stats_text = (f"Mean: {mean_vol:.2f} Å³\nStd: {std_vol:.2f} Å³\n"
                         f"Min: {min_vol:.2f} Å³\nMax: {max_vol:.2f} Å³")
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                   verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        if title is None:
            title = "Volume Change Through Frames"
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)

        if show_stats and len(valid_volume) > 0:
            ax.legend(loc="best", fontsize=9)

        plt.tight_layout()
        plot_path = f"{out_prefix}_volume_change.png"
        plt.savefig(plot_path, dpi=self.dpi, bbox_inches="tight")
        plt.close()
        self.output_prefix = out_prefix
        self.plots_generated.append(plot_path)
        print(f"Volume change plot saved to {plot_path}")

    def plot_residue_endpoints(
        self,
        u: mda.Universe,
        endpoint_observer: Optional[EndpointAnalyzerObserver] = None,
        residue_sel: Union[str, List[str]] = "resid 1",
        out_prefix: str = "residue_endpoints",
    ) -> None:
        """Plot the endpoints of a residue or multiple residues.
        
        Args:
            u: MDAnalysis Universe
            endpoint_observer: Optional EndpointAnalyzerObserver. If provided, uses stored_ep_indices for labeling
            residue_sel: Single residue selection string or list of residue selection strings
            out_prefix: Output file prefix
        """
        if not HAS_MATPLOTLIB or plt is None:
            warnings.warn("matplotlib not available. Skipping residue endpoint plots.")
            return

        ef = EndpointsFinder() if EndpointsFinder is not None else None  # type: ignore[call-arg]
        if ef is None:
            raise RuntimeError("EndpointsFinder is not available for plotting.")

        # Handle both string and list inputs
        if isinstance(residue_sel, str):
            residue_sel_list = [residue_sel]
        else:
            residue_sel_list = residue_sel

        n_residues = len(residue_sel_list)
        
        # Create subplots if multiple residues
        if n_residues == 1:
            fig, ax = plt.subplots(figsize=(10, 8))
            axes = [ax]
        else:
            n_cols = min(2, n_residues)
            n_rows = (n_residues + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(10 * n_cols, 8 * n_rows))
            # Flatten axes array if it's 2D
            if hasattr(axes, 'flatten'):
                axes = axes.flatten()
            elif not isinstance(axes, (list, np.ndarray)):
                axes = [axes]

        for idx, sel_str in enumerate(residue_sel_list):
            ax = axes[idx] if n_residues > 1 else axes[0]
            
            sel = u.select_atoms(sel_str)
            mol = sel.convert_to("RDKIT")
            if mol is None:
                warnings.warn(f"RDKit conversion failed for {sel_str}. Skipping.")
                ax.text(0.5, 0.5, f"Failed to convert\n{sel_str}", 
                       ha='center', va='center', transform=ax.transAxes)
                continue

            m2d, xy = ef.to_2d_coords(mol)
            
            # Get endpoint indices - use endpoint_observer if provided, otherwise find them
            if endpoint_observer is not None and hasattr(endpoint_observer, 'stored_ep_indices'):
                # Find the index of sel_str in endpoint_observer.residue_sel_list
                try:
                    residue_idx = endpoint_observer.residue_sel_list.index(sel_str)
                    mda_endpoint_indices = endpoint_observer.stored_ep_indices[residue_idx]
                except (ValueError, IndexError):
                    # If sel_str not found in observer's list, fall back to finding endpoints
                    mda_endpoint_indices = None
            else:
                mda_endpoint_indices = None
            
            if mda_endpoint_indices is not None and len(mda_endpoint_indices) > 0:
                # Use stored endpoint indices from observer
                # Map MDAnalysis universe indices to RDKit indices for highlighting
                universe_to_rdkit = {}
                for rdkit_idx in range(len(sel)):
                    universe_id = int(sel[rdkit_idx].id)
                    universe_to_rdkit[universe_id] = rdkit_idx
                
                # Get RDKit indices for highlighting, preserving order
                highlight_atoms = []
                valid_mda_indices = []
                for universe_idx in mda_endpoint_indices:
                    if universe_idx in universe_to_rdkit:
                        highlight_atoms.append(universe_to_rdkit[universe_idx])
                        valid_mda_indices.append(universe_idx)
                mda_endpoint_indices = valid_mda_indices  # Update to only valid indices
            else:
                # Fall back to finding endpoints using EndpointsFinder
                highlight_atoms = ef.find_endpoints(mol)  # RDKit indices for highlighting
                # Map RDKit indices to MDAnalysis universe indices
                mda_endpoint_indices = [int(sel[rdkit_idx].id) for rdkit_idx in highlight_atoms]
            
            # Create mapping for labeling (RDKit index -> MDAnalysis universe index)
            rdkit_to_universe = {rdkit_idx: int(sel[rdkit_idx].id) for rdkit_idx in highlight_atoms}
            
            # Try to use rdMolDraw2D for better coordinate control
            try:
                from rdkit.Chem import rdMolDraw2D
                import io
                try:
                    from PIL import Image
                except ImportError:
                    try:
                        from PIL import Image as PILImage
                        Image = PILImage
                    except ImportError:
                        Image = None
                
                if Image is None:
                    raise ImportError("PIL not available")
                
                # Create drawer
                drawer = rdMolDraw2D.MolDraw2DCairo(800, 600)
                drawer.SetDrawOptions(drawer.drawOptions())
                
                # Set up highlight colors
                highlight_colors = {atom_idx: (1.0, 0.0, 0.0) for atom_idx in highlight_atoms}
                
                # Draw molecule
                drawer.DrawMolecule(m2d, highlightAtoms=highlight_atoms, highlightAtomColors=highlight_colors)
                drawer.FinishDrawing()
                
                # Get the image
                img_data = drawer.GetDrawingText()
                img = Image.open(io.BytesIO(img_data))
                
            except (ImportError, AttributeError):
                # Fallback to MolToImage if rdMolDraw2D is not available
                img = Draw.MolToImage(
                    m2d, size=(800, 600), highlightAtoms=highlight_atoms, highlightColor=(1, 0, 0)
                )
            
            # Calculate coordinate transformation
            # Use the 2D coordinates from to_2d_coords
            x_coords = xy[:, 0]
            y_coords = xy[:, 1]
            x_min, x_max = x_coords.min(), x_coords.max()
            y_min, y_max = y_coords.min(), y_coords.max()
            
            x_range = x_max - x_min if x_max > x_min else 1.0
            y_range = y_max - y_min if y_max > y_min else 1.0
            padding_factor = 0.15  # RDKit typically uses ~15% padding
            
            # Calculate scale factors (drawer uses most of the image with some padding)
            scale_x = (800 * (1 - 2 * padding_factor)) / x_range if x_range > 0 else 1.0
            scale_y = (600 * (1 - 2 * padding_factor)) / y_range if y_range > 0 else 1.0
            scale = min(scale_x, scale_y)  # Use uniform scaling
            
            # Calculate offsets (center the molecule)
            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2
            offset_x = 400 - center_x * scale
            offset_y = 300 + center_y * scale  # Note: y is inverted in image coordinates
            
            # Display image
            ax.imshow(img, extent=[0, 800, 600, 0])  # Set extent so coordinates match
            ax.axis('off')
            
            # Label each endpoint with its universe index
            # highlight_atoms and mda_endpoint_indices are now aligned
            index=0
            for rdkit_idx, universe_idx in zip(highlight_atoms, mda_endpoint_indices):
                if rdkit_idx < len(xy):
                    x, y = xy[rdkit_idx]
                    
                    # Transform to image coordinates
                    x_pixel = x * scale + offset_x
                    y_pixel = -y * scale + offset_y  # Invert y for image coordinates
                    
                    # Add label with universe index
                    ax.annotate(
                        f"{str(index)+':'+str(universe_idx)}",
                        xy=(x_pixel, y_pixel),
                        xytext=(8, -8),
                        textcoords='offset points',
                        fontsize=14,
                        fontweight='bold',
                        color='blue',
                        bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.8, edgecolor='blue', linewidth=2),
                        arrowprops=dict(arrowstyle='->', color='blue', lw=2, connectionstyle='arc3,rad=0.1')
                    )
                    index+=1
            # Set title
            title = f"Endpoints: {sel_str}"
            if n_residues > 1:
                title += f" (Residue {idx + 1})"
            ax.set_title(title, fontsize=12, fontweight='bold')

        # Hide unused subplots
        for idx in range(n_residues, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        plot_path = f"{out_prefix}.png"
        plt.savefig(plot_path, dpi=self.dpi, bbox_inches="tight")
        plt.close()
        self.output_prefix = out_prefix
        self.plots_generated.append(plot_path)
        print(f"Residue endpoint plot saved to {plot_path}")

    def plot_guest_entering_events(
        self,
        guest_residence_stats: dict,
        out_prefix: str,
        show_exits: bool = True,
        show_durations: bool = True,
        max_frames: Optional[int] = None,
    ) -> None:
        """
        Plot guest entering events as a function of frame using datashader.
        
        Args:
            guest_residence_stats: Dictionary from GSAnalyzerObserver.get_guest_residence_stats()
            out_prefix: Output file prefix
            show_exits: If True, also plot exit events
            show_durations: If True, show shaded regions for durations inside host
            max_frames: Optional maximum frame number for x-axis limit. If None, uses max frame from events
        """
        if not HAS_DATASHADER or ds is None or not HAS_PIL:
            warnings.warn("datashader or PIL not available. Skipping guest entering events plot.")
            return

        if not guest_residence_stats:
            warnings.warn("Guest residence stats is empty. Skipping plot.")
            return

        entry_frames = guest_residence_stats.get('entry_frames', [])
        exit_frames = guest_residence_stats.get('exit_frames', [])
        durations_inside = guest_residence_stats.get('durations_inside', [])

        if not entry_frames and not exit_frames:
            warnings.warn("No entry or exit events found. Skipping plot.")
            return

        # Determine frame range
        all_frames = entry_frames + exit_frames
        if all_frames:
            min_frame = min(all_frames)
            max_frame_val = max(all_frames)
            if max_frames is not None:
                max_frame_val = max(max_frame_val, max_frames)
            frame_range = max_frame_val - min_frame
            x_min = max(0, min_frame - frame_range * 0.05)
            x_max = max_frame_val + frame_range * 0.05
        else:
            x_min, x_max = 0, 100

        # Fixed y range for this plot type
        y_min, y_max = -0.2, 1.5
        
        # Calculate pixel dimensions
        pixel_width = int(self.figure_size[0] * self.dpi / 4)
        pixel_height = int(self.figure_size[1] * self.dpi / 4)

        try:
            # Build scatter data
            scatter_markers = []
            legend_items = []
            
            if entry_frames:
                entry_y = [1.0] * len(entry_frames)
                scatter_markers.append((entry_frames, entry_y, 'green', f'Entry (n={len(entry_frames)})', 'triangle_up'))
                legend_items.append((f'Entry events (n={len(entry_frames)})', 'green'))
            
            if show_exits and exit_frames:
                exit_y = [0.5] * len(exit_frames)
                scatter_markers.append((exit_frames, exit_y, 'red', f'Exit (n={len(exit_frames)})', 'triangle_down'))
                legend_items.append((f'Exit events (n={len(exit_frames)})', 'red'))

            # Add margins for annotations
            margin = {'top': 60, 'bottom': 50, 'left': 80, 'right': 180}
            new_width = pixel_width + margin['left'] + margin['right']
            new_height = pixel_height + margin['top'] + margin['bottom']

            stay_intervals = _merge_frame_intervals(
                _pair_guest_entry_exit_intervals(guest_residence_stats)
            )

            # Build statistics text
            stats_lines = []
            if guest_residence_stats.get('n_entries', 0) > 0:
                stats_lines.append(f"Total entries: {guest_residence_stats['n_entries']}")
                stats_lines.append(f"Total exits: {guest_residence_stats['n_exits']}")
                if guest_residence_stats.get('first_entry_frame') is not None:
                    stats_lines.append(f"First entry: Frame {guest_residence_stats['first_entry_frame']}")
                if durations_inside:
                    stats_lines.append(f"Avg stay: {np.mean(durations_inside):.1f} ps")
                    stats_lines.append(f"Max stay: {np.max(durations_inside):.1f} ps")
                if guest_residence_stats.get('total_time_inside', 0) > 0:
                    stats_lines.append(f"Total inside: {guest_residence_stats['total_time_inside']:.1f} ps")
            stats_text = '\n'.join(stats_lines) if stats_lines else ""

            final_img = Image.new('RGBA', (new_width, new_height), (255, 255, 255, 255))
            draw = ImageDraw.Draw(final_img)

            # Shaded "inside host" intervals: per-guest–matched, merged (no stacked alpha slab)
            if show_durations and stay_intervals:
                for entry_frame, exit_frame in stay_intervals:
                    x1_frac = (entry_frame - x_min) / (x_max - x_min) if x_max > x_min else 0
                    x2_frac = (exit_frame - x_min) / (x_max - x_min) if x_max > x_min else 0
                    x1_pixel = margin['left'] + int(pixel_width * x1_frac)
                    x2_pixel = margin['left'] + int(pixel_width * x2_frac)
                    draw.rectangle(
                        [x1_pixel, margin['top'], x2_pixel, margin['top'] + pixel_height],
                        fill=(220, 248, 220, 255),
                    )
            
            # Draw scatter markers
            final_img = _draw_scatter_markers_pil(
                final_img, scatter_markers, (x_min, x_max), (y_min, y_max),
                margin=margin, marker_size=10
            )
            
            # Add annotations
            final_img = _add_text_annotations(
                final_img,
                title="Guest Entering/Exiting Events",
                xlabel="Frame",
                ylabel="Event Type",
                stats_text=stats_text,
                legend_items=legend_items,
                x_range=(x_min, x_max),
                y_range=(y_min, y_max),
                margin=margin,
            )

            # Save image
            plot_path = f"{out_prefix}_guest_entering_events.png"
            final_img.save(plot_path)
            self.output_prefix = out_prefix
            self.plots_generated.append(plot_path)
            print(f"Guest entering events plot saved to {plot_path}")
            
        except Exception as e:
            warnings.warn(f"Failed to create datashader plot: {e}. Trying matplotlib fallback.")
            if HAS_MATPLOTLIB and plt is not None:
                self._plot_guest_entering_events_matplotlib(
                    guest_residence_stats, out_prefix, show_exits, show_durations, max_frames
                )
    
    def _plot_guest_entering_events_matplotlib(
        self,
        guest_residence_stats: dict,
        out_prefix: str,
        show_exits: bool = True,
        show_durations: bool = True,
        max_frames: Optional[int] = None,
    ) -> None:
        """Matplotlib fallback for guest entering events plot."""
        if not HAS_MATPLOTLIB or plt is None:
            return

        entry_frames = guest_residence_stats.get('entry_frames', [])
        exit_frames = guest_residence_stats.get('exit_frames', [])

        if not entry_frames and not exit_frames:
            return

        all_frames = entry_frames + exit_frames
        if all_frames:
            min_frame = min(all_frames)
            max_frame_val = max(all_frames)
            if max_frames is not None:
                max_frame_val = max(max_frame_val, max_frames)
            frame_range = max_frame_val - min_frame
            x_min = max(0, min_frame - frame_range * 0.05)
            x_max = max_frame_val + frame_range * 0.05
        else:
            x_min, x_max = 0, 100

        fig, ax = plt.subplots(figsize=self.figure_size)

        stay_intervals = _merge_frame_intervals(
            _pair_guest_entry_exit_intervals(guest_residence_stats)
        )
        if show_durations and stay_intervals:
            for i, (entry_frame, exit_frame) in enumerate(stay_intervals):
                ax.axvspan(
                    entry_frame,
                    exit_frame,
                    alpha=0.18,
                    color="green",
                    label="Inside host" if i == 0 else "",
                )

        if entry_frames:
            ax.scatter(entry_frames, [1] * len(entry_frames), color='green', marker='^',
                      s=100, zorder=3, label=f'Entry events (n={len(entry_frames)})',
                      edgecolors='darkgreen', linewidths=1.5)

        if show_exits and exit_frames:
            ax.scatter(exit_frames, [0.5] * len(exit_frames), color='red', marker='v',
                      s=100, zorder=3, label=f'Exit events (n={len(exit_frames)})',
                      edgecolors='darkred', linewidths=1.5)

        ax.set_xlabel("Frame", fontsize=12)
        ax.set_ylabel("Event Type", fontsize=12)
        ax.set_title("Guest Entering/Exiting Events", fontsize=14, fontweight="bold")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(-0.2, 1.5)
        ax.set_yticks([0.5, 1.0])
        ax.set_yticklabels(['Exit', 'Entry'])
        ax.grid(True, alpha=0.3, axis='x')
        ax.legend(loc='upper left', fontsize=9)

        plt.tight_layout()
        plot_path = f"{out_prefix}_guest_entering_events.png"
        plt.savefig(plot_path, dpi=self.dpi, bbox_inches="tight")
        plt.close()
        self.output_prefix = out_prefix
        self.plots_generated.append(plot_path)
        print(f"Guest entering events plot saved to {plot_path}")

    def plot_guest_entry_exit_timeline(
        self,
        guest_residence_stats: dict,
        out_prefix: str,
        show_connections: bool = False,
        max_frames: Optional[int] = None,
    ) -> None:
        """
        Plot guest entry and exit frames with guest index on Y-axis and frame on X-axis using datashader.
        
        Args:
            guest_residence_stats: Dictionary from GSAnalyzerObserver.get_guest_residence_stats()
            out_prefix: Output file prefix
            show_connections: If True, draw horizontal lines connecting entry-exit pairs for each guest
            max_frames: Optional maximum frame number for x-axis limit. If None, uses max frame from events
        """
        if not HAS_DATASHADER or ds is None or not HAS_PIL:
            warnings.warn("datashader or PIL not available. Skipping guest entry/exit timeline plot.")
            return

        if not guest_residence_stats:
            warnings.warn("Guest residence stats is empty. Skipping plot.")
            return

        entry_frames = guest_residence_stats.get('entry_frames', [])
        exit_frames = guest_residence_stats.get('exit_frames', [])
        entry_guest_indices = guest_residence_stats.get('entry_guest_indices', [])
        exit_guest_indices = guest_residence_stats.get('exit_guest_indices', [])

        if not entry_frames and not exit_frames:
            warnings.warn("No entry or exit events found. Skipping plot.")
            return

        # Collect all unique guest indices
        all_guest_indices = set()
        for indices in entry_guest_indices:
            all_guest_indices.update(indices)
        for indices in exit_guest_indices:
            all_guest_indices.update(indices)
        
        if not all_guest_indices:
            warnings.warn("No guest indices found in events. Skipping plot.")
            return

        # Sort guest indices for consistent Y-axis ordering
        sorted_guest_indices = sorted(all_guest_indices)
        guest_index_to_y = {gidx: i for i, gidx in enumerate(sorted_guest_indices)}

        # Determine frame range
        all_frames_list = entry_frames + exit_frames
        if all_frames_list:
            min_frame = min(all_frames_list)
            max_frame_val = max(all_frames_list)
            if max_frames is not None:
                max_frame_val = max(max_frame_val, max_frames)
            frame_range = max_frame_val - min_frame
            x_min = max(0, min_frame - frame_range * 0.05)
            x_max = max_frame_val + frame_range * 0.05
        else:
            x_min, x_max = 0, 100

        # Y range based on number of guests
        y_min, y_max = -0.5, len(sorted_guest_indices) - 0.5

        # Calculate pixel dimensions
        pixel_width = int(self.figure_size[0] * self.dpi / 4)
        pixel_height = int(self.figure_size[1] * self.dpi / 4)

        try:
            margin = {'top': 60, 'bottom': 50, 'left': 80, 'right': 180}
            new_width = pixel_width + margin['left'] + margin['right']
            new_height = pixel_height + margin['top'] + margin['bottom']
            
            # Create final image
            final_img = Image.new('RGBA', (new_width, new_height), (255, 255, 255, 255))
            draw = ImageDraw.Draw(final_img)

            # Draw connections if requested
            if show_connections:
                guest_timelines = {gidx: [] for gidx in sorted_guest_indices}
                
                for entry_frame, guest_indices in zip(entry_frames, entry_guest_indices):
                    for gidx in guest_indices:
                        if gidx in guest_timelines:
                            guest_timelines[gidx].append({'entry': entry_frame, 'exit': None})
                
                for exit_frame, guest_indices in zip(exit_frames, exit_guest_indices):
                    for gidx in guest_indices:
                        if gidx in guest_timelines:
                            for timeline_entry in reversed(guest_timelines[gidx]):
                                if timeline_entry['exit'] is None:
                                    timeline_entry['exit'] = exit_frame
                                    break
                
                # Draw horizontal lines for connections
                for gidx, timeline in guest_timelines.items():
                    y_pos = guest_index_to_y[gidx]
                    for period in timeline:
                        entry_frame = period['entry']
                        exit_frame = period['exit'] if period['exit'] is not None else x_max
                        
                        # Calculate pixel positions
                        x1_frac = (entry_frame - x_min) / (x_max - x_min) if x_max > x_min else 0
                        x2_frac = (exit_frame - x_min) / (x_max - x_min) if x_max > x_min else 0
                        y_frac = (y_pos - y_min) / (y_max - y_min) if y_max > y_min else 0.5
                        
                        x1_pixel = margin['left'] + int(pixel_width * x1_frac)
                        x2_pixel = margin['left'] + int(pixel_width * x2_frac)
                        y_pixel = margin['top'] + int(pixel_height * (1 - y_frac))
                        
                        draw.line([(x1_pixel, y_pixel), (x2_pixel, y_pixel)],
                                 fill=(135, 206, 235, 180), width=4)

            # Build scatter data
            entry_x = []
            entry_y = []
            for entry_frame, guest_indices in zip(entry_frames, entry_guest_indices):
                for gidx in guest_indices:
                    if gidx in guest_index_to_y:
                        entry_x.append(entry_frame)
                        entry_y.append(guest_index_to_y[gidx])

            exit_x = []
            exit_y = []
            for exit_frame, guest_indices in zip(exit_frames, exit_guest_indices):
                for gidx in guest_indices:
                    if gidx in guest_index_to_y:
                        exit_x.append(exit_frame)
                        exit_y.append(guest_index_to_y[gidx])

            # Build scatter markers
            scatter_markers = []
            legend_items = []
            
            if entry_x:
                scatter_markers.append((entry_x, entry_y, 'green', f'Entry (n={len(entry_x)})', 'triangle_up'))
                legend_items.append((f'Entry events (n={len(entry_x)})', 'green'))

            if exit_x:
                scatter_markers.append((exit_x, exit_y, 'red', f'Exit (n={len(exit_x)})', 'triangle_down'))
                legend_items.append((f'Exit events (n={len(exit_x)})', 'red'))

            # Draw scatter markers
            final_img = _draw_scatter_markers_pil(
                final_img, scatter_markers, (x_min, x_max), (y_min, y_max),
                margin=margin, marker_size=10
            )

            # Build statistics text
            stats_lines = []
            if guest_residence_stats.get('n_entries', 0) > 0:
                stats_lines.append(f"Total entries: {guest_residence_stats['n_entries']}")
                stats_lines.append(f"Total exits: {guest_residence_stats['n_exits']}")
                stats_lines.append(f"Unique guests: {len(sorted_guest_indices)}")
                if guest_residence_stats.get('first_entry_frame') is not None:
                    stats_lines.append(f"First entry: Frame {guest_residence_stats['first_entry_frame']}")
            stats_text = '\n'.join(stats_lines) if stats_lines else ""

            # Add text annotations
            final_img = _add_text_annotations(
                final_img,
                title="Guest Entry/Exit Timeline",
                xlabel="Frame",
                ylabel="Guest Index",
                stats_text=stats_text,
                legend_items=legend_items,
                x_range=(x_min, x_max),
                y_range=(y_min, y_max),
                margin=margin,
            )

            # Save image
            plot_path = f"{out_prefix}_guest_entry_exit_timeline.png"
            final_img.save(plot_path)
            self.output_prefix = out_prefix
            self.plots_generated.append(plot_path)
            print(f"Guest entry/exit timeline plot saved to {plot_path}")
            
        except Exception as e:
            warnings.warn(f"Failed to create datashader plot: {e}. Trying matplotlib fallback.")
            if HAS_MATPLOTLIB and plt is not None:
                self._plot_guest_entry_exit_timeline_matplotlib(
                    guest_residence_stats, out_prefix, show_connections, max_frames
                )
    
    def _plot_guest_entry_exit_timeline_matplotlib(
        self,
        guest_residence_stats: dict,
        out_prefix: str,
        show_connections: bool = False,
        max_frames: Optional[int] = None,
    ) -> None:
        """Matplotlib fallback for guest entry/exit timeline plot."""
        if not HAS_MATPLOTLIB or plt is None:
            return

        entry_frames = guest_residence_stats.get('entry_frames', [])
        exit_frames = guest_residence_stats.get('exit_frames', [])
        entry_guest_indices = guest_residence_stats.get('entry_guest_indices', [])
        exit_guest_indices = guest_residence_stats.get('exit_guest_indices', [])

        if not entry_frames and not exit_frames:
            return

        all_guest_indices = set()
        for indices in entry_guest_indices:
            all_guest_indices.update(indices)
        for indices in exit_guest_indices:
            all_guest_indices.update(indices)
        
        if not all_guest_indices:
            return

        sorted_guest_indices = sorted(all_guest_indices)
        guest_index_to_y = {gidx: i for i, gidx in enumerate(sorted_guest_indices)}

        all_frames_list = entry_frames + exit_frames
        if all_frames_list:
            min_frame = min(all_frames_list)
            max_frame_val = max(all_frames_list)
            if max_frames is not None:
                max_frame_val = max(max_frame_val, max_frames)
            frame_range = max_frame_val - min_frame
            x_min = max(0, min_frame - frame_range * 0.05)
            x_max = max_frame_val + frame_range * 0.05
        else:
            x_min, x_max = 0, 100

        fig, ax = plt.subplots(figsize=self.figure_size)

        if show_connections:
            guest_timelines = {gidx: [] for gidx in sorted_guest_indices}
            for entry_frame, guest_indices in zip(entry_frames, entry_guest_indices):
                for gidx in guest_indices:
                    if gidx in guest_timelines:
                        guest_timelines[gidx].append({'entry': entry_frame, 'exit': None})
            for exit_frame, guest_indices in zip(exit_frames, exit_guest_indices):
                for gidx in guest_indices:
                    if gidx in guest_timelines:
                        for timeline_entry in reversed(guest_timelines[gidx]):
                            if timeline_entry['exit'] is None:
                                timeline_entry['exit'] = exit_frame
                                break
            for gidx, timeline in guest_timelines.items():
                y_pos = guest_index_to_y[gidx]
                for period in timeline:
                    entry_frame = period['entry']
                    exit_frame = period['exit'] if period['exit'] is not None else x_max
                    ax.plot([entry_frame, exit_frame], [y_pos, y_pos],
                           color='lightblue', linewidth=3, alpha=0.5, zorder=1)

        entry_x, entry_y = [], []
        for entry_frame, guest_indices in zip(entry_frames, entry_guest_indices):
            for gidx in guest_indices:
                if gidx in guest_index_to_y:
                    entry_x.append(entry_frame)
                    entry_y.append(guest_index_to_y[gidx])

        if entry_x:
            ax.scatter(entry_x, entry_y, color='green', marker='^', s=150, zorder=3,
                      label=f'Entry events (n={len(entry_x)})', edgecolors='darkgreen', linewidths=1.5)

        exit_x, exit_y = [], []
        for exit_frame, guest_indices in zip(exit_frames, exit_guest_indices):
            for gidx in guest_indices:
                if gidx in guest_index_to_y:
                    exit_x.append(exit_frame)
                    exit_y.append(guest_index_to_y[gidx])

        if exit_x:
            ax.scatter(exit_x, exit_y, color='red', marker='v', s=150, zorder=3,
                      label=f'Exit events (n={len(exit_x)})', edgecolors='darkred', linewidths=1.5)

        ax.set_xlabel("Frame", fontsize=12)
        ax.set_ylabel("Guest Index", fontsize=12)
        ax.set_title("Guest Entry/Exit Timeline", fontsize=14, fontweight="bold")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(-0.5, len(sorted_guest_indices) - 0.5)
        ax.set_yticks(range(len(sorted_guest_indices)))
        ax.set_yticklabels([str(gidx) for gidx in sorted_guest_indices])
        ax.grid(True, alpha=0.3, axis='x')
        ax.legend(loc='upper left', fontsize=9)

        plt.tight_layout()
        plot_path = f"{out_prefix}_guest_entry_exit_timeline.png"
        plt.savefig(plot_path, dpi=self.dpi, bbox_inches="tight")
        plt.close()
        self.output_prefix = out_prefix
        self.plots_generated.append(plot_path)
        print(f"Guest entry/exit timeline plot saved to {plot_path}")

    def plot_guest_entry_exit_timeline_gif(
        self,
        csv_path: str,
        out_prefix: str,
        show_connections: bool = True,
        max_frames: Optional[int] = None,
        fps: float = 5.0,
        frame_step: int = 5,
        n_jobs: Optional[int] = None,
    ) -> None:
        """
        Create an animated GIF version of guest entry/exit timeline plot from CSV files.
        
        Args:
            csv_path: Path to guest_entering_events.csv file (will also look for guest_entering_stats.csv for metadata)
            out_prefix: Output file prefix
            show_connections: If True, draw horizontal lines connecting entry-exit pairs for each guest
            max_frames: Optional maximum frame number for x-axis limit. If None, uses max frame from events
            fps: Frames per second for the GIF animation
            frame_step: Step size for animation frames (1 = every frame, 10 = every 10th frame, etc.)
            n_jobs: Number of parallel workers for frame generation. If None, uses all available CPU cores.
                   If 1, runs sequentially. Use > 1 for parallel processing.
        """
        if not HAS_MATPLOTLIB or plt is None:
            warnings.warn("matplotlib not available. Skipping guest entry/exit timeline GIF.")
            return
        
        if not HAS_IMAGEIO or imageio is None:
            warnings.warn("imageio not available. Skipping GIF generation.")
            return

        # Determine if csv_path is events or stats file
        csv_dir = os.path.dirname(csv_path)
        csv_basename = os.path.basename(csv_path)
        
        if "_guest_entering_events.csv" in csv_basename:
            events_csv_path = csv_path
            # Try to find stats file
            stats_csv_path = os.path.join(
                csv_dir,
                csv_basename.replace("_guest_entering_events.csv", "_guest_entering_stats.csv")
            )
        elif "_guest_entering_stats.csv" in csv_basename:
            # Legacy: if stats file is provided, look for events file
            stats_csv_path = csv_path
            events_csv_path = os.path.join(
                csv_dir,
                csv_basename.replace("_guest_entering_stats.csv", "_guest_entering_events.csv")
            )
        else:
            # Assume it's the events file
            events_csv_path = csv_path
            stats_csv_path = None

        if not os.path.exists(events_csv_path):
            warnings.warn(f"Events CSV file not found at {events_csv_path}. Cannot create GIF.")
            return

        guest_residence_stats = guest_residence_stats_from_entering_events_csv(
            events_csv_path, stats_csv_path
        )
        entry_frames = guest_residence_stats.get("entry_frames", [])
        exit_frames = guest_residence_stats.get("exit_frames", [])
        entry_guest_indices = guest_residence_stats.get("entry_guest_indices", [])
        exit_guest_indices = guest_residence_stats.get("exit_guest_indices", [])

        if not entry_frames and not exit_frames:
            warnings.warn("No entry or exit events found. Skipping GIF.")
            return

        # Collect all unique guest indices
        all_guest_indices = set()
        for indices in entry_guest_indices:
            all_guest_indices.update(indices)
        for indices in exit_guest_indices:
            all_guest_indices.update(indices)
        
        if not all_guest_indices:
            warnings.warn("No guest indices found in events. Skipping GIF.")
            return

        # Sort guest indices for consistent Y-axis ordering
        sorted_guest_indices = sorted(all_guest_indices)
        guest_index_to_y = {gidx: i for i, gidx in enumerate(sorted_guest_indices)}

        # Determine frame range
        all_frames = entry_frames + exit_frames
        if all_frames:
            min_frame = min(all_frames)
            max_frame = max(all_frames)
            if max_frames is not None:
                max_frame = max(max_frame, max_frames)
            # Add some padding
            frame_range = max_frame - min_frame
            x_min = max(0, min_frame - frame_range * 0.05)
            x_max = max_frame + frame_range * 0.05
        else:
            x_min, x_max = 0, 100

        # Create frames for animation
        animation_frames = list(range(int(x_min), int(x_max) + 1, frame_step))
        if not animation_frames:
            animation_frames = [int(x_min), int(x_max)]

        print(f"Creating animated GIF with {len(animation_frames)} frames...")
        
        # Determine number of workers
        if n_jobs is None:
            if PARALLEL_AVAILABLE and cpu_count is not None:
                n_jobs = cpu_count()
            else:
                n_jobs = 1
        elif n_jobs == -1:
            if PARALLEL_AVAILABLE and cpu_count is not None:
                n_jobs = cpu_count()
            else:
                n_jobs = 1
        
        # Use parallel processing if n_jobs > 1 and parallel is available
        if n_jobs > 1 and PARALLEL_AVAILABLE and ProcessPoolExecutor is not None:
            print(f"Using {n_jobs} parallel workers for frame generation...")
            images_dict = {}  # Store images by frame index to maintain order
            
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                # Submit all frame generation tasks
                future_to_frame = {
                    executor.submit(
                        _generate_timeline_frame,
                        current_frame,
                        entry_frames,
                        exit_frames,
                        entry_guest_indices,
                        exit_guest_indices,
                        sorted_guest_indices,
                        guest_index_to_y,
                        guest_residence_stats,
                        x_min,
                        x_max,
                        self.figure_size,
                        self.dpi,
                        show_connections,
                    ): current_frame
                    for current_frame in animation_frames
                }
                
                # Collect results as they complete
                completed = 0
                for future in as_completed(future_to_frame):
                    current_frame = future_to_frame[future]
                    try:
                        img = future.result()
                        images_dict[current_frame] = img
                        completed += 1
                        if completed % 10 == 0 or completed == len(animation_frames):
                            print(f"  Progress: {completed}/{len(animation_frames)} frames completed")
                    except Exception as e:
                        warnings.warn(f"Failed to generate frame {current_frame}: {e}")
            
            # Sort images by frame index to maintain order
            images = [images_dict[frame] for frame in animation_frames if frame in images_dict]
        else:
            # Sequential processing
            if n_jobs > 1:
                warnings.warn("Parallel processing requested but not available. Using sequential processing.")
            images = []
            for i, current_frame in enumerate(animation_frames):
                img = _generate_timeline_frame(
                    current_frame,
                    entry_frames,
                    exit_frames,
                    entry_guest_indices,
                    exit_guest_indices,
                    sorted_guest_indices,
                    guest_index_to_y,
                    guest_residence_stats,
                    x_min,
                    x_max,
                    self.figure_size,
                    self.dpi,
                    show_connections,
                )
                images.append(img)
                if (i + 1) % 10 == 0 or (i + 1) == len(animation_frames):
                    print(f"  Progress: {i + 1}/{len(animation_frames)} frames completed")

        if not images:
            warnings.warn("No frames collected for GIF.")
            return

        # Save GIF
        gif_path = f"{out_prefix}_guest_entry_exit_timeline.gif"
        imageio.mimsave(gif_path, images, duration=1.0 / fps)
        self.output_prefix = out_prefix
        self.plots_generated.append(gif_path)
        print(f"Guest entry/exit timeline GIF saved to {gif_path}")

    # ========== Volume plotting methods (moved from VolumeAnalyzer) ==========
    
    @staticmethod
    def plot_cavity_slice(
        inside: np.ndarray,
        cavities: np.ndarray,
        axis: str = "z",
        index: int | None = None,
        outfile: str = "cavity_slice.png",
    ):
        """
        Plot a 2D slice through the 3D grid for sanity checking using datashader/PIL.

        Parameters
        ----------
        inside : 3D bool array
            True = target.
        cavities : 3D bool array
            True = cavity voxels.
        axis : {'x', 'y', 'z'}
            Axis normal to the slice.
        index : int or None
            Slice index along chosen axis. If None, use middle slice.
        outfile : str
            Output PNG filename.
        """
        if not HAS_PIL or Image is None:
            # Fall back to matplotlib
            if not HAS_MATPLOTLIB or plt is None or ListedColormap is None:
                warnings.warn("Neither PIL nor matplotlib available. Cannot plot cavity slice.")
                return
            Plotter._plot_cavity_slice_matplotlib(inside, cavities, axis, index, outfile)
            return

        nx, ny, nz = inside.shape
        axis = axis.lower()

        if axis == "z":
            if index is None or index < 0 or index >= nz:
                index = nz // 2
            prot2d = inside[:, :, index]
            cav2d = cavities[:, :, index]
        elif axis == "y":
            if index is None or index < 0 or index >= ny:
                index = ny // 2
            prot2d = inside[:, index, :]
            cav2d = cavities[:, index, :]
        elif axis == "x":
            if index is None or index < 0 or index >= nx:
                index = nx // 2
            prot2d = inside[index, :, :]
            cav2d = cavities[index, :, :]
        else:
            raise ValueError("axis must be 'x', 'y', or 'z'")

        arr = np.zeros_like(prot2d, dtype=int)
        arr[prot2d] = 1   # target
        arr[cav2d] = 2    # cavity

        # Create image using PIL
        height, width = arr.T.shape
        img_array = arr.T
        
        # Create RGB image
        rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
        rgb_image[img_array == 0] = [255, 255, 255]  # white - outside
        rgb_image[img_array == 1] = [0, 0, 0]        # black - target
        rgb_image[img_array == 2] = [255, 0, 0]      # red - cavity
        
        # Scale up for visibility
        scale_factor = max(1, 300 // max(height, width))
        new_height = height * scale_factor
        new_width = width * scale_factor
        
        pil_img = Image.fromarray(rgb_image, mode='RGB')
        pil_img = pil_img.resize((new_width, new_height), Image.NEAREST)
        
        # Add margins for annotations
        margin = {'top': 40, 'bottom': 50, 'left': 60, 'right': 100}
        full_width = new_width + margin['left'] + margin['right']
        full_height = new_height + margin['top'] + margin['bottom']
        
        final_img = Image.new('RGB', (full_width, full_height), (255, 255, 255))
        final_img.paste(pil_img, (margin['left'], margin['top']))
        
        draw = ImageDraw.Draw(final_img)
        
        # Get fonts
        try:
            font_title = ImageFont.truetype("arial.ttf", 14)
            font_label = ImageFont.truetype("arial.ttf", 12)
            font_tick = ImageFont.truetype("arial.ttf", 10)
        except (OSError, IOError):
            try:
                font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
                font_label = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
                font_tick = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
            except (OSError, IOError):
                font_title = ImageFont.load_default()
                font_label = font_title
                font_tick = font_title
        
        # Add title
        title = f"Cavity slice (axis={axis}, index={index})"
        title_bbox = draw.textbbox((0, 0), title, font=font_title)
        title_x = margin['left'] + (new_width - (title_bbox[2] - title_bbox[0])) // 2
        draw.text((title_x, 10), title, fill='black', font=font_title)
        
        # Add axis labels
        draw.text((margin['left'] + new_width // 2 - 30, full_height - 25), "grid index", fill='black', font=font_label)
        
        # Y-axis label
        ylabel_img = Image.new('RGB', (100, 15), (255, 255, 255))
        ylabel_draw = ImageDraw.Draw(ylabel_img)
        ylabel_draw.text((0, 0), "grid index", fill='black', font=font_label)
        ylabel_img = ylabel_img.rotate(90, expand=True)
        final_img.paste(ylabel_img, (5, margin['top'] + (new_height - ylabel_img.size[1]) // 2))
        
        # Add legend
        legend_x = margin['left'] + new_width + 10
        legend_y = margin['top'] + 20
        legend_items = [("Outside", (255, 255, 255)), ("Target", (0, 0, 0)), ("Cavity", (255, 0, 0))]
        for idx, (label, color) in enumerate(legend_items):
            draw.rectangle([legend_x, legend_y + idx * 25, legend_x + 20, legend_y + idx * 25 + 15],
                          fill=color, outline='black')
            draw.text((legend_x + 25, legend_y + idx * 25), label, fill='black', font=font_tick)
        
        final_img.save(outfile)

    @staticmethod
    def _plot_cavity_slice_matplotlib(
        inside: np.ndarray,
        cavities: np.ndarray,
        axis: str = "z",
        index: int | None = None,
        outfile: str = "cavity_slice.png",
    ):
        """Matplotlib fallback for cavity slice plot."""
        if not HAS_MATPLOTLIB or plt is None or ListedColormap is None:
            return

        nx, ny, nz = inside.shape
        axis = axis.lower()

        if axis == "z":
            if index is None or index < 0 or index >= nz:
                index = nz // 2
            prot2d = inside[:, :, index]
            cav2d = cavities[:, :, index]
        elif axis == "y":
            if index is None or index < 0 or index >= ny:
                index = ny // 2
            prot2d = inside[:, index, :]
            cav2d = cavities[:, index, :]
        elif axis == "x":
            if index is None or index < 0 or index >= nx:
                index = nx // 2
            prot2d = inside[index, :, :]
            cav2d = cavities[index, :, :]
        else:
            return

        arr = np.zeros_like(prot2d, dtype=int)
        arr[prot2d] = 1
        arr[cav2d] = 2

        cmap = ListedColormap(["white", "black", "red"])

        plt.figure(figsize=(5, 5))
        im = plt.imshow(arr.T, origin="lower", cmap=cmap, interpolation="nearest")
        plt.title(f"Cavity slice (axis={axis}, index={index})")
        plt.xlabel("grid index")
        plt.ylabel("grid index")
        cbar = plt.colorbar(im, ticks=[0, 1, 2])
        cbar.ax.set_yticklabels(["Outside", "target", "Cavity"])
        plt.tight_layout()
        plt.savefig(outfile, dpi=300)
        plt.close()

    def plot_interactive_3d(
        self,
        volume_analyzer,  # VolumeAnalyzer instance
        frame_index: int = 0,
        voxel_stride: int = 1,
    ):
        """
        Interactive 3D visualization of:
        - molecule/target (atoms + bonds, with annotation menu)
        - target volume (inside mask) as a Volume isosurface
        - cavity volume (cavities mask) as a Volume isosurface
        
        Parameters
        ----------
        volume_analyzer : VolumeAnalyzer
            VolumeAnalyzer instance to use for computation.
        frame_index : int
            Frame index in universe.trajectory.
        voxel_stride : int
            Subsampling step for the grid in each dimension (>=1).
            1 = full grid; 2 = take every 2nd voxel, etc.
            
        Returns
        -------
        fig : go.Figure
        """
        if not HAS_PLOTLY or go is None or make_molecule_components is None:
            raise RuntimeError("plotly and plotly_molecule are required for interactive 3D visualization.")

        # Get universe for this frame
        u = volume_analyzer._get_universe(frame_index)
        
        # --- 1. Compute volumes & masks on the same grid used in compute_frame ---
        target_vol, cavity_vol, inside, cavities = volume_analyzer.compute_frame(
            frame_index, return_masks=True, universe=u
        )
        
        if volume_analyzer._last_grid_axes is None:
            raise RuntimeError(
                "Grid axes not stored; make sure compute_frame sets _last_grid_axes."
            )
        x_axis, y_axis, z_axis = volume_analyzer._last_grid_axes
        
        # Optional grid subsampling for rendering speed
        if voxel_stride > 1:
            inside_sub = inside[::voxel_stride, ::voxel_stride, ::voxel_stride]
            cavities_sub = cavities[::voxel_stride, ::voxel_stride, ::voxel_stride]
            x_sub = x_axis[::voxel_stride]
            y_sub = y_axis[::voxel_stride]
            z_sub = z_axis[::voxel_stride]
        else:
            inside_sub = inside
            cavities_sub = cavities
            x_sub, y_sub, z_sub = x_axis, y_axis, z_axis
            
        # Build grid of voxel centers
        X, Y, Z = np.meshgrid(x_sub, y_sub, z_sub, indexing="ij")
        
        # Scalar fields for Volume plots (0/1 occupancy)
        values_target = inside_sub.astype(float)
        values_cavity = cavities_sub.astype(float)
        
        # Flatten for Plotly Volume
        Xr = X.ravel()
        Yr = Y.ravel()
        Zr = Z.ravel()
        Vr_target = values_target.ravel()
        Vr_cavity = values_cavity.ravel()
        
        # --- 2. Molecule representation from helper (atoms + bonds + menus) ---
        # Get coords/elements from universe (already at frame_index)
        ag = u.select_atoms(volume_analyzer.selection)
        coords = ag.positions.copy()
        elements = [volume_analyzer._get_element(atom) or "C" for atom in ag.atoms]
        
        atom_trace, bond_trace, annotations_id, annotations_length, updatemenus = \
            make_molecule_components(coords, elements)
        
        # --- 3. Volume traces for target & cavities ---
        # target volume: semi-transparent gray shell
        vol_target = go.Volume(
            x=Xr,
            y=Yr,
            z=Zr,
            value=Vr_target,
            isomin=0.5,   # "inside" voxels are 1; surface at ~0.5
            isomax=1.5,
            opacity=0.1,  # overall opacity of the volume
            surface_count=20,
            name="target volume",
            colorscale=[[0, "rgba(0,0,0,0)"], [1, "grey"]],
            showscale=False,
            showlegend=True,
        )
        
        # Cavity volume: more opaque red shell
        vol_cavity = go.Volume(
            x=Xr,
            y=Yr,
            z=Zr,
            value=Vr_cavity,
            isomin=0.5,
            isomax=1.5,
            opacity=0.3,
            surface_count=20,
            name="Cavity volume",
            colorscale=[[0, "rgba(0,0,0,0)"], [1, "red"]],
            showscale=False,
            showlegend=True,
        )
        
        # --- 4. Layout (same axis style as your molecule plot) ---
        axis_params = dict(
            showgrid=True,
            showbackground=True,
            showticklabels=True,
            zeroline=False,
            tickfont=dict(color='black'),
        )
        
        # existing annotation menu from make_molecule_components
        annotation_menus = updatemenus
        # Trace order: 0: atoms, 1: bonds, 2: target volume, 3: cavity volume
        volume_menu = dict(
            type="buttons",
            direction="right",
            x=0.5,
            y=1.08,
            xanchor="center",
            yanchor="bottom",
            buttons=[
                dict(
                    label="Show both vols",
                    method="update",
                    args=[{"visible": [True, True, True, True]}],
                ),
                dict(
                    label="Hide vols",
                    method="update",
                    args=[{"visible": [True, True, False, False]}],
                ),
                dict(
                   label="Only target vol",
                    method="update",
                    args=[{"visible": [True, True, True, False]}],
                ),
                dict(
                    label="Only cavity vol",
                    method="update",
                    args=[{"visible": [True, True, False, True]}],
                ),
            ],
        )        
        
        layout = dict(
            scene=dict(
                xaxis=axis_params,
                yaxis=axis_params,
                zaxis=axis_params,
                annotations=annotations_id,
                aspectmode="data",
            ),
            margin=dict(r=100, l=100, b=100, t=100),
            showlegend=True,
            updatemenus=updatemenus+[volume_menu],
            title=(
                f"Frame {frame_index} — "
                f"V_target ≈ {target_vol:.0f} Å³, "
                f"V_cavity ≈ {cavity_vol:.0f} Å³"
            ),
        )
        
        fig = go.Figure(
            data=[atom_trace, bond_trace, vol_target, vol_cavity],
            layout=layout,
        )
        return fig


