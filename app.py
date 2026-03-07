import gradio as gr
import torch
import spaces
import numpy as np
import supervision as sv
from typing import Iterable
from gradio.themes import Soft
from gradio.themes.utils import colors, fonts, sizes
import json
import ast
import re
import cv2
import tempfile
from PIL import Image, ImageDraw, ImageFont
from threading import Thread
from transformers import (
    Qwen3_5ForConditionalGeneration,
    AutoProcessor,
    TextIteratorStreamer,
)

try:
    from qwen_vl_utils import process_vision_info
    HAS_QWEN_VL_UTILS = True
except ImportError:
    HAS_QWEN_VL_UTILS = False
    print("[WARN] qwen_vl_utils not found. Install: pip install qwen-vl-utils")
    print("       Video QA will use manual frame-extraction fallback.")


colors.steel_blue = colors.Color(
    name="steel_blue",
    c50="#EBF3F8", c100="#D3E5F0", c200="#A8CCE1", c300="#7DB3D2",
    c400="#529AC3", c500="#4682B4", c600="#3E72A0", c700="#36638C",
    c800="#2E5378", c900="#264364", c950="#1E3450",
)


class SteelBlueTheme(Soft):
    def __init__(
        self,
        *,
        primary_hue: colors.Color | str = colors.gray,
        secondary_hue: colors.Color | str = colors.steel_blue,
        neutral_hue: colors.Color | str = colors.slate,
        text_size: sizes.Size | str = sizes.text_lg,
        font: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("Outfit"), "Arial", "sans-serif",
        ),
        font_mono: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue, secondary_hue=secondary_hue,
            neutral_hue=neutral_hue, text_size=text_size,
            font=font, font_mono=font_mono,
        )
        super().set(
            background_fill_primary="*primary_50",
            background_fill_primary_dark="*primary_900",
            body_background_fill="linear-gradient(135deg, *primary_200, *primary_100)",
            body_background_fill_dark="linear-gradient(135deg, *primary_900, *primary_800)",
            button_primary_text_color="white",
            button_primary_text_color_hover="white",
            button_primary_background_fill="linear-gradient(90deg, *secondary_500, *secondary_600)",
            button_primary_background_fill_hover="linear-gradient(90deg, *secondary_600, *secondary_700)",
            button_primary_background_fill_dark="linear-gradient(90deg, *secondary_600, *secondary_800)",
            button_primary_background_fill_hover_dark="linear-gradient(90deg, *secondary_500, *secondary_500)",
            button_secondary_text_color="black",
            button_secondary_text_color_hover="white",
            button_secondary_background_fill="linear-gradient(90deg, *primary_300, *primary_300)",
            button_secondary_background_fill_hover="linear-gradient(90deg, *primary_400, *primary_400)",
            button_secondary_background_fill_dark="linear-gradient(90deg, *primary_500, *primary_600)",
            button_secondary_background_fill_hover_dark="linear-gradient(90deg, *primary_500, *primary_500)",
            slider_color="*secondary_500",
            slider_color_dark="*secondary_600",
            block_title_text_weight="600",
            block_border_width="3px",
            block_shadow="*shadow_drop_lg",
            button_primary_shadow="*shadow_drop_lg",
            button_large_padding="11px",
            color_accent_soft="*primary_100",
            block_label_background_fill="*primary_200",
        )


steel_blue_theme = SteelBlueTheme()


css = r"""
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

body, .gradio-container { font-family: 'Outfit', sans-serif !important; }
footer { display: none !important; }

/* -- App Header -- */
.app-header {
    background: linear-gradient(135deg, #1E3450 0%, #264364 30%, #3E72A0 70%, #4682B4 100%);
    border-radius: 16px; padding: 32px 40px; margin-bottom: 24px;
    position: relative; overflow: hidden;
    box-shadow: 0 8px 32px rgba(30,52,80,0.25);
}
.app-header::before {
    content:''; position:absolute; top:-50%; right:-20%;
    width:400px; height:400px;
    background:radial-gradient(circle,rgba(255,255,255,0.06) 0%,transparent 70%);
    border-radius:50%;
}
.app-header::after {
    content:''; position:absolute; bottom:-30%; left:-10%;
    width:300px; height:300px;
    background:radial-gradient(circle,rgba(70,130,180,0.15) 0%,transparent 70%);
    border-radius:50%;
}
.header-content {
    display:flex; align-items:center; gap:24px;
    position:relative; z-index:1;
}
.header-icon-wrap {
    width:64px; height:64px; background:rgba(255,255,255,0.12);
    border-radius:16px; display:flex; align-items:center; justify-content:center;
    flex-shrink:0; backdrop-filter:blur(8px); border:1px solid rgba(255,255,255,0.15);
}
.header-icon-wrap svg { width:36px; height:36px; color:rgba(255,255,255,0.9); }
.header-text h1 {
    font-family:'Outfit',sans-serif; font-size:2rem; font-weight:700;
    color:#fff; margin:0 0 8px 0; letter-spacing:-0.02em; line-height:1.2;
}
.header-meta { display:flex; align-items:center; gap:12px; flex-wrap:wrap; }
.meta-badge {
    display:inline-flex; align-items:center; gap:6px;
    background:rgba(255,255,255,0.12); color:rgba(255,255,255,0.9);
    padding:4px 12px; border-radius:20px;
    font-family:'IBM Plex Mono',monospace; font-size:0.8rem; font-weight:500;
    border:1px solid rgba(255,255,255,0.1); backdrop-filter:blur(4px);
}
.meta-badge svg { width:14px; height:14px; }
.meta-sep {
    width:4px; height:4px; background:rgba(255,255,255,0.35);
    border-radius:50%; flex-shrink:0;
}
.meta-cap { color:rgba(255,255,255,0.65); font-size:0.85rem; font-weight:400; }

/* -- Tab transitions -- */
.gradio-tabitem { animation: tabFadeIn 0.35s ease-out; }
@keyframes tabFadeIn {
    from { opacity:0; transform:translateY(6px); }
    to   { opacity:1; transform:translateY(0); }
}

/* -- Tab Intro Panels -- */
.tab-intro {
    display:flex; align-items:flex-start; gap:16px;
    background:linear-gradient(135deg,rgba(70,130,180,0.06),rgba(70,130,180,0.02));
    border:1px solid rgba(70,130,180,0.15); border-left:4px solid #4682B4;
    border-radius:10px; padding:18px 22px; margin-bottom:20px;
}
.dark .tab-intro {
    background:linear-gradient(135deg,rgba(70,130,180,0.1),rgba(70,130,180,0.04));
    border-color:rgba(70,130,180,0.25);
}
.intro-icon {
    width:40px; height:40px; background:rgba(70,130,180,0.1);
    border-radius:10px; display:flex; align-items:center; justify-content:center;
    flex-shrink:0; margin-top:2px;
}
.intro-icon svg { width:22px; height:22px; color:#4682B4; }
.dark .intro-icon svg { color:#7DB3D2; }
.intro-text { flex:1; }
.intro-text p { margin:0; color:#2E5378; font-size:0.95rem; line-height:1.6; }
.dark .intro-text p { color:#A8CCE1; }
.intro-text p.intro-sub { color:#64748b; font-size:0.85rem; margin-top:4px; }
.dark .intro-text p.intro-sub { color:#94a3b8; }

/* -- Section Headers -- */
.section-heading {
    display:flex; align-items:center; gap:14px;
    margin:22px 0 14px 0; padding:0 2px;
}
.heading-icon {
    width:32px; height:32px;
    background:linear-gradient(135deg,#4682B4,#3E72A0);
    border-radius:8px; display:flex; align-items:center; justify-content:center;
    flex-shrink:0; box-shadow:0 2px 8px rgba(70,130,180,0.2);
}
.heading-icon svg { width:18px; height:18px; color:#fff; }
.heading-label {
    font-family:'Outfit',sans-serif; font-weight:600; font-size:1.05rem;
    color:#1E3450; letter-spacing:-0.01em;
}
.dark .heading-label { color:#D3E5F0; }
.heading-line {
    flex:1; height:1px;
    background:linear-gradient(90deg,rgba(70,130,180,0.2),transparent);
}

/* -- Status Indicators -- */
.status-indicator {
    display:flex; align-items:center; gap:10px;
    padding:10px 16px; margin-top:10px;
    background:rgba(70,130,180,0.04); border:1px solid rgba(70,130,180,0.12);
    border-radius:8px;
}
.dark .status-indicator {
    background:rgba(70,130,180,0.08); border-color:rgba(70,130,180,0.2);
}
.status-dot {
    width:8px; height:8px; background:#22c55e;
    border-radius:50%; flex-shrink:0;
    animation:statusPulse 2s ease-in-out infinite;
}
@keyframes statusPulse {
    0%,100% { opacity:1; box-shadow:0 0 0 0 rgba(34,197,94,0.4); }
    50%     { opacity:0.7; box-shadow:0 0 0 4px rgba(34,197,94,0); }
}
.status-text { font-size:0.85rem; color:#64748b; font-style:italic; }
.dark .status-text { color:#94a3b8; }

/* -- Card Labels -- */
.card-label {
    display:flex; align-items:center; gap:8px;
    font-family:'Outfit',sans-serif; font-weight:600; font-size:0.8rem;
    text-transform:uppercase; letter-spacing:0.06em; color:#4682B4;
    margin-bottom:14px; padding-bottom:10px;
    border-bottom:1px solid rgba(70,130,180,0.1);
}
.dark .card-label { color:#7DB3D2; border-bottom-color:rgba(70,130,180,0.2); }
.card-label svg { width:16px; height:16px; }

/* -- Buttons -- */
.primary {
    border-radius:10px !important; font-weight:600 !important;
    letter-spacing:0.02em !important; transition:all 0.25s ease !important;
    font-family:'Outfit',sans-serif !important;
}
.primary:hover {
    transform:translateY(-2px) !important;
    box-shadow:0 6px 20px rgba(70,130,180,0.3) !important;
}
.primary:active { transform:translateY(0) !important; }

/* -- Textbox -- */
.gradio-textbox textarea {
    font-family:'IBM Plex Mono',monospace !important;
    font-size:0.92rem !important; line-height:1.7 !important;
    border-radius:8px !important;
}

/* -- Accordion -- */
.gradio-accordion {
    border-radius:10px !important; border:1px solid rgba(70,130,180,0.15) !important;
}
.gradio-accordion>.label-wrap { border-radius:10px !important; }

/* -- Labels -- */
label { font-weight:600 !important; font-family:'Outfit',sans-serif !important; }

/* -- Slider -- */
.gradio-slider input[type="range"] { accent-color:#4682B4 !important; }

/* -- Scrollbar -- */
::-webkit-scrollbar { width:8px; height:8px; }
::-webkit-scrollbar-track { background:rgba(70,130,180,0.04); border-radius:4px; }
::-webkit-scrollbar-thumb { background:linear-gradient(135deg,#4682B4,#3E72A0); border-radius:4px; }
::-webkit-scrollbar-thumb:hover { background:linear-gradient(135deg,#3E72A0,#2E5378); }

/* -- Gallery -- */
.gradio-gallery { border-radius:10px !important; }

/* -- Divider -- */
.section-divider {
    height:1px; background:linear-gradient(90deg,transparent,rgba(70,130,180,0.2),transparent);
    margin:16px 0; border:none;
}

/* ============================== */
/* -- Graph Indicator Panel --    */
/* ============================== */

.graph-panel {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 20px;
    margin: 10px 0;
    box-shadow: 0 1px 4px rgba(0,0,0,0.03);
    transition: box-shadow 0.3s ease;
}
.graph-panel:hover {
    box-shadow: 0 3px 14px rgba(70,130,180,0.08);
}
.dark .graph-panel {
    background: rgba(30,52,80,0.35);
    border-color: rgba(70,130,180,0.2);
}

.graph-panel-header {
    display: flex;
    align-items: center;
    gap: 8px;
    font-family: 'Outfit', sans-serif;
    font-weight: 600;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #4682B4;
    margin-bottom: 16px;
    padding-bottom: 12px;
    border-bottom: 1px solid #f1f5f9;
}
.dark .graph-panel-header {
    color: #7DB3D2;
    border-bottom-color: rgba(70,130,180,0.15);
}
.graph-panel-header svg { width: 16px; height: 16px; }

/* Metric Cards Grid */
.graph-metrics-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
    margin-bottom: 18px;
}

.graph-metric-card {
    background: #f8fafc;
    border: 1px solid #f1f5f9;
    border-radius: 10px;
    padding: 16px 14px;
    text-align: center;
    transition: border-color 0.2s ease;
}
.graph-metric-card:first-child { border-left: 3px solid #4682B4; }
.graph-metric-card:last-child  { border-left: 3px solid #3E72A0; }
.graph-metric-card:hover { border-color: #A8CCE1; }
.dark .graph-metric-card {
    background: rgba(30,52,80,0.45);
    border-color: rgba(70,130,180,0.15);
}
.dark .graph-metric-card:first-child { border-left-color: #529AC3; }
.dark .graph-metric-card:last-child  { border-left-color: #4682B4; }

.graph-metric-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.75rem;
    font-weight: 700;
    color: #1E3450;
    line-height: 1.1;
}
.dark .graph-metric-value { color: #D3E5F0; }

.graph-metric-unit {
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #94a3b8;
    font-weight: 600;
    margin-top: 2px;
    margin-bottom: 10px;
}

.graph-metric-bar-track {
    height: 6px;
    background: #e2e8f0;
    border-radius: 3px;
    overflow: hidden;
}
.dark .graph-metric-bar-track { background: rgba(100,116,139,0.2); }

.graph-metric-bar-fill {
    height: 100%;
    border-radius: 3px;
    animation: graphBarGrow 0.55s ease-out;
    transform-origin: left;
}
.graph-bar-primary  { background: linear-gradient(90deg, #4682B4, #529AC3); }
.graph-bar-secondary { background: linear-gradient(90deg, #3E72A0, #4682B4); }

@keyframes graphBarGrow {
    from { transform: scaleX(0); }
    to   { transform: scaleX(1); }
}

.graph-metric-pct {
    font-size: 0.68rem;
    color: #94a3b8;
    margin-top: 6px;
    font-family: 'IBM Plex Mono', monospace;
}

/* Estimate Chart Section */
.graph-estimates {
    border-top: 1px solid #f1f5f9;
    padding-top: 16px;
}
.dark .graph-estimates { border-top-color: rgba(70,130,180,0.15); }

.graph-est-title {
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #94a3b8;
    font-weight: 600;
    margin-bottom: 12px;
    font-family: 'Outfit', sans-serif;
}

.graph-est-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 10px;
}
.graph-est-row:last-child { margin-bottom: 0; }

.graph-est-label {
    width: 32px;
    font-size: 0.78rem;
    font-weight: 600;
    color: #475569;
    text-align: right;
    flex-shrink: 0;
    font-family: 'IBM Plex Mono', monospace;
}
.dark .graph-est-label { color: #94a3b8; }

.graph-est-track {
    flex: 1;
    height: 26px;
    background: #f1f5f9;
    border-radius: 7px;
    overflow: hidden;
    position: relative;
}
.dark .graph-est-track { background: rgba(100,116,139,0.15); }

.graph-est-fill {
    height: 100%;
    border-radius: 7px;
    min-width: 4px;
    animation: graphBarGrow 0.55s ease-out;
    transform-origin: left;
    position: relative;
}

.graph-est-fill-normal {
    background: linear-gradient(90deg, #4682B4, #529AC3);
}
.graph-est-fill-capped {
    background: linear-gradient(90deg, #e69500, #cc8400);
}

.graph-est-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    font-weight: 600;
    color: #1E3450;
    min-width: 80px;
    text-align: right;
    flex-shrink: 0;
}
.dark .graph-est-value { color: #D3E5F0; }

/* Capped Badge */
.graph-est-badge {
    display: inline-block;
    font-size: 0.6rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    padding: 1px 6px;
    border-radius: 4px;
    margin-left: 6px;
    vertical-align: middle;
}
.badge-capped {
    background: rgba(230,149,0,0.12);
    color: #b47b00;
    border: 1px solid rgba(230,149,0,0.25);
}
.dark .badge-capped {
    background: rgba(230,149,0,0.15);
    color: #f0c040;
    border-color: rgba(230,149,0,0.3);
}

/* Graph Note */
.graph-note {
    margin-top: 12px;
    padding: 8px 12px;
    background: rgba(230,149,0,0.06);
    border: 1px solid rgba(230,149,0,0.15);
    border-left: 3px solid #e69500;
    border-radius: 6px;
    font-size: 0.72rem;
    color: #92400e;
    font-family: 'Outfit', sans-serif;
    line-height: 1.5;
}
.dark .graph-note {
    background: rgba(230,149,0,0.08);
    border-color: rgba(230,149,0,0.2);
    border-left-color: #cc8400;
    color: #fbbf24;
}

/* -- Responsive -- */
@media (max-width: 768px) {
    .app-header { padding: 20px 24px; }
    .header-text h1 { font-size: 1.5rem; }
    .header-content { flex-direction: column; align-items: flex-start; gap: 16px; }
    .header-meta { gap: 8px; }
    .graph-metrics-grid { grid-template-columns: 1fr; gap: 10px; }
}
"""


SVG_BRAIN = '<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M9.813 15.904 9 18.75l-.813-2.846a4.5 4.5 0 0 0-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 0 0 3.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 0 0 3.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 0 0-3.09 3.09ZM18.259 8.715 18 9.75l-.259-1.035a3.375 3.375 0 0 0-2.455-2.456L14.25 6l1.036-.259a3.375 3.375 0 0 0 2.455-2.456L18 2.25l.259 1.035a3.375 3.375 0 0 0 2.456 2.456L21.75 6l-1.035.259a3.375 3.375 0 0 0-2.456 2.456ZM16.894 20.567 16.5 21.75l-.394-1.183a2.25 2.25 0 0 0-1.423-1.423L13.5 18.75l1.183-.394a2.25 2.25 0 0 0 1.423-1.423l.394-1.183.394 1.183a2.25 2.25 0 0 0 1.423 1.423l1.183.394-1.183.394a2.25 2.25 0 0 0-1.423 1.423Z"/></svg>'
SVG_IMAGE = '<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="m2.25 15.75 5.159-5.159a2.25 2.25 0 0 1 3.182 0l5.159 5.159m-1.5-1.5 1.409-1.409a2.25 2.25 0 0 1 3.182 0l2.909 2.909M3.75 21h16.5A2.25 2.25 0 0 0 22.5 18.75V5.25A2.25 2.25 0 0 0 20.25 3H3.75A2.25 2.25 0 0 0 1.5 5.25v13.5A2.25 2.25 0 0 0 3.75 21Z"/></svg>'
SVG_VIDEO = '<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="m15.75 10.5 4.72-4.72a.75.75 0 0 1 1.28.53v11.38a.75.75 0 0 1-1.28.53l-4.72-4.72M4.5 18.75h9a2.25 2.25 0 0 0 2.25-2.25v-9A2.25 2.25 0 0 0 13.5 5.25h-9A2.25 2.25 0 0 0 2.25 7.5v9A2.25 2.25 0 0 0 4.5 18.75Z"/></svg>'
SVG_DETECT = '<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M7.5 3.75H6A2.25 2.25 0 0 0 3.75 6v1.5M16.5 3.75H18A2.25 2.25 0 0 1 20.25 6v1.5m0 9V18A2.25 2.25 0 0 1 18 20.25h-1.5m-9 0H6A2.25 2.25 0 0 1 3.75 18v-1.5M15 12a3 3 0 1 1-6 0 3 3 0 0 1 6 0Z"/></svg>'
SVG_TRACK = '<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M15 10.5a3 3 0 1 1-6 0 3 3 0 0 1 6 0Z"/><path stroke-linecap="round" stroke-linejoin="round" d="M19.5 10.5c0 7.142-7.5 11.25-7.5 11.25S4.5 17.642 4.5 10.5a7.5 7.5 0 1 1 15 0Z"/></svg>'
SVG_SETTINGS = '<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M10.5 6h9.75M10.5 6a1.5 1.5 0 1 1-3 0m3 0a1.5 1.5 0 1 0-3 0M3.75 6H7.5m3 12h9.75m-9.75 0a1.5 1.5 0 0 1-3 0m3 0a1.5 1.5 0 0 0-3 0m-3.75 0H7.5m9-6h3.75m-3.75 0a1.5 1.5 0 0 1-3 0m3 0a1.5 1.5 0 0 0-3 0m-9.75 0h9.75"/></svg>'
SVG_CHIP = '<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M8.25 3v1.5M4.5 8.25H3m18 0h-1.5M4.5 12H3m18 0h-1.5m-15 3.75H3m18 0h-1.5M8.25 19.5V21M12 3v1.5m0 15V21m3.75-18v1.5m0 15V21m-9-1.5h10.5a2.25 2.25 0 0 0 2.25-2.25V6.75a2.25 2.25 0 0 0-2.25-2.25H6.75A2.25 2.25 0 0 0 4.5 6.75v10.5a2.25 2.25 0 0 0 2.25 2.25Z"/></svg>'
SVG_UPLOAD = '<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M3 16.5v2.25A2.25 2.25 0 0 0 5.25 21h13.5A2.25 2.25 0 0 0 21 18.75V16.5m-13.5-9L12 3m0 0 4.5 4.5M12 3v13.5"/></svg>'
SVG_OUTPUT = '<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M3.75 9.776c.112-.017.227-.026.344-.026h15.812c.117 0 .232.009.344.026m-16.5 0a2.25 2.25 0 0 0-1.883 2.542l.857 6a2.25 2.25 0 0 0 2.227 1.932H19.05a2.25 2.25 0 0 0 2.227-1.932l.857-6a2.25 2.25 0 0 0-1.883-2.542m-16.5 0V6A2.25 2.25 0 0 1 6 3.75h3.879a1.5 1.5 0 0 1 1.06.44l2.122 2.12a1.5 1.5 0 0 0 1.06.44H18A2.25 2.25 0 0 1 20.25 9v.776"/></svg>'
SVG_TEXT = '<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M7.5 8.25h9m-9 3H12m-9.75 1.51c0 1.6 1.123 2.994 2.707 3.227 1.129.166 2.27.293 3.423.379.35.026.67.21.865.501L12 21l2.755-4.133a1.14 1.14 0 0 1 .865-.501 48.172 48.172 0 0 0 3.423-.379c1.584-.233 2.707-1.626 2.707-3.228V6.741c0-1.602-1.123-2.995-2.707-3.228A48.394 48.394 0 0 0 12 3c-2.392 0-4.744.175-7.043.513C3.373 3.746 2.25 5.14 2.25 6.741v6.018Z"/></svg>'
SVG_CHART = '<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M3 13.125C3 12.504 3.504 12 4.125 12h2.25c.621 0 1.125.504 1.125 1.125v6.75C7.5 20.496 6.996 21 6.375 21h-2.25A1.125 1.125 0 0 1 3 19.875v-6.75ZM9.75 8.625c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125v11.25c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 0 1-1.125-1.125V8.625ZM16.5 4.125c0-.621.504-1.125 1.125-1.125h2.25C20.496 3 21 3.504 21 4.125v15.75c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 0 1-1.125-1.125V4.125Z"/></svg>'

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = (
    torch.bfloat16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else torch.float16
)

MODEL_NAME = "Qwen/Qwen3.5-2B"
CATEGORIES = ["Query", "Caption", "Point", "Detect"]

BRIGHT_YELLOW = sv.Color(r=255, g=230, b=0)
DARK_OUTLINE = sv.Color(r=40, g=40, b=40)
BLACK = sv.Color(r=0, g=0, b=0)
WHITE = sv.Color(r=255, g=255, b=255)

TRACK_RED = (255, 50, 50)
TRACK_WHITE = (255, 255, 255)
TRACK_BLACK = (0, 0, 0)

print(f"Loading model: {MODEL_NAME} ...")
qwen_model = Qwen3_5ForConditionalGeneration.from_pretrained(
    MODEL_NAME, torch_dtype=DTYPE, device_map=DEVICE,
).eval()
qwen_processor = AutoProcessor.from_pretrained(MODEL_NAME)
print("Model loaded.")


def safe_parse_json(text: str):
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text)
    text = re.sub(r"```$", "", text)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    try:
        return ast.literal_eval(text)
    except Exception:
        return {}


def annotate_image(image: Image.Image, result: dict):
    if not isinstance(image, Image.Image) or not isinstance(result, dict):
        return image
    image = image.convert("RGB")
    ow, oh = image.size

    if "points" in result and result["points"]:
        pts = [[int(p["x"] * ow), int(p["y"] * oh)] for p in result["points"]]
        if not pts:
            return image
        kp = sv.KeyPoints(xy=np.array(pts).reshape(1, -1, 2))
        scene = np.array(image.copy())
        scene = sv.VertexAnnotator(radius=8, color=DARK_OUTLINE).annotate(scene=scene, key_points=kp)
        scene = sv.VertexAnnotator(radius=5, color=BRIGHT_YELLOW).annotate(scene=scene, key_points=kp)
        labels = [p.get("label", "") for p in result["points"]]
        if any(labels):
            tb, vl = [], []
            for i, p in enumerate(result["points"]):
                if labels[i]:
                    cx, cy = int(p["x"] * ow), int(p["y"] * oh)
                    tb.append([cx - 2, cy - 2, cx + 2, cy + 2])
                    vl.append(labels[i])
            if tb:
                scene = sv.LabelAnnotator(
                    color=BRIGHT_YELLOW, text_color=BLACK, text_scale=0.5,
                    text_thickness=1, text_padding=5,
                    text_position=sv.Position.TOP_CENTER,
                    color_lookup=sv.ColorLookup.INDEX,
                ).annotate(scene=scene, detections=sv.Detections(xyxy=np.array(tb)), labels=vl)
        return Image.fromarray(scene)

    if "objects" in result and result["objects"]:
        boxes, labels = [], []
        for obj in result["objects"]:
            boxes.append([
                obj.get("x_min", 0.0) * ow, obj.get("y_min", 0.0) * oh,
                obj.get("x_max", 0.0) * ow, obj.get("y_max", 0.0) * oh,
            ])
            labels.append(obj.get("label", "object"))
        if not boxes:
            return image
        scene = np.array(image.copy())
        h, w = scene.shape[:2]
        masks = np.zeros((len(boxes), h, w), dtype=bool)
        for i, box in enumerate(boxes):
            x1, y1 = max(0, int(box[0])), max(0, int(box[1]))
            x2, y2 = min(w, int(box[2])), min(h, int(box[3]))
            masks[i, y1:y2, x1:x2] = True
        dets = sv.Detections(xyxy=np.array(boxes), mask=masks)
        if len(dets) == 0:
            return image
        scene = sv.MaskAnnotator(color=BRIGHT_YELLOW, opacity=0.18, color_lookup=sv.ColorLookup.INDEX).annotate(scene=scene, detections=dets)
        scene = sv.BoxAnnotator(color=BRIGHT_YELLOW, thickness=2, color_lookup=sv.ColorLookup.INDEX).annotate(scene=scene, detections=dets)
        scene = sv.LabelAnnotator(
            color=BRIGHT_YELLOW, text_color=BLACK, text_scale=0.5,
            text_thickness=1, text_padding=6, color_lookup=sv.ColorLookup.INDEX,
        ).annotate(scene=scene, detections=dets, labels=labels)
        return Image.fromarray(scene)
    return image


def annotate_image_red_points(image: Image.Image, result: dict):
    if not isinstance(image, Image.Image) or not isinstance(result, dict):
        return image
    image = image.convert("RGB")
    w, h = image.size
    if "points" not in result or not result["points"]:
        return image
    draw = ImageDraw.Draw(image)
    for p in result["points"]:
        cx, cy = int(p["x"] * w), int(p["y"] * h)
        draw.ellipse((cx - 10, cy - 10, cx + 10, cy + 10), outline=TRACK_WHITE, width=3)
        draw.ellipse((cx - 7, cy - 7, cx + 7, cy + 7), fill=TRACK_RED, outline=TRACK_RED)
        label = p.get("label", "")
        if label:
            try:
                font = ImageFont.truetype("arial.ttf", 14)
            except (IOError, OSError):
                font = ImageFont.load_default()
            bbox = draw.textbbox((cx + 14, cy - 8), label, font=font)
            draw.rectangle((bbox[0] - 3, bbox[1] - 3, bbox[2] + 3, bbox[3] + 3), fill=TRACK_RED)
            draw.text((cx + 14, cy - 8), label, fill=TRACK_WHITE, font=font)
    return image


def extract_video_frames(video_path, max_frames=16, target_fps=1.0):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total / fps if fps > 0 else 0
    n_desired = min(max_frames, max(1, int(duration * target_fps)))
    interval = max(1, total // n_desired)
    frames, indices = [], []
    for i in range(0, total, interval):
        if len(frames) >= max_frames:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            indices.append(i)
    cap.release()
    return frames, indices, fps, vid_w, vid_h, total


def reconstruct_annotated_video(video_path, all_results, frame_indices, annotator_fn):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = tempfile.mktemp(suffix=".mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (vid_w, vid_h))
    det_map = {fidx: all_results[i] for i, fidx in enumerate(frame_indices)}
    sorted_idx = sorted(det_map.keys())
    cur = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        nearest = min(sorted_idx, key=lambda x: abs(x - cur))
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        annotated = annotator_fn(pil, det_map[nearest])
        writer.write(cv2.cvtColor(np.array(annotated), cv2.COLOR_RGB2BGR))
        cur += 1
    cap.release()
    writer.release()
    return out_path


def _run_detection_on_frame(frame: Image.Image, prompt_text: str) -> dict:
    small = frame.copy()
    small.thumbnail((512, 512))
    messages = [{"role": "user", "content": [{"type": "image", "image": small}, {"type": "text", "text": prompt_text}]}]
    text = qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = qwen_processor(text=[text], images=[small], return_tensors="pt", padding=True).to(qwen_model.device)
    with torch.inference_mode():
        gen_ids = qwen_model.generate(**inputs, max_new_tokens=1024, use_cache=True, temperature=1.5, min_p=0.1)
    raw = qwen_processor.batch_decode(gen_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    parsed = safe_parse_json(raw)
    result: dict = {"objects": []}
    if isinstance(parsed, list):
        for item in parsed:
            if "bbox_2d" in item and len(item["bbox_2d"]) == 4:
                xmin, ymin, xmax, ymax = item["bbox_2d"]
                result["objects"].append({
                    "label": item.get("label", "object"),
                    "x_min": xmin / 1000.0, "y_min": ymin / 1000.0,
                    "x_max": xmax / 1000.0, "y_max": ymax / 1000.0,
                })
    return result


def _run_point_detection_on_frame(frame: Image.Image, prompt_text: str) -> dict:
    small = frame.copy()
    small.thumbnail((512, 512))
    messages = [{"role": "user", "content": [{"type": "image", "image": small}, {"type": "text", "text": prompt_text}]}]
    text = qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = qwen_processor(text=[text], images=[small], return_tensors="pt", padding=True).to(qwen_model.device)
    with torch.inference_mode():
        gen_ids = qwen_model.generate(**inputs, max_new_tokens=1024, use_cache=True, temperature=1.5, min_p=0.1)
    raw = qwen_processor.batch_decode(gen_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    parsed = safe_parse_json(raw)
    result: dict = {"points": []}
    if isinstance(parsed, list):
        for item in parsed:
            if "point_2d" in item and len(item["point_2d"]) == 2:
                x, y = item["point_2d"]
                result["points"].append({"label": item.get("label", ""), "x": x / 1000.0, "y": y / 1000.0})
    return result


FPS_SLIDER_MAX = 48.0
FRAMES_SLIDER_MAX = 120


def _build_graph_panel(title, rate_label, rate_value, rate_unit,
                       cap_value, sample_fps, max_frames):
    """Build a visual graph-indicator HTML panel."""
    n30 = min(int(max_frames), int(30 * sample_fps))
    n60 = min(int(max_frames), int(60 * sample_fps))

    fps_pct = min(100, (sample_fps / FPS_SLIDER_MAX) * 100)
    frames_pct = min(100, (max_frames / FRAMES_SLIDER_MAX) * 100)

    safe_cap = max(int(max_frames), 1)
    n30_pct = min(100, (n30 / safe_cap) * 100)
    n60_pct = min(100, (n60 / safe_cap) * 100)

    n30_capped = n30 >= int(max_frames) and int(30 * sample_fps) > int(max_frames)
    n60_capped = n60 >= int(max_frames) and int(60 * sample_fps) > int(max_frames)

    n30_fill_cls = "graph-est-fill-capped" if n30_capped else "graph-est-fill-normal"
    n60_fill_cls = "graph-est-fill-capped" if n60_capped else "graph-est-fill-normal"

    n30_badge = '<span class="graph-est-badge badge-capped">capped</span>' if n30_capped else ""
    n60_badge = '<span class="graph-est-badge badge-capped">capped</span>' if n60_capped else ""

    note = ""
    if n30_capped or n60_capped:
        note = (
            '<div class="graph-note">'
            "One or more estimates have reached the frame cap. "
            "Increase the Max Frames slider to sample more frames from longer videos."
            "</div>"
        )

    return f"""
    <div class="graph-panel">
        <div class="graph-panel-header">
            {SVG_CHART}
            <span>{title}</span>
        </div>

        <div class="graph-metrics-grid">
            <div class="graph-metric-card">
                <div class="graph-metric-value">{rate_value}</div>
                <div class="graph-metric-unit">{rate_unit}</div>
                <div class="graph-metric-bar-track">
                    <div class="graph-metric-bar-fill graph-bar-primary"
                         style="width:{fps_pct:.1f}%"></div>
                </div>
                <div class="graph-metric-pct">{fps_pct:.0f}% of {FPS_SLIDER_MAX:.0f} max</div>
            </div>
            <div class="graph-metric-card">
                <div class="graph-metric-value">{int(max_frames)}</div>
                <div class="graph-metric-unit">Frame Cap</div>
                <div class="graph-metric-bar-track">
                    <div class="graph-metric-bar-fill graph-bar-secondary"
                         style="width:{frames_pct:.1f}%"></div>
                </div>
                <div class="graph-metric-pct">{frames_pct:.0f}% of {int(FRAMES_SLIDER_MAX)} max</div>
            </div>
        </div>

        <div class="graph-estimates">
            <div class="graph-est-title">Estimated Frames by Video Duration</div>
            <div class="graph-est-row">
                <span class="graph-est-label">30s</span>
                <div class="graph-est-track">
                    <div class="graph-est-fill {n30_fill_cls}"
                         style="width:{n30_pct:.1f}%"></div>
                </div>
                <span class="graph-est-value">{n30} frames{n30_badge}</span>
            </div>
            <div class="graph-est-row">
                <span class="graph-est-label">60s</span>
                <div class="graph-est-track">
                    <div class="graph-est-fill {n60_fill_cls}"
                         style="width:{n60_pct:.1f}%"></div>
                </div>
                <span class="graph-est-value">{n60} frames{n60_badge}</span>
            </div>
        </div>
        {note}
    </div>
    """


def update_sampling_info(sample_fps, max_frames):
    return _build_graph_panel(
        title="Detection Sampling Metrics",
        rate_label="Sample Rate",
        rate_value=f"{sample_fps:.1f}",
        rate_unit="Sample FPS",
        cap_value=int(max_frames),
        sample_fps=sample_fps,
        max_frames=max_frames,
    )


def update_tracking_info(sample_fps, max_frames):
    return _build_graph_panel(
        title="Tracking Sampling Metrics",
        rate_label="Track Rate",
        rate_value=f"{sample_fps:.1f}",
        rate_unit="Track FPS",
        cap_value=int(max_frames),
        sample_fps=sample_fps,
        max_frames=max_frames,
    )


@spaces.GPU
def process_inputs(image, category, prompt):
    if image is None:
        raise gr.Error("Please upload an image.")
    if not prompt or not prompt.strip():
        raise gr.Error("Please provide a prompt.")
    image = image.convert("RGB")
    image.thumbnail((512, 512))
    if category == "Query":
        full_prompt = prompt
    elif category == "Caption":
        full_prompt = f"Provide a {prompt} length caption for the image."
    elif category == "Point":
        full_prompt = f"Provide 2d point coordinates for {prompt}. Report in JSON format."
    elif category == "Detect":
        full_prompt = f"Provide bounding box coordinates for {prompt}. Report in JSON format."
    else:
        full_prompt = prompt
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": full_prompt}]}]
    text = qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = qwen_processor(text=[text], images=[image], return_tensors="pt", padding=True).to(qwen_model.device)
    streamer = TextIteratorStreamer(qwen_processor.tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=120)
    thread = Thread(target=qwen_model.generate, kwargs=dict(**inputs, streamer=streamer, max_new_tokens=1024, use_cache=True, temperature=1.5, min_p=0.1))
    thread.start()
    full_text = ""
    for tok in streamer:
        full_text += tok
        yield image, full_text
    thread.join()
    if category == "Point":
        parsed = safe_parse_json(full_text)
        result = {"points": []}
        if isinstance(parsed, list):
            for item in parsed:
                if "point_2d" in item and len(item["point_2d"]) == 2:
                    x, y = item["point_2d"]
                    result["points"].append({"label": item.get("label", ""), "x": x / 1000.0, "y": y / 1000.0})
        yield annotate_image(image.copy(), result), json.dumps(result, indent=2)
    elif category == "Detect":
        parsed = safe_parse_json(full_text)
        result = {"objects": []}
        if isinstance(parsed, list):
            for item in parsed:
                if "bbox_2d" in item and len(item["bbox_2d"]) == 4:
                    xmin, ymin, xmax, ymax = item["bbox_2d"]
                    result["objects"].append({
                        "label": item.get("label", "object"),
                        "x_min": xmin / 1000.0, "y_min": ymin / 1000.0,
                        "x_max": xmax / 1000.0, "y_max": ymax / 1000.0,
                    })
        yield annotate_image(image.copy(), result), json.dumps(result, indent=2)


def on_category_change(category: str):
    placeholders = {
        "Query": "e.g., Count the total number of boats and describe the environment.",
        "Caption": "e.g., short, normal, detailed",
        "Point": "e.g., The gun held by the person.",
        "Detect": "e.g., The headlight of the car.",
    }
    return gr.Textbox(placeholder=placeholders.get(category, "Enter your prompt here."))


@spaces.GPU
def process_video_qa(video_path, prompt):
    if video_path is None:
        raise gr.Error("Please upload a video.")
    if not prompt or not prompt.strip():
        raise gr.Error("Please provide a prompt.")
    if HAS_QWEN_VL_UTILS:
        messages = [{"role": "user", "content": [
            {"type": "video", "video": video_path, "max_pixels": 360 * 420, "fps": 1.0},
            {"type": "text", "text": prompt},
        ]}]
        text = qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        try:
            _vis = process_vision_info(messages)
            if isinstance(_vis, tuple) and len(_vis) >= 3:
                image_inputs, video_inputs = _vis[0], _vis[1]
            else:
                image_inputs, video_inputs = _vis
            inputs = qwen_processor(
                text=[text],
                images=image_inputs if image_inputs else None,
                videos=video_inputs if video_inputs else None,
                padding=True, return_tensors="pt",
            ).to(qwen_model.device)
        except Exception:
            inputs = None
    else:
        inputs = None
    if inputs is None:
        frames, _, _, _, _, _ = extract_video_frames(video_path, max_frames=8, target_fps=1.0)
        if not frames:
            raise gr.Error("Could not extract any frames from the video.")
        content = [{"type": "text", "text": f"The following images are sampled frames from a video. {prompt}"}]
        img_list = []
        for f in frames:
            f.thumbnail((512, 512))
            content.append({"type": "image", "image": f})
            img_list.append(f)
        messages = [{"role": "user", "content": content}]
        text = qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = qwen_processor(text=[text], images=img_list, return_tensors="pt", padding=True).to(qwen_model.device)
    streamer = TextIteratorStreamer(qwen_processor.tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=180)
    thread = Thread(target=qwen_model.generate, kwargs=dict(**inputs, streamer=streamer, max_new_tokens=1024, use_cache=True, temperature=1.5, min_p=0.1))
    thread.start()
    full_text = ""
    for tok in streamer:
        full_text += tok
        yield full_text
    thread.join()


@spaces.GPU
def process_video_detection(video_path, prompt, sample_fps, max_frames, progress=gr.Progress()):
    if video_path is None:
        raise gr.Error("Please upload a video.")
    if not prompt or not prompt.strip():
        raise gr.Error("Please specify what to detect.")
    sample_fps = max(0.1, min(float(sample_fps), 5.0))
    max_frames = max(1, min(int(max_frames), 60))
    frames, frame_indices, fps, vid_w, vid_h, total = extract_video_frames(video_path, max_frames=max_frames, target_fps=sample_fps)
    if not frames:
        raise gr.Error("Could not extract frames from the video.")
    det_prompt = f"Provide bounding box coordinates for {prompt}. Report in JSON format."
    all_results, gallery_images = [], []
    for i, frame in enumerate(progress.tqdm(frames, desc="Detecting objects in frames")):
        result = _run_detection_on_frame(frame, det_prompt)
        all_results.append(result)
        gallery_images.append(annotate_image(frame.copy(), result))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    annotated_video_path = reconstruct_annotated_video(video_path, all_results, frame_indices, annotate_image)
    duration = total / fps if fps > 0 else 0
    summary = json.dumps({
        "sampling_fps": sample_fps, "max_frames_cap": int(max_frames),
        "frames_actually_sampled": len(frames), "video_fps": round(fps, 2),
        "video_duration_seconds": round(duration, 2),
        "video_resolution": f"{vid_w}x{vid_h}",
        "total_video_frames": total,
        "per_frame_detections": all_results,
    }, indent=2)
    return annotated_video_path, gallery_images, summary


@spaces.GPU
def process_video_point_tracking(video_path, prompt, sample_fps, max_frames, progress=gr.Progress()):
    if video_path is None:
        raise gr.Error("Please upload a video.")
    if not prompt or not prompt.strip():
        raise gr.Error("Please specify what to track.")
    sample_fps = max(0.1, min(float(sample_fps), 5.0))
    max_frames = max(1, min(int(max_frames), 60))
    frames, frame_indices, fps, vid_w, vid_h, total = extract_video_frames(video_path, max_frames=max_frames, target_fps=sample_fps)
    if not frames:
        raise gr.Error("Could not extract frames from the video.")
    point_prompt = f"Provide 2d point coordinates for {prompt}. Report in JSON format."
    all_results, gallery_images = [], []
    for i, frame in enumerate(progress.tqdm(frames, desc="Tracking points in frames")):
        result = _run_point_detection_on_frame(frame, point_prompt)
        all_results.append(result)
        gallery_images.append(annotate_image_red_points(frame.copy(), result))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    annotated_video_path = reconstruct_annotated_video(video_path, all_results, frame_indices, annotate_image_red_points)
    duration = total / fps if fps > 0 else 0
    per_frame_summary = []
    for fidx, result in zip(frame_indices, all_results):
        ts = round(fidx / fps, 2) if fps > 0 else 0
        per_frame_summary.append({
            "frame_index": fidx, "timestamp_sec": ts,
            "num_points": len(result.get("points", [])),
            "points": result.get("points", []),
        })
    summary = json.dumps({
        "tracking_target": prompt, "sampling_fps": sample_fps,
        "max_frames_cap": int(max_frames),
        "frames_actually_sampled": len(frames),
        "video_fps": round(fps, 2),
        "video_duration_seconds": round(duration, 2),
        "video_resolution": f"{vid_w}x{vid_h}",
        "total_video_frames": total,
        "total_points_found": sum(len(r.get("points", [])) for r in all_results),
        "per_frame_tracking": per_frame_summary,
    }, indent=2)
    return annotated_video_path, gallery_images, summary


def html_header():
    return f"""
    <div class="app-header">
        <div class="header-content">
            <div class="header-icon-wrap">{SVG_BRAIN}</div>
            <div class="header-text">
                <h1>Qwen 3.5 &mdash; Multimodal Understanding</h1>
                <div class="header-meta">
                    <span class="meta-badge">{SVG_CHIP} {MODEL_NAME}</span>
                    <span class="meta-sep"></span>
                    <span class="meta-cap">Image QA</span>
                    <span class="meta-sep"></span>
                    <span class="meta-cap">Video QA</span>
                    <span class="meta-sep"></span>
                    <span class="meta-cap">Video Detection</span>
                    <span class="meta-sep"></span>
                    <span class="meta-cap">Point Tracking</span>
                </div>
            </div>
        </div>
    </div>
    """


def html_tab_intro(icon_svg, title, description, detail=""):
    sub = f'<p class="intro-sub">{detail}</p>' if detail else ""
    return f"""
    <div class="tab-intro">
        <div class="intro-icon">{icon_svg}</div>
        <div class="intro-text">
            <p><strong>{title}</strong> &mdash; {description}</p>
            {sub}
        </div>
    </div>
    """


def html_section_heading(icon_svg, label):
    return f"""
    <div class="section-heading">
        <div class="heading-icon">{icon_svg}</div>
        <span class="heading-label">{label}</span>
        <div class="heading-line"></div>
    </div>
    """


def html_card_label(icon_svg, label):
    return f'<div class="card-label">{icon_svg}<span>{label}</span></div>'


def html_status_indicator(text):
    return f"""
    <div class="status-indicator">
        <span class="status-dot"></span>
        <span class="status-text">{text}</span>
    </div>
    """


def html_divider():
    return '<div class="section-divider"></div>'


with gr.Blocks() as demo:

    gr.HTML(html_header())

    with gr.Tabs():

        with gr.Tab("Image Understanding"):
            gr.HTML(html_tab_intro(
                SVG_IMAGE,
                "Image Understanding",
                "Upload an image and select a task category. Supports free-form queries, captioning, point localization, and object detection.",
                "Tokens are streamed in real time as the model generates.",
            ))
            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(type="pil", label="Upload Image", height=350)
                    category_select = gr.Dropdown(
                        choices=CATEGORIES, value="Query",
                        label="Task Category", interactive=True,
                    )
                    prompt_input = gr.Textbox(
                        placeholder="e.g., Count the total number of boats and describe the environment.",
                        label="Prompt", lines=3,
                    )
                    img_btn = gr.Button("Process Image", variant="primary")
                    gr.HTML(html_divider())
                    gr.Examples(
                        examples=[["examples/1.jpg"], ["examples/2.JPG"]],
                        inputs=[image_input], label="Image Examples",
                    )

                with gr.Column(scale=2):
                    output_image = gr.Image(label="Output Image", height=330)
                    output_text = gr.Textbox(label="Text Output", lines=10, interactive=True)
                    gr.HTML(html_status_indicator(
                        "Streaming enabled -- tokens appear as they are generated."
                    ))

            category_select.change(
                fn=on_category_change, inputs=[category_select], outputs=[prompt_input],
            )
            img_btn.click(
                fn=process_inputs,
                inputs=[image_input, category_select, prompt_input],
                outputs=[output_image, output_text],
            )

        with gr.Tab("Video QA"):
            gr.HTML(html_tab_intro(
                SVG_VIDEO,
                "Video Question Answering",
                "Upload a video and ask any question about its content. The model samples key frames and reasons across them.",
                "The response is streamed token-by-token as it is generated.",
            ))
            with gr.Row():
                with gr.Column():
                    vid_qa_input = gr.Video(label="Upload Video", format="mp4", height=350)
                    vid_qa_prompt = gr.Textbox(
                        label="Prompt",
                        placeholder="e.g., What is happening in this video? Summarise the key events.",
                        lines=3,
                    )
                    vid_qa_btn = gr.Button("Analyze Video", variant="primary")
                    gr.HTML(html_divider())
                    gr.Examples(
                        examples=[["examples/1.mp4"], ["examples/2.mp4"]],
                        inputs=[vid_qa_input], label="Video Examples",
                    )

                with gr.Column(scale=2):
                    vid_qa_text = gr.Textbox(label="Response", lines=15, interactive=True)
                    gr.HTML(html_status_indicator(
                        "Streaming enabled -- tokens appear as they are generated."
                    ))

            vid_qa_btn.click(
                fn=process_video_qa,
                inputs=[vid_qa_input, vid_qa_prompt],
                outputs=[vid_qa_text],
            )

        with gr.Tab("Video Detection"):
            gr.HTML(html_tab_intro(
                SVG_DETECT,
                "Video Object Detection",
                "Specify what to detect. The model samples frames at your chosen rate, "
                "runs detection on each, then overlays bounding-box masks on the full video. (max_secs<=7)",
            ))
            with gr.Row():
                with gr.Column():
                    vid_det_input = gr.Video(label="Upload Video", format="mp4", height=300)
                    vid_det_prompt = gr.Textbox(
                        label="Detection Target",
                        placeholder="e.g., all cars, people walking, the ball",
                        lines=2,
                    )
                    gr.HTML(html_section_heading(SVG_SETTINGS, "Sampling Configuration"))
                    vid_det_fps_slider = gr.Slider(
                        minimum=0.1, maximum=FPS_SLIDER_MAX, value=1.0, step=0.1,
                        label="Sample FPS",
                        info="Frames per second to sample. Higher values yield more frames but run slower.",
                    )
                    vid_det_maxframes_slider = gr.Slider(
                        minimum=1, maximum=int(FRAMES_SLIDER_MAX), value=8, step=1,
                        label="Max Frames",
                        info="Upper cap on total frames sampled. Increase for thorough but slower analysis.",
                    )
                    det_sampling_info = gr.HTML(value=update_sampling_info(1.0, 8))
                    vid_det_btn = gr.Button("Detect in Video", variant="primary")
                    gr.HTML(html_divider())
                    gr.Examples(
                        examples=[["examples/1.mp4"], ["examples/2.mp4"]],
                        inputs=[vid_det_input], label="Video Examples",
                    )

                with gr.Column(scale=2):
                    vid_det_video = gr.Video(label="Annotated Video", height=300)
                    vid_det_gallery = gr.Gallery(label="Annotated Key-Frames", columns=4, height=250)
                    vid_det_json = gr.Textbox(label="Detection Results (JSON)", lines=8, interactive=True)

            vid_det_fps_slider.change(
                fn=update_sampling_info,
                inputs=[vid_det_fps_slider, vid_det_maxframes_slider],
                outputs=[det_sampling_info],
            )
            vid_det_maxframes_slider.change(
                fn=update_sampling_info,
                inputs=[vid_det_fps_slider, vid_det_maxframes_slider],
                outputs=[det_sampling_info],
            )
            vid_det_btn.click(
                fn=process_video_detection,
                inputs=[vid_det_input, vid_det_prompt, vid_det_fps_slider, vid_det_maxframes_slider],
                outputs=[vid_det_video, vid_det_gallery, vid_det_json],
            )

        with gr.Tab("Video Point Track"):
            gr.HTML(html_tab_intro(
                SVG_TRACK,
                "Video Point Tracking",
                "Specify what to track. The model locates 2D point coordinates on sampled "
                "frames and overlays tracking dots across the full video. (max_secs<=7)",
            ))
            with gr.Row():
                with gr.Column():
                    vid_trk_input = gr.Video(label="Upload Video", format="mp4", height=300)
                    vid_trk_prompt = gr.Textbox(
                        label="Tracking Target",
                        placeholder="e.g., the football, the runner's head, the cat",
                        lines=2,
                    )
                    gr.HTML(html_section_heading(SVG_SETTINGS, "Tracking Configuration"))
                    vid_trk_fps_slider = gr.Slider(
                        minimum=0.1, maximum=FPS_SLIDER_MAX, value=1.0, step=0.1,
                        label="Sample FPS",
                        info="Frames per second to sample for tracking. Higher values yield smoother tracking.",
                    )
                    vid_trk_maxframes_slider = gr.Slider(
                        minimum=1, maximum=int(FRAMES_SLIDER_MAX), value=8, step=1,
                        label="Max Frames",
                        info="Upper cap on total frames tracked. Increase for more thorough tracking.",
                    )
                    trk_sampling_info = gr.HTML(value=update_tracking_info(1.0, 8))
                    vid_trk_btn = gr.Button("Track in Video", variant="primary")
                    gr.HTML(html_divider())
                    gr.Examples(
                        examples=[["examples/1.mp4"], ["examples/2.mp4"]],
                        inputs=[vid_trk_input], label="Video Examples",
                    )

                with gr.Column(scale=2):
                    vid_trk_video = gr.Video(label="Tracked Video", height=300)
                    vid_trk_gallery = gr.Gallery(label="Tracked Key-Frames", columns=4, height=250)
                    vid_trk_json = gr.Textbox(label="Tracking Results (JSON)", lines=8, interactive=True)

            vid_trk_fps_slider.change(
                fn=update_tracking_info,
                inputs=[vid_trk_fps_slider, vid_trk_maxframes_slider],
                outputs=[trk_sampling_info],
            )
            vid_trk_maxframes_slider.change(
                fn=update_tracking_info,
                inputs=[vid_trk_fps_slider, vid_trk_maxframes_slider],
                outputs=[trk_sampling_info],
            )
            vid_trk_btn.click(
                fn=process_video_point_tracking,
                inputs=[vid_trk_input, vid_trk_prompt, vid_trk_fps_slider, vid_trk_maxframes_slider],
                outputs=[vid_trk_video, vid_trk_gallery, vid_trk_json],
            )


if __name__ == "__main__":
    demo.launch(css=css, theme=steel_blue_theme, show_error=True, ssr_mode=False)