"""Visual theme. Fonts are the OFL files shipped with the app, with a system fallback."""

from __future__ import annotations

import base64
from functools import lru_cache
from pathlib import Path

_FONT_DIR = Path(__file__).resolve().parents[1] / "static" / "fonts"


def _face(filename: str, family: str) -> str:
    path = _FONT_DIR / filename
    if not path.is_file():
        return ""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"""
    @font-face {{
      font-family: "{family}";
      src: url("data:font/ttf;base64,{encoded}") format("truetype");
      font-weight: 400 800;
      font-style: normal;
      font-display: swap;
    }}
    """


@lru_cache(maxsize=1)
def css() -> str:
    bricolage = _face("BricolageGrotesque.ttf", "Bricolage Grotesque")
    instrument = _face("InstrumentSans.ttf", "Instrument Sans")
    return f"""
    <style>
    {bricolage}
    {instrument}
    html, body, [class*="css"], .stApp, .stMarkdown, button, input, textarea, select {{
      font-family: "Instrument Sans", "Segoe UI", sans-serif;
      color: #101831;
    }}
    h1, h2, h3, .wordmark, .readouts strong {{
      font-family: "Bricolage Grotesque", "Segoe UI", sans-serif;
      letter-spacing: -0.03em;
      color: #101831;
    }}
    .stApp {{ background: #F3F6FD; }}
    [data-testid="stAppViewContainer"] {{ background: #F3F6FD; }}
    [data-testid="stHeader"] {{ background: transparent; }}
    [data-testid="stToolbar"] {{ display: none; }}
    footer {{ visibility: hidden; }}
    .block-container {{ padding-top: 1.1rem; max-width: 1240px; }}
    .wordmark {{
      font-size: 1.35rem;
      font-weight: 700;
      margin: 0.35rem 0 0;
    }}
    .eyebrow {{
      color: #5C6B8A;
      font-size: 0.78rem;
      letter-spacing: 0.04em;
      text-transform: none;
      margin: 0;
    }}
    .synthetic {{
      background: #FFF6E8;
      border: 1px solid #F0D7A8;
      color: #6A4B12;
      border-radius: 12px;
      padding: 0.55rem 0.8rem;
      margin: 0.2rem 0 0.8rem;
      font-size: 0.92rem;
    }}
    .readouts {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
      margin: 0.4rem 0 0.2rem;
    }}
    .readouts div {{
      background: #FFFFFF;
      border: 1px solid #D9E1F2;
      border-radius: 14px;
      padding: 10px 12px 8px;
    }}
    .readouts span {{
      display: block;
      color: #5C6B8A;
      font-size: 0.82rem;
    }}
    .readouts strong {{
      display: block;
      color: #0B8F78;
      font-size: 1.7rem;
      font-weight: 680;
      font-variant-numeric: tabular-nums;
    }}
    .readouts em {{
      display: block;
      color: #5C6B8A;
      font-style: normal;
      font-size: 0.75rem;
    }}
    .mint-marker, p:has(.mint-marker) {{ display: none; }}
    [data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .mint-marker) {{
      background: #E6F7EF;
      border: 1px solid #B7E4D0 !important;
      border-radius: 16px;
      padding: 0.35rem 0.85rem 0.7rem;
    }}
    button:focus-visible, input:focus-visible, textarea:focus-visible, select:focus-visible,
    [data-baseweb="select"] :focus-visible {{
      outline: 2px solid #1243FF !important;
      outline-offset: 2px !important;
    }}
    .stButton > button {{
      border-radius: 999px;
      border: 1px solid #D5DCEC;
    }}
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stCaptionContainer"] p {{
      color: #31405F;
    }}
    button p, button span {{
      color: inherit !important;
    }}
    @media (max-width: 860px) {{
      .readouts {{ grid-template-columns: 1fr; }}
      [data-testid="stHorizontalBlock"] {{ flex-wrap: wrap; }}
      [data-testid="stHorizontalBlock"] > [data-testid="column"] {{
        width: 100% !important;
        flex: 1 1 100% !important;
        min-width: 100% !important;
      }}
    }}
    </style>
    """
