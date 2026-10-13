"""Render a finding + explanation as a shareable "ledger slip" PNG.

Portrait, readable at WhatsApp phone-screen size, using the same paper/ink
palette and fonts as the web UI (see CLAUDE.md's design system section).
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FONTS_DIR = Path(__file__).resolve().parent.parent / "static" / "fonts"
WIDTH, HEIGHT = 1080, 1500
MARGIN = 64

COLORS = {
    "paper": "#F6F1E7",
    "card": "#FBF8F1",
    "ink": "#1F1B16",
    "muted": "#6B6358",
    "hairline": "#D9D0BE",
    "drop": "#B3382C",
    "positive": "#3E6B4A",
    "highlight": "#E0A526",
}

CAUSE_COPY = {
    "price_increase": "Likely cause: price rose {change_pct:.0f}%",
    "freight_spike": "Likely cause: shipping cost rose {change_pct:.0f}%",
    "stockout": "Likely cause: products were out of stock",
    "price_decrease": "Likely cause: price dropped {change_pct:.0f}%",
    "freight_decrease": "Likely cause: shipping cost dropped {change_pct:.0f}%",
    "more_products": "Likely cause: more products were available",
    "unknown": "Cause: not sure -- nothing in the data explains this",
}


def _heading_font(size: int, weight: int = 600) -> ImageFont.FreeTypeFont:
    font = ImageFont.truetype(str(FONTS_DIR / "Fraunces.ttf"), size)
    font.set_variation_by_axes([9, weight, 0, 1])
    return font


def _body_font(size: int, telugu: bool = False) -> ImageFont.FreeTypeFont:
    # Instrument Sans has no Telugu glyphs -- Pillow silently renders them as
    # empty boxes instead of falling back, unlike a browser's font-family
    # list, so the explanation text needs an explicit Telugu-capable font.
    name = "NotoSansTelugu.ttf" if telugu else "InstrumentSans.ttf"
    return ImageFont.truetype(str(FONTS_DIR / name), size)


def _mono_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "IBMPlexMono-Bold.ttf" if bold else "IBMPlexMono-Regular.ttf"
    return ImageFont.truetype(str(FONTS_DIR / name), size)


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    words = text.split()
    lines, current = [], ""
    for word in words:
        trial = f"{current} {word}".strip()
        if draw.textlength(trial, font=font) <= max_width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _draw_content(draw: ImageDraw.ImageDraw, finding: dict, explanation_text: str, shop_name: str, backend_footer: str, language: str = "en") -> int:
    """Draws everything and returns the final y (content bottom)."""
    x = MARGIN + 48
    y = MARGIN + 48
    content_width = WIDTH - 2 * (MARGIN + 48)
    line_x2 = WIDTH - MARGIN - 48

    draw.text((x, y), "SalesWhy", font=_heading_font(56), fill=COLORS["ink"])
    y += 76
    draw.text((x, y), shop_name, font=_body_font(28), fill=COLORS["muted"])
    y += 48

    draw.line((x, y, line_x2, y), fill=COLORS["hairline"], width=1)
    y += 40

    date_range = f"{finding['window_start']} to {finding['window_end']}"
    draw.text((x, y), date_range, font=_mono_font(26), fill=COLORS["muted"])
    y += 56

    is_spike = finding.get("direction") == "spike"
    if finding["cause"] == "unknown":
        headline_color = COLORS["muted"]
    else:
        headline_color = COLORS["positive"] if is_spike else COLORS["drop"]
    shortfall_text = f"Sales {finding['shortfall_pct']:.0f}% {'above' if is_spike else 'below'} normal"
    draw.text((x, y), shortfall_text, font=_heading_font(44, weight=500), fill=headline_color)
    y += 64

    cause_text = CAUSE_COPY.get(finding["cause"], finding["cause"]).format(change_pct=abs(finding.get("change_pct", 0)))
    draw.text((x, y), cause_text, font=_mono_font(28), fill=COLORS["ink"])
    y += 64

    draw.line((x, y, line_x2, y), fill=COLORS["hairline"], width=1)
    y += 48

    is_telugu = language == "te"
    explanation_font = _body_font(28, telugu=is_telugu)
    line_height = 48 if is_telugu else 40  # Telugu vowel marks/conjuncts need more vertical room

    draw.text((x, y), "What this means", font=_heading_font(32, weight=500), fill=COLORS["ink"])
    y += 52
    for line in _wrap_text(draw, explanation_text, explanation_font, content_width):
        draw.text((x, y), line, font=explanation_font, fill=COLORS["ink"])
        y += line_height

    y += 32
    draw.line((x, y, line_x2, y), fill=COLORS["hairline"], width=1)
    y += 32
    draw.text((x, y), backend_footer, font=_mono_font(20), fill=COLORS["muted"])
    y += 32
    return y


def render_slip(finding: dict, explanation_text: str, shop_name: str, out_path: str, backend_footer: str, language: str = "en") -> str:
    # Pass 1: lay out on a scratch canvas just to measure how tall the real
    # content is (text height isn't known until wrapped/drawn).
    scratch = Image.new("RGB", (WIDTH, HEIGHT * 3), COLORS["paper"])
    content_bottom = _draw_content(ImageDraw.Draw(scratch), finding, explanation_text, shop_name, backend_footer, language)
    final_height = content_bottom + MARGIN + 48

    # Pass 2: real canvas, card background sized to the measured content.
    img = Image.new("RGB", (WIDTH, final_height), COLORS["paper"])
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle(
        (MARGIN, MARGIN, WIDTH - MARGIN, final_height - MARGIN),
        radius=6, fill=COLORS["card"], outline=COLORS["hairline"], width=1,
    )
    _draw_content(draw, finding, explanation_text, shop_name, backend_footer, language)

    img.save(out_path, "PNG")
    return out_path


if __name__ == "__main__":
    sample_finding = {
        "window_start": "2026-09-04",
        "window_end": "2026-09-05",
        "shortfall_pct": 54.8,
        "cause": "stockout",
        "change_pct": 40.0,
        "confidence": 0.7,
    }
    sample_text = (
        "During the last couple of days, sales were lower than expected because several "
        "products were out of stock. This likely turned away customers who wanted to buy. "
        "Checking inventory levels during busy periods could help avoid this next time."
    )
    out = render_slip(
        sample_finding,
        sample_text,
        shop_name="Demo Candle Co.",
        out_path=str(Path(__file__).resolve().parent.parent / "notebooks" / "slip_sample.png"),
        backend_footer="TabPFN: local  ·  Gemma: local (gemma3:4b)",
    )
    print("wrote", out)
