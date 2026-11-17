"""Deterministically redraw the screenshot-style fixture images.

Run from anywhere: ``python3 fixtures/make_images.py`` rewrites
``fixtures/images/*.png`` byte-for-byte.  Every image is generated with a fixed
per-image seed and only Pillow's deterministic drawing primitives, so the
fixtures are reproducible without network access.  The sizes are chosen so the
official Qwen2-VL ``smart_resize`` (factor 32, shortest edge 65536) keeps or
derives distinct ``(t, h, w)`` patch grids.
"""
from __future__ import annotations

import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

IMAGES = Path(__file__).resolve().parent / "images"

INK = (32, 38, 46)
MUTED = (96, 106, 118)
RULE = (206, 212, 220)
PAPER = (250, 251, 252)
BAR = (28, 92, 168)
OK = (22, 140, 92)
WARN = (214, 132, 24)
BAD = (196, 54, 54)


def _font(size: int):
    return ImageFont.load_default(size=size)


def _text(draw, x, y, text, size=15, fill=INK, bold_box=None):
    font = _font(size)
    draw.text((x, y), text, font=font, fill=fill)
    if bold_box is not None:
        w = draw.textlength(text, font=font)
        draw.line((x, y + size + 4, x + w, y + size + 4), fill=bold_box, width=1)


def _lines(draw, x, y, width, count, rng, gap=18, height=7, fill=(178, 186, 196)):
    for i in range(count):
        w = int(width * (0.62 + 0.34 * rng.random()))
        draw.rectangle((x, y + i * gap, x + w, y + i * gap + height), fill=fill)
        if i == count - 1:
            return y + i * gap + height + 10


def _button(draw, x, y, w, h, label, fill=BAR, fg=(255, 255, 255), size=15):
    draw.rounded_rectangle((x, y, x + w, y + h), radius=8, fill=fill)
    font = _font(size)
    tw = draw.textlength(label, font=font)
    draw.text((x + (w - tw) / 2, y + (h - size) / 2 - 1), label, font=font, fill=fg)


def invoice(path, size, seed, title, accent=BAR):
    w, h = size
    rng = random.Random(seed)
    im = Image.new("RGB", size, PAPER)
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, w, 64), fill=accent)
    _text(d, 28, 20, title, size=22, fill=(255, 255, 255))
    _text(d, w - 150, 24, "INVOICE", size=16, fill=(235, 242, 252))
    d.rectangle((28, 92, 28 + int(w * 0.34), 156), outline=RULE, width=2)
    _text(d, 42, 102, "ACME SUPPLY CO.", size=16)
    _text(d, 42, 126, "18 Harbor Street, Bay City", size=12, fill=MUTED)
    d.rectangle((w - int(w * 0.34) - 28, 92, w - 28, 156), outline=RULE, width=2)
    _text(d, w - int(w * 0.34) - 14, 102, "Invoice #1042", size=16)
    _text(d, w - int(w * 0.34) - 14, 126, "Due 2026-12-01", size=12, fill=MUTED)
    top = 190
    d.rectangle((28, top, w - 28, top + 34), fill=(234, 238, 244))
    _text(d, 42, top + 8, "Description", size=13, fill=MUTED)
    _text(d, int(w * 0.52), top + 8, "Qty", size=13, fill=MUTED)
    _text(d, int(w * 0.66), top + 8, "Unit price", size=13, fill=MUTED)
    _text(d, w - 130, top + 8, "Amount", size=13, fill=MUTED)
    rows = 4 if h >= 480 else 3
    y = top + 34
    step = 38
    for i in range(rows):
        d.line((28, y, w - 28, y), fill=RULE, width=1)
        _text(d, 42, y + 10, f"Line item {i + 1}: hardware bundle", size=13)
        _text(d, int(w * 0.52), y + 10, str(1 + i), size=13)
        _text(d, int(w * 0.66), y + 10, f"{19 + 7 * i}.00", size=13)
        _text(d, w - 130, y + 10, f"{19 + 7 * i}.00", size=13)
        y += step
    d.line((28, y, w - 28, y), fill=RULE, width=1)
    _text(d, w - 200, y + 14, "Total due", size=16)
    _text(d, w - 130, y + 14, "EUR 412.00", size=18, fill=BAD)
    _button(d, 28, h - 66, 150, 40, "Pay invoice", fill=OK)
    _button(d, 192, h - 66, 130, 40, "Download", fill=(238, 241, 246), fg=INK)
    _text(d, 340, h - 56, "Questions? billing@acme.example", size=11, fill=MUTED)
    im.save(path)


def dashboard(path, size, seed):
    w, h = size
    rng = random.Random(seed)
    im = Image.new("RGB", size, (244, 247, 250))
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, 210, h), fill=(24, 34, 48))
    _text(d, 22, 22, "CONSOLE", size=18, fill=(255, 255, 255))
    for i, label in enumerate(("Overview", "Jobs", "Alerts", "Settings")):
        fill = (54, 72, 96) if i == 1 else (24, 34, 48)
        d.rectangle((12, 70 + i * 44, 198, 104 + i * 44), fill=fill)
        _text(d, 26, 81 + i * 44, label, size=14, fill=(224, 232, 242) if i == 1 else (150, 162, 178))
    _text(d, 240, 30, "Job queue", size=24)
    _text(d, 240, 62, "4 running, 2 waiting, 3 failed", size=13, fill=MUTED)
    cards = (("Running", "4", OK), ("Waiting", "2", WARN), ("Failed", "3", BAD))
    for i, (label, value, color) in enumerate(cards):
        x = 240 + i * (int((w - 260) / 3) + 12)
        cw = int((w - 260) / 3)
        d.rounded_rectangle((x, 92, x + cw, 172), radius=10, fill=(255, 255, 255), outline=RULE)
        _text(d, x + 18, 106, label, size=12, fill=MUTED)
        _text(d, x + 18, 126, value, size=28, fill=color)
    top = 196
    d.rectangle((240, top, w - 20, top + 32), fill=(232, 237, 243))
    heads = ("Job", "Owner", "Elapsed", "State")
    xs = (254, int(w * 0.46), int(w * 0.64), w - 130)
    for x, label in zip(xs, heads):
        _text(d, x, top + 8, label, size=12, fill=MUTED)
    y = top + 32
    for i in range(5):
        d.line((240, y, w - 20, y), fill=RULE)
        _text(d, 254, y + 12, f"job-{8800 + i}", size=13)
        _text(d, int(w * 0.46), y + 12, ("ci", "ana", "bo", "dex", "eve")[i], size=13)
        _text(d, int(w * 0.64), y + 12, f"{3 + i}m", size=13)
        _text(d, w - 130, y + 12, ("running", "failed", "running", "waiting", "failed")[i],
              size=13, fill=(OK if i % 3 == 0 else BAD if i % 3 == 1 else WARN))
        y += 34
    d.line((240, y, w - 20, y), fill=RULE)
    _button(d, 240, h - 62, 160, 40, "Retry failed", fill=BAD)
    _button(d, 414, h - 62, 140, 40, "Refresh", fill=(236, 240, 246), fg=INK)
    _lines(d, w - 330, 92, 290, 4, rng, gap=16)
    im.save(path)


def table(path, size, seed):
    w, h = size
    rng = random.Random(seed)
    im = Image.new("RGB", size, (255, 255, 255))
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, w, 58), fill=(38, 48, 60))
    _text(d, 24, 18, "Customers", size=20, fill=(255, 255, 255))
    search = (232, 236, 242)
    d.rounded_rectangle((int(w * 0.46), 14, w - 230, 46), radius=6, fill=search)
    _text(d, int(w * 0.46) + 14, 23, "Search by name or email", size=12, fill=(120, 130, 144))
    _button(d, w - 210, 13, 100, 34, "Export", fill=BAR, size=13)
    _button(d, w - 102, 13, 78, 34, "New", fill=OK, size=13)
    y0 = 82
    cols = (24, 90, int(w * 0.34), int(w * 0.62), int(w * 0.80))
    heads = ("#", "Name", "Email", "Plan", "Status")
    d.rectangle((16, y0, w - 16, y0 + 32), fill=(240, 243, 247))
    for x, label in zip(cols, heads):
        _text(d, x, y0 + 8, label, size=12, fill=MUTED)
    y = y0 + 32
    names = ("Ada Lovelace", "Grace Hopper", "Alan Turing", "Edsger Dijkstra", "Barbara Liskov",
             "Ken Thompson", "Frances Allen", "Donald Knuth")
    plans = ("team", "pro", "free", "team", "pro", "free", "pro", "team")
    states = ("active", "active", "trial", "active", "paused", "active", "trial", "active")
    rows = 8 if h >= 700 else 6
    for i in range(rows):
        if i % 2 == 0:
            d.rectangle((16, y, w - 16, y + 40), fill=(251, 252, 253))
        d.line((16, y, w - 16, y), fill=RULE)
        _text(d, cols[0], y + 12, str(1001 + i), size=12)
        _text(d, cols[1], y + 12, names[i], size=13)
        _text(d, cols[2], y + 12, f"user{i + 1}@example.com", size=12, fill=MUTED)
        _text(d, cols[3], y + 12, plans[i], size=12)
        _text(d, cols[4], y + 12, states[i], size=12,
              fill=(OK if states[i] == "active" else WARN if states[i] == "trial" else MUTED))
        y += 40
    d.line((16, y, w - 16, y), fill=RULE)
    _text(d, 24, y + 14, f"Showing {rows} of 128 customers", size=12, fill=MUTED)
    for i, label in enumerate(("Prev", "1", "2", "3", "Next")):
        _button(d, w - 330 + i * 58, y + 8, 50, 30, label,
                fill=BAR if label == "1" else (238, 241, 246), fg=(255, 255, 255) if label == "1" else INK, size=12)
    im.save(path)


def app(path, size, seed):
    w, h = size
    rng = random.Random(seed)
    im = Image.new("RGB", size, (247, 248, 250))
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, w, 72), fill=(255, 255, 255))
    d.line((0, 72, w, 72), fill=RULE)
    _text(d, 26, 22, "TravelDesk", size=22, fill=(18, 60, 120))
    for i, label in enumerate(("Trips", "Book", "Expenses", "Profile")):
        _text(d, 200 + i * 118, 30, label, size=14, fill=MUTED if i else INK)
    d.rounded_rectangle((w - 190, 18, w - 26, 54), radius=18, fill=(238, 242, 248))
    _text(d, w - 176, 27, "Signed in as Ana", size=12, fill=(90, 100, 114))
    d.rounded_rectangle((26, 96, w - 26, 180), radius=10, fill=(230, 240, 253))
    _text(d, 46, 112, "Trip to Lisbon", size=20, fill=(20, 62, 120))
    _text(d, 46, 142, "2026-12-03 to 2026-12-09  ·  2 travellers  ·  Confirmed", size=13, fill=(58, 82, 116))
    _button(d, w - 220, 120, 160, 40, "Manage booking", fill=BAR)
    y = 204
    left_w = int(w * 0.56)
    d.rounded_rectangle((26, y, 26 + left_w, h - 92), radius=10, fill=(255, 255, 255), outline=RULE)
    _text(d, 46, y + 18, "Itinerary", size=16)
    rows = ("09:10  LIS arrival, terminal 1", "11:30  Hotel check-in", "14:00  Client workshop",
            "19:30  Dinner reservation", "Day 2  Site visit")
    for i, row in enumerate(rows):
        yy = y + 54 + i * 40
        d.ellipse((48, yy + 4, 60, yy + 16), fill=BAR if i < 2 else RULE)
        _text(d, 74, yy, row, size=13, fill=INK if i < 2 else MUTED)
    rx = 26 + left_w + 22
    d.rounded_rectangle((rx, y, w - 26, y + 200), radius=10, fill=(255, 255, 255), outline=RULE)
    _text(d, rx + 18, y + 16, "Actions", size=16)
    for i, label in enumerate(("Add expense", "Message agent", "Cancel trip")):
        _button(d, rx + 18, y + 52 + i * 48, w - rx - 62, 36, label,
                fill=(238, 242, 248) if i != 2 else (250, 232, 232), fg=INK if i != 2 else BAD, size=13)
    d.rounded_rectangle((rx, y + 218, w - 26, h - 92), radius=10, fill=(255, 252, 240), outline=(238, 224, 188))
    _text(d, rx + 18, y + 234, "Agent note", size=14, fill=(132, 98, 20))
    _lines(d, rx + 18, y + 262, w - rx - 70, 4, rng, gap=16, fill=(212, 200, 172))
    _text(d, 26, h - 66, "Support: help@traveldesk.example  ·  status: all systems normal", size=11, fill=MUTED)
    im.save(path)


def main():
    IMAGES.mkdir(parents=True, exist_ok=True)
    invoice(IMAGES / "invoice-512.png", (512, 512), 101, "ACME Billing")
    invoice(IMAGES / "invoice-640x480.png", (640, 480), 102, "Northwind Invoice", accent=(22, 120, 96))
    table(IMAGES / "customers-1024x768.png", (1024, 768), 103)
    dashboard(IMAGES / "job-queue-896x512.png", (896, 512), 104)
    app(IMAGES / "travel-app-1280x704.png", (1280, 704), 105)
    for path in sorted(IMAGES.glob("*.png")):
        with Image.open(path) as im:
            print(f"make_images: {path.name} {im.size[0]}x{im.size[1]} {path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
