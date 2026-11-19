import os, json, asyncio, datetime
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import edge_tts
from google import genai
from google.genai import types
from moviepy.editor import VideoClip, AudioFileClip, concatenate_videoclips

W, H, FPS = 1080, 1920, 24
VOICE = "en-IN-PrabhatNeural"

THEMES = {
    0: "Looks: skin, hair, eyebrows, grooming",
    1: "Body: one muscle group or exercise explained simply",
    2: "Outfit and personal style",
    3: "Focus and avoiding distractions",
    4: "Psychology of habits and confidence",
    5: "Money, earning skills and career growth",
    6: "Philosophy and mindset",
}

PALETTE = [
    ((10, 10, 40), (60, 20, 120)),
    ((5, 30, 30), (10, 90, 90)),
    ((40, 10, 10), (120, 40, 20)),
    ((10, 10, 10), (50, 50, 70)),
]

FONT_PATHS = [
    "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]
FONT_PATH = next(p for p in FONT_PATHS if os.path.exists(p))


def get_script():
    theme = THEMES[datetime.date.today().weekday()]
    prompt = (
        "You write scripts for a 45-second self-improvement YouTube Short.\n"
        f"Today's theme: {theme}.\n"
        "Rules: pick one specific, useful topic. Use simple spoken English and "
        "short sentences. Give only safe, evidence-based advice. No miracle claims, "
        "no guaranteed results or income, no extreme diets or dangerous hacks. "
        "Include one thing NOT to do.\n"
        'Return ONLY JSON like {"title": "...", "lines": ["...", "..."]}.\n'
        "lines: 7 to 9 items, each at most 14 words. The first line is a strong "
        "hook. The last line is a short call to follow the channel."
    )
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    last_err = None
    for model in ["gemini-3.5-flash-lite"]:
        try:
            r = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                ),
            )
            text = r.text.strip().replace("```json", "").replace("```", "")
            data = json.loads(text)
            assert data["lines"]
            return data
        except Exception as e:
            last_err = e
            print("Model failed:", model, e)
    raise last_err


def bg(i):
    a, b = (np.array(c) for c in PALETTE[i % len(PALETTE)])
    g = np.linspace(0, 1, H)[:, None, None]
    return (a + (b - a) * g).astype("uint8").repeat(W, axis=1)


def layout(words, font):
    sp = font.getlength(" ")
    rows, cur, w = [], [], 0
    for wd in words:
        l = font.getlength(wd)
        if cur and w + sp + l > W - 160:
            rows.append((cur, w))
            cur, w = [], 0
        w += (sp if cur else 0) + l
        cur.append((wd, l))
    if cur:
        rows.append((cur, w))
    pos = []
    y = (H - len(rows) * 130) // 2
    for row, rw in rows:
        x = (W - rw) / 2
        for wd, l in row:
            pos.append((wd, x, y))
            x += l + sp
        y += 130
    return pos


def make_clip(text, dur, idx):
    words = text.split()
    base = bg(idx)
    font = ImageFont.truetype(FONT_PATH, 90)
    pos = layout(words, font)

    def frame(t):
        n = max(1, min(len(words), int(len(words) * min(1.0, t / (dur * 0.75))) + 1))
        img = Image.fromarray(base.copy())
        d = ImageDraw.Draw(img)
        for k, (wd, x, y) in enumerate(pos[:n]):
            color = (255, 215, 0) if k == n - 1 else (255, 255, 255)
            d.text((x, y), wd, font=font, fill=color)
        return np.array(img)

    return VideoClip(frame, duration=dur)


async def tts(text, path):
    await edge_tts.Communicate(text, VOICE, rate="+5%").save(path)


os.makedirs("output", exist_ok=True)
os.makedirs("tmp", exist_ok=True)
data = get_script()
print("TITLE:", data.get("title"))
clips = []
for i, line in enumerate(data["lines"]):
    p = f"tmp/{i}.mp3"
    asyncio.run(tts(line, p))
    a = AudioFileClip(p)
    c = make_clip(line, a.duration + 0.25, i).set_audio(a)
    clips.append(c)
final = concatenate_videoclips(clips, method="compose")
final.write_videofile("output/short.mp4", fps=FPS, codec="libx264", audio_codec="aac")
