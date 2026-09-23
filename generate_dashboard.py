# =============================================================================
# Dashboard Screenshot Generator
# Creates screenshots/Dashboard_or_Project_Result.png
# =============================================================================
import os
from PIL import Image, ImageDraw, ImageFont

os.makedirs("screenshots", exist_ok=True)

# ── Canvas dimensions ────────────────────────────────────────────────────────
W, H = 1600, 1040
NAVY    = (26,  58,  92)
BLUE    = (45, 110, 168)
LTBLUE  = (220, 235, 250)
BG      = (245, 247, 251)
WHITE   = (255, 255, 255)
DGREY   = (68,  68,  68)
MGREY   = (130, 130, 140)
GREEN   = (34, 139,  80)
GOLD    = (210, 160,  20)

canvas = Image.new("RGB", (W, H), BG)
draw   = ImageDraw.Draw(canvas)

# ── Font loader ──────────────────────────────────────────────────────────────
def font(size, bold=False):
    candidates_bold   = ["arialbd.ttf","Arial Bold.ttf","DejaVuSans-Bold.ttf",
                         "C:/Windows/Fonts/arialbd.ttf","C:/Windows/Fonts/calibrib.ttf"]
    candidates_regular= ["arial.ttf","Arial.ttf","DejaVuSans.ttf",
                         "C:/Windows/Fonts/arial.ttf","C:/Windows/Fonts/calibri.ttf"]
    for path in (candidates_bold if bold else candidates_regular):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()

F_TITLE   = font(38, bold=True)
F_SUBTITLE= font(18, bold=False)
F_CARD_L  = font(32, bold=True)
F_CARD_S  = font(13, bold=False)
F_SECT    = font(15, bold=True)
F_BADGE   = font(12, bold=True)
F_SMALL   = font(11, bold=False)

# ── Helper: rounded rectangle ────────────────────────────────────────────────
def rrect(draw, xy, fill, radius=12, outline=None, outline_width=1):
    x0, y0, x1, y1 = xy
    draw.rounded_rectangle([x0, y0, x1, y1], radius=radius, fill=fill,
                           outline=outline, width=outline_width)

# ── Helper: paste plot ───────────────────────────────────────────────────────
def paste_plot(path, box, canvas):
    """Fit plot image into box (x0,y0,x1,y1) with white background."""
    bw = box[2] - box[0]
    bh = box[3] - box[1]
    try:
        img = Image.open(path).convert("RGB")
        img.thumbnail((bw, bh), Image.LANCZOS)
        tw, th = img.size
        ox = box[0] + (bw - tw) // 2
        oy = box[1] + (bh - th) // 2
        rrect(draw, box, WHITE, radius=10, outline=(210,215,225), outline_width=1)
        canvas.paste(img, (ox, oy))
    except Exception:
        rrect(draw, box, (230,230,230), radius=10)

# =============================================================================
# HEADER BAR
# =============================================================================
draw.rectangle([0, 0, W, 74], fill=NAVY)
# accent stripe
draw.rectangle([0, 0, 6, 74], fill=BLUE)

draw.text((24, 12), "House Price Prediction", font=F_TITLE, fill=WHITE)
draw.text((26, 52), "Machine Learning Project  •  Dataset: 50,000 rows, 19 features  •  Python | scikit-learn | pandas | seaborn",
          font=F_SUBTITLE, fill=LTBLUE)

# badge — FINAL MODEL
bx0, bx1 = W - 260, W - 18
draw.rounded_rectangle([bx0, 14, bx1, 58], radius=8, fill=BLUE)
draw.text((bx0 + 14, 18), "FINAL MODEL", font=F_BADGE, fill=LTBLUE)
draw.text((bx0 + 14, 34), "Linear Regression", font=font(16, bold=True), fill=WHITE)

# =============================================================================
# METRIC CARDS  (row just below header)
# =============================================================================
metrics = [
    ("R² Score",       "0.9981",  "(99.81% variance explained)", GREEN),
    ("RMSE",           "19,941",  "Avg. error in price units",    BLUE),
    ("MAE",            "15,954",  "Mean absolute error",          GOLD),
    ("Test Samples",   "10,000",  "20% hold-out test set",        (120,80,180)),
    ("Train Samples",  "40,000",  "80% training set",             (60,130,130)),
    ("Features",       "18",      "After OHE: 22 features",       (170,70,70)),
]

CARD_TOP = 84
CARD_H   = 90
cpad     = 10
cw       = (W - 2*cpad - 5*cpad) // 6 - 2

for i, (label, value, sub, colour) in enumerate(metrics):
    x0 = cpad + i * (cw + cpad + 2)
    x1 = x0 + cw
    rrect(draw, [x0, CARD_TOP, x1, CARD_TOP + CARD_H], WHITE, radius=10,
          outline=(210,215,225), outline_width=1)
    # colour accent left border
    draw.rounded_rectangle([x0, CARD_TOP, x0+5, CARD_TOP+CARD_H], radius=4, fill=colour)
    draw.text((x0 + 16, CARD_TOP + 8),  label, font=F_CARD_S, fill=MGREY)
    draw.text((x0 + 16, CARD_TOP + 26), value, font=F_CARD_L, fill=colour)
    draw.text((x0 + 16, CARD_TOP + 66), sub,   font=F_SMALL,  fill=MGREY)

# =============================================================================
# PLOT GRID  (2 rows × 3 cols + 1 wide col)
# =============================================================================
ROW1_TOP  = CARD_TOP + CARD_H + 14
ROW1_BOT  = ROW1_TOP + 295

ROW2_TOP  = ROW1_BOT + 10
ROW2_BOT  = ROW2_TOP + 295

PAD   = 10
PW    = (W - 4*PAD) // 3          # plot width  (3 columns)
PW2   = (W - 3*PAD) // 2 + 10     # wide plot (2/3 width for actual-vs-predicted)
PW_SM = (W - 3*PAD) // 2 - 10 - PW2 + W - 3*PAD - PW2 - PAD  # remaining

# Row 1 ──────────────────────────────────────────────────────────────────────
# col 0: price distribution
paste_plot("plots/price_distribution.png",
           [PAD, ROW1_TOP, PAD+PW, ROW1_BOT], canvas)
draw.text((PAD+8, ROW1_TOP+6), "Price Distribution", font=F_SECT, fill=NAVY)

# col 1: area vs price
paste_plot("plots/area_vs_price.png",
           [PAD*2+PW, ROW1_TOP, PAD*2+PW*2, ROW1_BOT], canvas)
draw.text((PAD*2+PW+8, ROW1_TOP+6), "Area vs Price", font=F_SECT, fill=NAVY)

# col 2: correlation heatmap
paste_plot("plots/correlation_heatmap.png",
           [PAD*3+PW*2, ROW1_TOP, W-PAD, ROW1_BOT], canvas)
draw.text((PAD*3+PW*2+8, ROW1_TOP+6), "Correlation Heatmap", font=F_SECT, fill=NAVY)

# Row 2 ──────────────────────────────────────────────────────────────────────
# col 0: model comparison
paste_plot("plots/model_comparison.png",
           [PAD, ROW2_TOP, PAD+PW, ROW2_BOT], canvas)
draw.text((PAD+8, ROW2_TOP+6), "Model Comparison", font=F_SECT, fill=NAVY)

# col 1: feature importance
paste_plot("plots/feature_importance.png",
           [PAD*2+PW, ROW2_TOP, PAD*2+PW*2, ROW2_BOT], canvas)
draw.text((PAD*2+PW+8, ROW2_TOP+6), "Feature Importance", font=F_SECT, fill=NAVY)

# col 2: actual vs predicted
paste_plot("plots/actual_vs_predicted.png",
           [PAD*3+PW*2, ROW2_TOP, W-PAD, ROW2_BOT], canvas)
draw.text((PAD*3+PW*2+8, ROW2_TOP+6), "Actual vs Predicted", font=F_SECT, fill=NAVY)

# =============================================================================
# FOOTER
# =============================================================================
FOOTER_TOP = ROW2_BOT + 10
draw.rectangle([0, FOOTER_TOP, W, H], fill=NAVY)
draw.rectangle([0, FOOTER_TOP, W, FOOTER_TOP+2], fill=BLUE)

footer_items = [
    ("Dataset",         "house_price.csv — 50,000 rows"),
    ("Target",          "price  (continuous regression)"),
    ("Best Model",      "Linear Regression"),
    ("Sample Prediction", "Area=2000, Medium, Mid income  →  818,999.68"),
    ("Top Feature",     "area  (r = 0.991 with price)"),
]
fx = 24
for label, val in footer_items:
    draw.text((fx,   FOOTER_TOP + 10), label + ":", font=font(11, bold=True), fill=LTBLUE)
    draw.text((fx,   FOOTER_TOP + 26), val,          font=font(11),           fill=WHITE)
    fx += (W - 48) // len(footer_items)

# model badge bottom-right
draw.text((W - 240, FOOTER_TOP + 12),
          "github.com/House-Price-Prediction",
          font=font(11), fill=LTBLUE)

# =============================================================================
# SAVE
# =============================================================================
out = "screenshots/Dashboard_or_Project_Result.png"
canvas.save(out, "PNG", optimize=True)
print(f"Saved: {out}  ({W}x{H}px)")
