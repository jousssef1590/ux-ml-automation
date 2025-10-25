import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# Paths
DB_PATH = Path("data/ux_feedback.db")
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

excel_path = REPORTS_DIR / f"feedback_report_{datetime.now():%Y%m%d}.xlsx"
pdf_path   = REPORTS_DIR / f"feedback_report_{datetime.now():%Y%m%d}.pdf"
plot_path  = REPORTS_DIR / "sentiment_plot.png"

# ---- Load data ----
import sqlite3
conn = sqlite3.connect(DB_PATH)
df = pd.read_sql("SELECT * FROM user_feedback_enriched", conn)
conn.close()

# ---- Excel summary ----
summary = (
    df.groupby("sentiment_label")
    .agg({"sentiment_confidence": "mean", "topic_id": "count"})
    .rename(columns={"sentiment_confidence": "avg_conf", "topic_id": "count"})
    .reset_index()
)
with pd.ExcelWriter(excel_path) as writer:
    df.to_excel(writer, sheet_name="Data", index=False)
    summary.to_excel(writer, sheet_name="Summary", index=False)
print(f"💾 Excel saved to {excel_path}")

# ---- Sentiment chart ----
plt.figure(figsize=(5,3))
df["sentiment_label"].value_counts().plot(kind="bar", color="skyblue")
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment"); plt.ylabel("Count")
plt.tight_layout(); plt.savefig(plot_path); plt.close()

# ---- Simple PDF ----
c = canvas.Canvas(str(pdf_path), pagesize=A4)
c.setFont("Helvetica-Bold", 14)
c.drawString(72, 800, "User Feedback Report")
c.setFont("Helvetica", 11)
c.drawString(72, 780, f"Generated on: {datetime.now():%Y-%m-%d %H:%M}")
c.drawImage(str(plot_path), 72, 600, width=400, height=200)
c.drawString(72, 580, "Sentiment Summary:")
y = 560
for _, row in summary.iterrows():
    c.drawString(90, y, f"{row['sentiment_label']}: {int(row['count'])} (avg_conf={row['avg_conf']:.2f})")
    y -= 20
c.save()

print(f"📄 PDF saved to {pdf_path}")
print("✅ Reporting complete.")
