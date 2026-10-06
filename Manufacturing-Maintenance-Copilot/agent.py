"""Step 3 (used by model.py): maintenance agent.
Retrieves the best knowledge-base entry (TF-IDF RAG) and builds a fix plan
with steps, spare parts and urgency.

Try it:  python agent.py "pump vibration is high and shaking"
"""
import json, sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

KB = json.load(open("data.json"))["kb"]
_vec = TfidfVectorizer(stop_words="english")
_mat = _vec.fit_transform([k["title"] + " " + k["text"] for k in KB])


def retrieve(query, k=1):
    """Return the k most relevant KB entries for a free-text question."""
    scores = cosine_similarity(_vec.transform([query]), _mat)[0]
    return [KB[i] for i in scores.argsort()[::-1][:k]]


def suggest(machine, fault, confidence, health, rul_hours):
    """Fix plan for a machine given the model's output."""
    entry = next(k for k in KB if k["fault"] == fault)
    if fault == "normal":
        urgency, summary = "Routine", f"{machine} is running normally. Keep to the preventive schedule."
    else:
        urgency = "Immediate" if health < 60 else "This week"
        eta = f" Estimated {rul_hours} h before it needs to stop." if rul_hours else ""
        summary = f"{machine} shows signs of {entry['title'].lower()} ({confidence}% confidence).{eta}"
    return {"summary": summary, "steps": entry["steps"], "parts": entry["parts"], "urgency": urgency}


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "motor is hot"
    hit = retrieve(q)[0]
    print(hit["title"], "->", *hit["steps"], sep="\n  ")
