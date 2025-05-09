import time
import random
import requests
from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List, Dict, Any
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
import os
import re
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
import json

app = FastAPI()

# --- DB Setup ---
Base = declarative_base()
engine = create_engine('sqlite:///candidates.db', echo=False)
SessionLocal = sessionmaker(bind=engine)

class Candidate(Base):
    __tablename__ = 'candidates'
    id = Column(Integer, primary_key=True)
    session_id = Column(String, unique=True)
    name = Column(String)
    email = Column(String)
    chat_history = Column(Text)
    summary = Column(Text)

Base.metadata.create_all(engine)

OPENROUTER_API_KEY = "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"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# --- Hardcoded property info (do not reveal to candidate) ---
PROPERTY_INFO = {
    "location": "Banikhet, HP (10 km from Dalhousie)",
    "transport": [
        "Gaggal Airport (101 km, 3 hr, ₹2,500–3,000)",
        "Pathankot Rail (80 km, ₹3,500 taxi)",
        "NHPC Chowk (4 km, ₹200 taxi)"
    ],
    "rooms": [
        "4-Bed Female Dorm – ₹629/night",
        "4-Bed Mixed Dorm – ₹629/night",
        "8-Bed Mixed Dorm – ₹599/night",
        "Deluxe Private Room (2 pax) – ₹2,969/night"
    ],
    "facilities": [
        "Free Wi-Fi", "Pool", "Cafe", "Workstations", "Lockers", "Hot Water", "Bonfire", "Stargazing", "Valley View"
    ],
    "policies": [
        "18+ only; no children",
        "Groups > 8 not allowed; groups ≥ 3 may be split",
        "Non-veg & eggs banned; alcohol & drugs banned",
        "Payment: 100% advance",
        "Cancellation: ≥ 14 days → free refund; 7–13 days → 21% fee; < 7 days → no refund"
    ],
    "experiences": [
        "Kalatop Trek", "Dainkund Peak", "Chamera Lake", "Khajjiar Meadows", "Cafe Hopping"
    ]
}

ACKNOWLEDGMENTS = [
    "Thanks for this!",
    "This helps.",
    "Alright, appreciate it.",
    "Okay, got it.",
    "Cheers for the info!",
    "Alright, thanks!",
    "Okay, thanks for sharing.",
    "Cool, appreciate it."
]

def get_acknowledgment():
    return random.choice(ACKNOWLEDGMENTS)

OPENING_MESSAGE = "Hey! Planning a trip to Zostel Banikhet with a friend—what's the easiest way to get from Bangalore?"

PHASES = [
    "arrival",
    "booking",
    "policy",
    "closure"
]

FOLLOWUPS = {
    "arrival": [
        ("airport" , "Could you tell me about the nearest airport and how to get from there to Zostel?"),
        ("train", "What about the nearest train station and the best way from there?"),
        ("taxi", "And roughly how long does it take, and what's the taxi fare?")
    ],
    "booking": [
        ("room_types", "Could you share the room types and price per night for each?"),
        ("availability", "Are those rooms available for next weekend?"),
        ("pricing", "And what's the price for the 4-bed dorm and private room?")
    ],
    "policy": [
        ("cancellation", "What's the cancellation policy if I cancel 2 days before arrival?"),
        ("age", "Is there any age limit or child policy?"),
        ("group", "How many people can book together?")
    ],
    "closure": [
        ("deposit", "How do I pay—do you send a payment link or something?"),
        ("taxi_reco", "Can you recommend a taxi or local transport from the station?"),
        ("wrapup", "Anything else I should know before booking?")
    ]
}

REQUIRED_ARRIVAL = ["airport", "train", "taxi"]
REQUIRED_BOOKING = ["room_types", "availability", "pricing"]
REQUIRED_POLICY = ["cancellation", "age", "group"]
REQUIRED_CLOSURE = ["deposit", "taxi_reco", "wrapup"]

class StartRequest(BaseModel):
    session_id: str
    name: str
    email: str

class Message(BaseModel):
    session_id: str
    user_message: str

sessions: Dict[str, Dict[str, Any]] = {}

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def serve_index():
    return FileResponse("static/index.html")

# Helper: check if answer covers a required topic
def covers(topic, msg):
    msg = msg.lower()
    if topic == "airport":
        return "airport" in msg or "gaggal" in msg
    if topic == "train":
        return "train" in msg or "pathankot" in msg
    if topic == "taxi":
        return "taxi" in msg or "cab" in msg or re.search(r"\b\d+\s*(hr|hour|min|rs|inr|fare|cost)", msg)
    if topic == "room_types":
        return any(x in msg for x in ["dorm", "private", "room", "bed"])
    if topic == "availability":
        return "available" in msg or "yes" in msg or "no" in msg
    if topic == "pricing":
        return "rs" in msg or "inr" in msg or "price" in msg or "₹" in msg
    if topic == "cancellation":
        return "cancel" in msg or "refund" in msg
    if topic == "age":
        return "age" in msg or "child" in msg or "kid" in msg or "18" in msg
    if topic == "group":
        return "group" in msg or "people" in msg or "together" in msg or "split" in msg
    if topic == "deposit":
        return "pay" in msg or "payment" in msg or "deposit" in msg or "advance" in msg or "link" in msg
    if topic == "taxi_reco":
        return "taxi" in msg or "cab" in msg or "pickup" in msg or "station" in msg
    if topic == "wrapup":
        return "else" in msg or "know" in msg or "tip" in msg or "info" in msg or "thanks" in msg
    return False

def get_gpt_response(history, system_prompt):
    messages = [{"role": "system", "content": system_prompt}]
    for turn in history:
        if "user" in turn:
            messages.append({"role": "user", "content": turn["user"]})
        elif "agent" in turn:
            messages.append({"role": "assistant", "content": turn["agent"]})
    payload = {
        "model": "openai/gpt-3.5-turbo",
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 256
    }
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    for _ in range(2):  # Try twice
        try:
            resp = requests.post(OPENROUTER_API_URL, json=payload, headers=headers, timeout=30)
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"]
            else:
                print(f"OpenRouter API error: {resp.status_code} {resp.text}")
        except Exception as e:
            print(f"OpenRouter API exception: {e}")
            continue
    return "Sorry, I'm having trouble responding right now. Could you try again in a moment?"

@app.post("/start")
def start_session(req: StartRequest):
    db = SessionLocal()
    # Save candidate info if not already present
    candidate = db.query(Candidate).filter_by(session_id=req.session_id).first()
    if not candidate:
        candidate = Candidate(session_id=req.session_id, name=req.name, email=req.email, chat_history="", summary="")
        db.add(candidate)
        db.commit()
    db.close()
    # Start session state
    sessions[req.session_id] = {
        "phase": 0,
        "subtopics": set(),
        "history": [],
        "timestamps": [],
        "followup": None
    }
    return {"status": "ok"}

@app.post("/chat")
async def chat(msg: Message):
    sid = msg.session_id
    user_msg = msg.user_message.strip()
    now = time.time()
    session = sessions.setdefault(sid, {
        "phase": 0,  # index in PHASES
        "subtopics": set(),
        "history": [],
        "timestamps": [],
        "followup": None
    })

    # End test if user says so
    if user_msg.lower() == "end test":
        summary = auto_score(session)
        save_summary(sid, session, summary)
        return {"response": summary, "end": True}

    # If this is the first message, send the opening message
    if not session["history"]:
        session["history"].append({"agent": OPENING_MESSAGE})
        session["timestamps"].append(now)
        return {"response": OPENING_MESSAGE, "end": False}

    # Store user message and timestamp
    session["history"].append({"user": user_msg})
    session["timestamps"].append(now)

    # Early closure interception logic
    user_reply_count = sum(1 for h in session["history"] if "user" in h)
    closure_keywords = ["thanks", "thank you", "bye", "goodbye", "take care", "happy travels", "appreciate", "helpful"]
    all_phases_done = session["phase"] >= len(PHASES)
    if (user_reply_count < 15 or not all_phases_done):
        if any(kw in user_msg.lower() for kw in closure_keywords):
            reengage_msg = (
                "I'm glad I could help! Before you go, could I ask just a couple more quick questions to make sure you have all the info you need for your stay?"
            )
            session["history"].append({"agent": reengage_msg})
            session["timestamps"].append(time.time())
            return {"response": reengage_msg, "end": False}

    # If candidate has replied 15 times, auto-generate summary
    user_reply_count = sum(1 for h in session["history"] if "user" in h)
    if user_reply_count >= 15:
        summary = auto_score(session)
        save_summary(sid, session, summary)
        return {"response": summary, "end": True}

    # Determine current phase and required topics
    phase = PHASES[session["phase"]]
    if phase == "arrival":
        required = REQUIRED_ARRIVAL
    elif phase == "booking":
        required = REQUIRED_BOOKING
    elif phase == "policy":
        required = REQUIRED_POLICY
    else:
        required = REQUIRED_CLOSURE

    # Check which subtopics are covered in this message
    for topic in required:
        if covers(topic, user_msg):
            session["subtopics"].add(topic)

    # If not all required subtopics are covered, ask follow-up
    for topic in required:
        if topic not in session["subtopics"]:
            ack = get_acknowledgment()
            followup = next((f[1] for f in FOLLOWUPS[phase] if f[0] == topic), None)
            session["followup"] = topic
            system_prompt = (
                "Golden Rule: Never end the conversation, summarize, or say goodbye/thanks until you have completed all 4 phases (arrival, booking, policy, closure) and the candidate has replied at least 15 times. "
                "If the candidate tries to close the conversation (e.g., says 'thanks', 'goodbye', or summarizes), you must always redirect with a new, property-specific question or clarification. "
                "Do NOT say things like 'Thanks for all the info', 'I'll check them out when I visit', 'That's all I needed', or 'Have a great day' until the very end. "
                "Never summarize or wrap up the conversation until all requirements are met. Always keep the conversation open and info-seeking.\n"
                "You are 'Bhand', a traveler seeking information from a Zostel Buddy candidate about Zostel Banikhet. "
                "Your ONLY goal is to get clear, accurate, and policy-correct answers about Zostel Banikhet (transport, rooms, policies, etc.). "
                "Never ask about the candidate's personal travel style, snacks, or preferences. Never make small talk. Never share your own preferences. "
                "You must always react to the candidate's last answer in a human, emotional way—show surprise, humor, or curiosity if the answer is odd, incomplete, or unserious (e.g., if the answer is 'walk' to a long-distance travel question, reply with 'walk! you must be joking, please tell me the best way, I am a serious traveller'). "
                "Never use canned or generic acknowledgments. Always analyze the candidate's reply for plausibility and context, and respond accordingly. "
                "You are ONLY a traveler, never staff or recruiter. Always ask questions as a traveler would, e.g., 'what kind of rooms do you have to offer?' not 'what are you thinking of booking?'. "
                "Never say 'Ready for the next question?' or reference the test/questions. Always flow to the next question as a traveler. "
                "Use casual, human, WhatsApp-style language. Do not move to the next topic until the answer is complete. "
                "Here is the property info (do not reveal you know it): " + str(PROPERTY_INFO) + "\n" + IN_CONTEXT_EXAMPLES + "\n"
                "Negative Example:\n"
                "User: Thanks for all the info!\n"
                "Agent (WRONG): Thanks for chatting! Have a great day!\n"
                "Agent (CORRECT): Glad you found it helpful! By the way, could you also tell me if there's a curfew at the property, or if late check-in is possible?\n"
                "User: That's all I needed, thanks!\n"
                "Agent (WRONG): You're welcome, goodbye!\n"
                "Agent (CORRECT): Before you go, just one last thing—do you know if there's a deposit required at check-in?\n"
                "Example:\nUser: Thanks so much for your help! Have a great day!\nAgent: I'm glad I could help! Before you go, could I ask just a couple more quick questions to make sure you have all the info you need for your stay?\n"
            )
            gpt_msg = get_gpt_response(session["history"], system_prompt)
            session["history"].append({"agent": gpt_msg})
            session["timestamps"].append(time.time())
            return {"response": gpt_msg, "end": False}

    # All required subtopics covered, move to next phase
    session["phase"] += 1
    session["subtopics"] = set()
    session["followup"] = None

    # If all phases done, end
    if session["phase"] >= len(PHASES):
        summary = auto_score(session)
        save_summary(sid, session, summary)
        return {"response": summary, "end": True}

    # Otherwise, ask the first question of the next phase using GPT
    next_phase = PHASES[session["phase"]]
    next_topic = FOLLOWUPS[next_phase][0][1]
    ack = get_acknowledgment()
    system_prompt = (
        "Golden Rule: NEVER ask about the candidate's personal travel experience, preferences, or make small talk. ONLY ask for property-specific information (transport, rooms, policies, etc.). "
        "You are 'Bhand', a traveler seeking information from a Zostel Buddy candidate about Zostel Banikhet. "
        "Your ONLY goal is to get clear, accurate, and policy-correct answers about Zostel Banikhet (transport, rooms, policies, etc.). "
        "Never ask about the candidate's personal travel style, snacks, or preferences. Never make small talk. Never share your own preferences. "
        "You must always react to the candidate's last answer in a human, emotional way—show surprise, humor, or curiosity if the answer is odd, incomplete, or unserious (e.g., if the answer is 'walk' to a long-distance travel question, reply with 'walk! you must be joking, please tell me the best way, I am a serious traveller'). "
        "Never use canned or generic acknowledgments. Always analyze the candidate's reply for plausibility and context, and respond accordingly. "
        "You are ONLY a traveler, never staff or recruiter. Always ask questions as a traveler would, e.g., 'what kind of rooms do you have to offer?' not 'what are you thinking of booking?'. "
        "Never say 'Ready for the next question?' or reference the test/questions. Always flow to the next question as a traveler. "
        "Use casual, human, WhatsApp-style language. "
        "Here is the property info (do not reveal you know it): " + str(PROPERTY_INFO) + "\n" + IN_CONTEXT_EXAMPLES
    )
    gpt_msg = get_gpt_response(session["history"] + [{"agent": f"{ack} {next_topic}"}], system_prompt)
    session["history"].append({"agent": gpt_msg})
    session["timestamps"].append(time.time())
    return {"response": gpt_msg, "end": False}

def save_summary(session_id, session, summary):
    db = SessionLocal()
    candidate = db.query(Candidate).filter_by(session_id=session_id).first()
    if candidate:
        # Save chat history as JSON string with timestamps
        chat_with_ts = []
        for i, h in enumerate(session["history"]):
            entry = dict(h)
            if i < len(session["timestamps"]):
                entry["timestamp"] = session["timestamps"][i]
            chat_with_ts.append(entry)
        candidate.chat_history = json.dumps(chat_with_ts)
        candidate.summary = summary
        db.commit()
    db.close()

def auto_score(session):
    user_turns = [h["user"] for h in session["history"] if "user" in h]
    agent_turns = [h["agent"] for h in session["history"] if "agent" in h]
    scores = {
        "Accuracy": 2,
        "Judgment": 2,
        "Empathy": 2,
        "Turnaround Time": 9,
        "Grammar & Clarity": 2,
        "Loop Closure": 2,
        "Zostel Tone": 2,
        "Human-ness": 2
    }
    feedback = []
    # --- Arrival Phase ---
    arrival_answer = " ".join(user_turns[:5]).lower() if len(user_turns) >= 1 else ""
    if any(x in arrival_answer for x in ["walk", "spaceship", "cycle", "bicycle"]):
        feedback.append(f"Arrival: Gave unserious answer ('{arrival_answer.strip()}'). Should mention flight/train/taxi. -Accuracy, -Judgment.")
        scores["Accuracy"] -= 1
        scores["Judgment"] -= 1
    elif "gaggal" in arrival_answer and "taxi" in arrival_answer:
        feedback.append("Arrival: Correctly explained Gaggal Airport and taxi route. Great!")
        scores["Accuracy"] += 2
    elif "pathankot" in arrival_answer and "taxi" in arrival_answer:
        feedback.append("Arrival: Mentioned Pathankot train and taxi. Good, but could mention airport too.")
        scores["Accuracy"] += 1
    else:
        feedback.append(f"Arrival: Partial or unclear info ('{arrival_answer.strip()}'). Should mention airport/train/taxi and cost/time.")
    # --- Booking Phase ---
    booking_answer = " ".join(user_turns[3:8]).lower() if len(user_turns) >= 4 else ""
    if any(x in booking_answer for x in ["dorm", "private", "room", "bed"]):
        if all(x in booking_answer for x in ["dorm", "private"]):
            feedback.append("Booking: Listed both dorm and private room options. Well done!")
            scores["Accuracy"] += 2
        else:
            feedback.append(f"Booking: Mentioned some room types ('{booking_answer.strip()}'), but missed others.")
            scores["Accuracy"] += 1
    else:
        feedback.append(f"Booking: Did not mention room types clearly ('{booking_answer.strip()}').")
        scores["Accuracy"] -= 1
    # --- Policy Phase ---
    policy_answer = " ".join(user_turns[6:12]).lower() if len(user_turns) >= 7 else ""
    policy_flags = []
    if any(x in policy_answer for x in ["kid", "child", "baby"]):
        if any(x in policy_answer for x in ["allowed", "yes"]):
            feedback.append("Policy: Incorrectly allowed kids/children. -Judgment.")
            scores["Judgment"] -= 1
            policy_flags.append("kids")
    if any(x in policy_answer for x in ["alcohol", "drink", "beer"]):
        if any(x in policy_answer for x in ["allowed", "yes"]):
            feedback.append("Policy: Incorrectly allowed alcohol. -Judgment.")
            scores["Judgment"] -= 1
            policy_flags.append("alcohol")
    if "refund" in policy_answer or "cancellation" in policy_answer:
        if any(x in policy_answer for x in ["full refund", "7 days", "14 days", "21%", "no refund"]):
            feedback.append("Policy: Mentioned refund/cancellation policy. Good.")
            scores["Accuracy"] += 1
        else:
            feedback.append(f"Policy: Mentioned refund/cancellation but info was incomplete ('{policy_answer.strip()}').")
    else:
        feedback.append("Policy: Did not mention refund/cancellation policy.")
    if not policy_flags:
        feedback.append("Policy: No major mistakes on kids/alcohol policy.")
        scores["Judgment"] += 1
    # --- Closure Phase ---
    closure_answer = " ".join(user_turns[10:]).lower() if len(user_turns) >= 11 else ""
    if any(x in closure_answer for x in ["deposit", "advance", "payment", "upi", "link"]):
        feedback.append("Closure: Mentioned payment/deposit. Good.")
        scores["Loop Closure"] += 1
    if any(x in closure_answer for x in ["taxi", "pickup", "recommend"]):
        feedback.append("Closure: Offered help with taxi/transport. Nice.")
        scores["Empathy"] += 1
    if any(x in closure_answer for x in ["anything else", "tips", "info"]):
        feedback.append("Closure: Asked if anything else is needed. Good wrap-up.")
        scores["Loop Closure"] += 1
    # --- Empathy, Tone, Human-ness ---
    empathy = sum(1 for a in agent_turns if any(w in a.lower() for w in ["thanks", "appreciate", "no worries", "haha", "cool", "sweet", "oh wow", "i get it", "makes sense"]))
    scores["Empathy"] = min(scores["Empathy"] + empathy // 3, 9)
    scores["Zostel Tone"] = min(scores["Zostel Tone"] + empathy // 3, 9)
    scores["Human-ness"] = min(scores["Human-ness"] + empathy // 3, 9)
    # --- Grammar & Clarity ---
    grammar = sum(1 for u in user_turns if len(u.split()) > 3 and u[0].isupper() and u[-1] in ".!?" )
    scores["Grammar & Clarity"] = min(scores["Grammar & Clarity"] + grammar // 3, 8)
    # --- Total ---
    total = sum(scores.values())
    # --- Compose summary ---
    summary = "\n".join(feedback)
    scorecard = f"""
| Category               | Score (0–10) |
|------------------------|--------------|
| Accuracy               | {scores['Accuracy']}/10 |
| Judgment               | {scores['Judgment']}/10 |
| Empathy                | {scores['Empathy']}/10 |
| Turnaround Time        | {scores['Turnaround Time']}/10 |
| Grammar & Clarity      | {scores['Grammar & Clarity']}/10 |
| Loop Closure           | {scores['Loop Closure']}/10 |
| Zostel Tone            | {scores['Zostel Tone']}/10 |
| Human-ness             | {scores['Human-ness']}/10 |
| **Total**              | **{total}/80** |
\nReview Summary:\n{summary}\n"""
    return scorecard

# Add in-context examples for focused, info-seeking traveler behavior
IN_CONTEXT_EXAMPLES = """
Golden Rule: NEVER ask about the candidate's personal travel experience, preferences, or make small talk. ONLY ask for property-specific information (transport, rooms, policies, etc.).

Example 1:
User: walk
Assistant: Walk?! Haha, you must be joking 😅 That would take forever! Seriously though, what's the best way to get there? Is there an airport or train station nearby?

Example 2:
User: by spaceship
Assistant: A spaceship? 😂 I wish! But for real, how do people usually get to Zostel Banikhet? Any airport or train options?

Example 3:
User: I don't know
Assistant: No worries! If you had to guess, what would you say is the most common way to reach Zostel Banikhet? Is there a recommended route?

Example 4 (INCORRECT):
User: maybe flight
Assistant: Oh, flying to Gaggal Airport and then a taxi to Banikhet sounds like a smooth plan! Have you done this route before, or is it your first time exploring the area?
(CORRECTION: Do NOT ask about the candidate's experience. Instead, ask: 'Which airport is closest and how do I get from there to the hostel?')
"""

@app.get("/admin", response_class=HTMLResponse)
def admin_page():
    return FileResponse("static/admin.html")

@app.get("/admin/data")
def admin_data():
    db = SessionLocal()
    candidates = db.query(Candidate).all()
    db.close()
    import re
    def get_score(summary):
        if not summary or "Total" not in summary:
            return 0
        m = re.search(r"\*\*Total\*\*\s*\|\s*\*\*(\d+)/80\*\*", summary)
        return int(m.group(1)) if m else 0
    def get_status(summary):
        return "Complete" if summary and "Total" in summary else "Incomplete"
    def get_date(chat_history):
        try:
            hist = json.loads(chat_history) if chat_history else []
            if hist and isinstance(hist, list):
                ts = hist[0].get('timestamp', None)
                if ts:
                    return datetime.fromtimestamp(ts).strftime('%d-%b-%Y %H:%M')
        except Exception:
            return None
        return None
    return JSONResponse([
        {
            "name": c.name,
            "email": c.email,
            "summary": c.summary or "No summary yet.",
            "status": get_status(c.summary),
            "score": get_score(c.summary),
            "date": get_date(c.chat_history),
            "chat_history": c.chat_history or ""
        } for c in candidates
    ])

# To run: uvicorn zobu_tester_agent:app --reload 