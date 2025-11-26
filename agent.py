
import logging
import os
import re
import json
import time
from dotenv import load_dotenv
from typing import Optional

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
    tokenize,
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")
load_dotenv(".env.local")


# -------------------------
# Simple FAQ-based SDR agent
# -------------------------
# Paths (file is located at backend/src/agent.py so data is ../data)
BASE_DIR = os.path.dirname(__file__)
FAQ_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "data", "faq.json"))
LEADS_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "data", "leads"))

if not os.path.exists(LEADS_DIR):
    os.makedirs(LEADS_DIR, exist_ok=True)

# Load FAQ if present; fallback to a tiny example FAQ to avoid crashes
try:
    with open(FAQ_PATH, "r", encoding="utf-8") as fh:
        FAQ = json.load(fh)
except Exception:
    FAQ = [
        {"q": "What is Zomato?", "a": "Zomato is a food delivery and restaurant discovery platform operating across India."},
        {"q": "Do you have a free tier?", "a": "The basic Zomato app is free to use. Membership plans are optional."},
        {"q": "What are your delivery hours?", "a": "Delivery hours depend on the restaurant; many operate 9 AM - 9 PM but it varies by location."},
    ]


# Lead fields and helpers
LEAD_FIELDS = ["name", "company", "email", "role", "use_case", "team_size", "timeline"]


def init_lead_state():
    return {k: None for k in LEAD_FIELDS}


def save_lead_json(lead: dict):
    """Save lead to a timestamped JSON file."""
    try:
        filename = f"lead_{int(time.time())}.json"
        path = os.path.join(LEADS_DIR, filename)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(lead, fh, indent=2, ensure_ascii=False)
        logger.info(f"Saved lead to {path}")
    except Exception as e:
        logger.exception(f"Failed to save lead: {e}")


def generate_summary(lead: dict) -> str:
    """Create a short verbal summary from lead fields."""
    parts = []
    if lead.get("name"):
        parts.append(f"{lead['name']}")
    if lead.get("company"):
        parts.append(f"from {lead['company']}")
    if lead.get("role"):
        parts.append(f"({lead['role']})")
    header = " ".join(parts) if parts else "A contact"
    use_case = lead.get("use_case", "no specific use case mentioned")
    team = lead.get("team_size", "unknown team size")
    timeline = lead.get("timeline", "unspecified timeline")
    summary = f"Here’s a quick summary: {header} wants to use the product for {use_case}. Team size is {team}. Timeline: {timeline}."
    return summary


def find_faq_answer(question: str) -> Optional[str]:
    """Very simple keyword-based FAQ lookup. Returns the answer text or None."""
    if not question:
        return None
    q = question.lower()
    # Try to match question tokens against FAQ question text first (exact-ish)
    for item in FAQ:
        q_text = (item.get("q") or "").lower()
        if q_text and any(tok in q_text for tok in q.split()):
            return item.get("a")
    # Fallback: look for any token in the answer text
    for item in FAQ:
        a_text = (item.get("a") or "").lower()
        if a_text and any(tok in a_text for tok in q.split()):
            return item.get("a")
    return None


# -------------------------
# Agent implementation
# -------------------------
class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""
You are a friendly Sales Development Representative (SDR) for a company.
Your job is to:
1) Greet visitors warmly and ask what they're looking for.
2) Use only the provided FAQ content to answer product/company/pricing questions.
3) Do not invent facts not present in the FAQ; if you don't know, say you don't have that information.
4) Collect lead information naturally during the conversation:
   - name, company, email, role, use_case, team_size, timeline
5) When the user signals they are finished (e.g. "that's all", "thanks", "i'm done"), give a short verbal summary of the lead and save the lead as JSON.
Keep responses short, polite, and spoken-friendly.
"""
        )


# Keep the same prewarm behavior (VAD)
def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


# Entrypoint
async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    # Build the voice agent pipeline (STT / LLM / TTS)
    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-matthew",
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True,
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    # metrics collector
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # Initialize session userdata for lead capture / conversational state
    session.userdata.setdefault("lead", None)
    session.userdata.setdefault("greeted", False)
    session.userdata.setdefault("asked_lead_offer", False)

    # Simple helper to attempt switching TTS voice per "persona" (best-effort)
    def set_tts_voice(voice_name: str):
        try:
            if hasattr(session, "tts") and session.tts is not None:
                session.tts.voice = voice_name
        except Exception as e:
            logger.debug(f"Could not set TTS voice: {e}")

    # Message handler
    @session.on("message")
    async def _on_message(ev):
        text = (ev.text or "").strip()
        if not text:
            return

        user_message = text.lower()

        # Initialize lead object at first user message
        if session.userdata.get("lead") is None:
            session.userdata["lead"] = init_lead_state()

        lead = session.userdata["lead"]

        # 1) Greeting on first contact
        if not session.userdata.get("greeted"):
            session.userdata["greeted"] = True
            set_tts_voice("en-US-matthew")
            await ev.respond("Hi! 👋 Thanks for reaching out. What brought you here today? What are you working on?")
            return

        # 2) Check for end-of-call keywords (user finishes)
        if any(k in user_message for k in ["that's all", "thats all", "i'm done", "im done", "i am done", "done", "thanks", "thank you"]):
            # If we haven't collected anything yet, say a polite close
            if not any(lead.values()):
                await ev.respond("Thanks for the chat! If you'd like to share contact details later, feel free to reach out.")
                return
            summary = generate_summary(lead)
            save_lead_json(lead)
            await ev.respond(summary)
            await ev.respond("Thanks — our team will reach out soon. Have a great day!")
            # Reset lead for the session if you want to allow new leads in same session
            session.userdata["lead"] = init_lead_state()
            session.userdata["greeted"] = False
            return

        # 3) If user asks a question related to FAQ, answer from FAQ only
        faq_answer = find_faq_answer(text)
        if faq_answer:
            # Use the product/company voice for FAQ responses
            set_tts_voice("en-US-matthew")
            await ev.respond(faq_answer)
            # After answering, prompt for lead capture gently if we haven't asked yet
            if not session.userdata.get("asked_lead_offer"):
                session.userdata["asked_lead_offer"] = True
                await ev.respond("By the way, would you like me to note your details so our team can follow up? If yes, may I have your name?")
            return

        # 4) Lead collection flow (step-by-step)
        # Determine next missing field
        next_field = None
        for f in LEAD_FIELDS:
            if not lead.get(f):
                next_field = f
                break

        # If user said 'yes' or similar when offered lead capture, prompt for name
        if session.userdata.get("asked_lead_offer") and next_field is None:
            # already collected everything
            await ev.respond("Thanks — I've got your details. Say 'that's all' when you're done.")
            return

        # If the system recently asked to start collecting and user replied with a name
        if session.userdata.get("asked_lead_offer") and next_field == "name":
            lead["name"] = text
            await ev.respond(f"Nice to meet you, {lead['name']}! Which company are you with?")
            return

        # Fill company
        if next_field == "company":
            lead["company"] = text
            await ev.respond("Great — what's your role there?")
            return

        # Fill role
        if next_field == "role":
            lead["role"] = text
            await ev.respond("What would you like to use our product for? (briefly)")
            return

        # Fill use_case
        if next_field == "use_case":
            lead["use_case"] = text
            await ev.respond("Understood. How big is your team?")
            return

        # Fill team_size
        if next_field == "team_size":
            lead["team_size"] = text
            await ev.respond("Thanks. When do you plan to get started — now, soon, or later?")
            return

        # Fill timeline
        if next_field == "timeline":
            lead["timeline"] = text
            await ev.respond("Almost done — could I get the best email to reach you at?")
            return

        # Fill email (validity check)
        if next_field == "email":
            # basic email validation
            if "@" not in text or "." not in text.split("@")[-1]:
                await ev.respond("Please provide a valid email address (e.g. name@example.com).")
                return
            lead["email"] = text
            await ev.respond("Thanks — I’ve saved that. Say 'that's all' when you're ready and I'll share a summary.")
            return

        # If none of the above matched, provide a polite fallback
        # Encourage user to ask product questions or start lead flow
        await ev.respond("I’m here to help with product questions (I can answer from the FAQ) or to note your contact details for follow up. What would you like to do?")
        return

    # Start the session and join the room
    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVC()),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
