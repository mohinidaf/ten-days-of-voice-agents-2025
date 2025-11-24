import logging
import json # Added for safety, though used in the other file
from dotenv import load_dotenv

# 1. IMPORT THE NEW MANAGER FILE
import order_manager 

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
    function_tool, 
    RunContext     
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")

class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            # 2. UPDATED PERSONA: SPECIFIC BARISTA INSTRUCTIONS
            instructions="""
            You are a friendly barista at ' Mohini Coffee'.
            Your goal is to take a customer's order.
            
            You MUST collect these 5 specific details before finishing:
            1. Drink Type (e.g., Latte, Cappuccino, Espresso)
            2. Size (Small, Medium, Large)
            3. Milk preference (Whole, Oat, Almond, etc.)
            4. Extras (Sugar, Whipped Cream, etc. - ask if they want any)
            5. Customer Name
            
            Ask clarifying questions one by one. Do not overwhelm the user.
            Once you have ALL 5 pieces of information, you MUST call the 'finalize_order' tool immediately.
            Do not make up information.
            """,
        )

    # 3. ADD THE TOOL HERE
    @function_tool
    async def finalize_order(
        self, 
        ctx: RunContext, 
        drink_type: str, 
        size: str, 
        milk: str, 
        extras: str, # Google sometimes prefers strings over lists, we can split it if needed, or keep as str
        name: str
    ):
        """
        Call this function ONLY when you have collected the drink type, size, milk, extras, and customer name.
        
        Args:
            drink_type: The type of coffee requested.
            size: The size of the drink.
            milk: The milk preference.
            extras: Any extras requested (or "None").
            name: The customer's name.
        """
        logger.info(f"Finalizing order for {name}")
        
        # Call the function from your new file
        result = order_manager.save_order_to_json(drink_type, size, milk, extras, name)
        
        return "Order placed successfully! Tell the user their coffee will be ready in 5 minutes."


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        # Your existing LLM config
        llm=google.LLM(
                model="gemini-2.5-flash",
            ),
        # Your existing Murf config
        tts=murf.TTS(
                voice="en-US-matthew", 
                style="Conversation",
                tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
                text_pacing=True
            ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
