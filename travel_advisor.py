"""Travel planner chatbot - Streamlit + OpenAI.

Setup (Windows):
    py -m pip install streamlit openai pydantic
    set OPENAI_API_KEY=your_key_here
    py -m streamlit run app.py
"""

import base64
import os
from typing import Optional

from openai import OpenAI
from pydantic import BaseModel, Field

MODEL = "gpt-4o-mini"
PRICE_IN = 0.15   # USD per 1M input tokens (check OpenAI pricing and update)
PRICE_OUT = 0.60  # USD per 1M output tokens

SYSTEM_PROMPT = """You are a friendly travel planning assistant.

1. Greet the user and collect at least: destination, trip duration, interests
   (food, museums, hiking, beaches, nightlife, shopping...), and optional
   preferences/constraints (budget, travel partners, dates).
2. Ask short follow-up questions until you have that information. Do not ask
   unnecessary questions once you have enough.
3. If the user's information is contradictory or ambiguous, ask for
   clarification before planning.
4. Once you have enough information, fill in the plan field. Until then,
   leave plan null and put your question in message.
5. Be creative but keep the itinerary realistic and consistent with the
   user's preferences. Reply in the user's language."""


class trip(BaseModel):
    title: str = Field(..., description="Trip title")
    summary: str = Field(..., description="Trip summary")
    itinerary: list[str] = Field(..., description="Day-by-day itinerary, one string per day")
    tips: str = Field(..., description="Practical travel tips")
    budget: str = Field(..., description="Estimated budget")


class Reply(BaseModel):
    message: str = Field(..., description="Text to show the user (a question, or a short intro to the plan)")
    plan: Optional[trip] = Field(None, description="Fill ONLY when enough info is collected; otherwise null")


def plan_to_markdown(p: trip) -> str:
    days = "\n".join(f"- {d}" for d in p.itinerary)
    return (f"# {p.title}\n\n## Summary\n{p.summary}\n\n## Itinerary\n{days}\n\n"
            f"## Practical Tips\n{p.tips}\n\n## Estimated Budget\n{p.budget}")


client = OpenAI()

def make_image_bytes(plan_text: str) -> bytes:
    p = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content":
                   "Write a one-paragraph prompt for a beautiful portrait travel "
                   "poster illustration based on this trip plan. Use a 2:3 vertical "
                   "composition with a wide establishing view. Keep the complete "
                   "landmark and scene inside the frame with clear space around "
                   "important details; avoid close-ups and cut-off edges.\n\n"
                   + plan_text}],
        max_tokens=200,
    )
    img = client.images.generate(
        model="gpt-image-1",
        prompt=(
            f"{p.choices[0].message.content}\n\n"
            "Portrait 2:3 travel poster. Show the complete scene and landmarks "
            "fully within the frame, with breathing room at every edge. Do not "
            "crop off buildings, people, or other important details."
        ),
        size="1024x1536",
    )
    return base64.b64decode(img.data[0].b64_json)


class TravelAdvisor:
    """Handles conversations with the AI Travel Advisor."""
    def __init__(self):
            api_key = os.getenv("OPENAI_API_KEY")
    
            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY is missing. "
                    "Add it to your .env file."
                )
    
            self.client = OpenAI(api_key=api_key)

    def get_response(self, user_text: str, previous_response_id: str | None = None,)-> tuple[Reply, str]:
        request = {
                    "model": MODEL,
                    "text_format":Reply,
                    "instructions": SYSTEM_PROMPT,
                    "input": user_text,
                }
        if previous_response_id:
            request["previous_response_id"] = previous_response_id

        response = self.client.responses.parse(**request)
        return response.output_parsed , response.id












