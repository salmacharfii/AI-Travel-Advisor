"""Travel planner chatbot - Streamlit + OpenAI.

Setup (Windows):
    py -m pip install streamlit openai pydantic
    set OPENAI_API_KEY=your_key_here
    py -m streamlit run app.py
"""

import base64
from typing import Optional

import streamlit as st
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


client = OpenAI()  # reads OPENAI_API_KEY from the environment

# --- Session state (survives Streamlit reruns) ---
if "history" not in st.session_state:
    st.session_state.history = [{"role": "system", "content": SYSTEM_PROMPT}]
    st.session_state.display = []  # what we show: {"role", "text", "plan_md"}
    st.session_state.usage = {"input": 0, "output": 0}
    st.session_state.last_plan = None
    st.session_state.image = None

ss = st.session_state


def chat(user_text: str) -> Reply:
    ss.history.append({"role": "user", "content": user_text})
    response = client.chat.completions.parse(
        model=MODEL,
        messages=ss.history,
        response_format=Reply,
        max_tokens=2000,
    )
    ss.usage["input"] += response.usage.prompt_tokens
    ss.usage["output"] += response.usage.completion_tokens
    ss.history.append({"role": "assistant", "content": response.choices[0].message.content})
    return response.choices[0].message.parsed


def make_image_bytes(plan_text: str) -> bytes:
    p = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content":
                   "Write a one-paragraph prompt for a beautiful travel poster "
                   "illustration based on this trip plan:\n\n" + plan_text}],
        max_tokens=200,
    )
    img = client.images.generate(
        model="gpt-image-1",
        prompt=p.choices[0].message.content,
        size="1024x1024",
    )
    return base64.b64decode(img.data[0].b64_json)


# --- UI ---
st.title("✈️ Travel Planner")

with st.sidebar:
    u = ss.usage
    cost = u["input"] / 1e6 * PRICE_IN + u["output"] / 1e6 * PRICE_OUT
    st.metric("Input tokens", u["input"])
    st.metric("Output tokens", u["output"])
    st.metric("Est. cost (USD)", f"${cost:.5f}")
    if st.button("New conversation"):
        st.session_state.clear()
        st.rerun()

for m in ss.display:
    with st.chat_message(m["role"]):
        st.markdown(m["text"])
        if m.get("plan_md"):
            st.markdown(m["plan_md"])

if ss.image:
    st.image(ss.image, width=500)

if ss.last_plan and st.button("🎨 Create image"):
    with st.spinner("Generating image..."):
        ss.image = make_image_bytes(ss.last_plan)
    st.rerun()

if user_text := st.chat_input("Where do you want to go?"):
    ss.display.append({"role": "user", "text": user_text})
    with st.spinner("Thinking..."):
        reply = chat(user_text)
    plan_md = plan_to_markdown(reply.plan) if reply.plan else None
    if plan_md:
        ss.last_plan = plan_md
        ss.image = None
    ss.display.append({"role": "assistant", "text": reply.message, "plan_md": plan_md})
    st.rerun()