from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status
from openai import OpenAIError
from pydantic import BaseModel, Field

from travel_advisor import TravelAdvisor, plan_to_markdown


load_dotenv()

app = FastAPI(
    title="AI Travel Advisor API",
    description="Create personalized travel plans with an AI advisor.",
    version="1.0.0",
)

advisor = TravelAdvisor()

class ChatRequest(BaseModel):
    """A message sent to the travel advisor."""

    message: str = Field(min_length=1)
    previous_response_id: str | None = None


class ChatResponse(BaseModel):
    """The travel advisor response and its conversation identifier."""

    answer: str
    plan: str | None = None
    response_id: str


@app.get("/health")
def health() -> dict[str, str]:
    """Report that the HTTP service is running."""

    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Send one message in a client-managed conversation."""

    message = request.message.strip()
    if not message:
        raise HTTPException(
            status_code=422,
            detail="Message must not be blank.",
        )

    try:
        reply, response_id = advisor.get_response(
            message,
            previous_response_id=request.previous_response_id,
        )
    except OpenAIError as error:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The AI provider could not process the request.",
        ) from error

    if reply is None:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The AI provider returned no structured reply.",
        )

    plan = plan_to_markdown(reply.plan) if reply.plan else None

    return ChatResponse(
        answer=reply.message,
        plan=plan,
        response_id=response_id,
    )