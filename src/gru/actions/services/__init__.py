"""Service-specific actions."""

from gru.actions.services.google import (
    CompileDocumentAction,
    CreateDocumentAction,
    SendEmailAction,
    WriteDocumentAction,
    get_google_connector,
    set_google_connector,
)
from gru.actions.services.research import (
    QuickAnswerAction,
    ResearchAction,
    get_research_claude,
    set_research_claude,
)
from gru.actions.services.search import (
    DistanceAction,
    LocalSearchAction,
    RestaurantSearchAction,
    WebSearchAction,
)
from gru.actions.services.ubereats import (
    UberEatsCartAction,
    UberEatsOrderAction,
    UberEatsSearchAction,
)
from gru.actions.services.web import (
    ClickAction,
    ExtractAction,
    NavigateAction,
    ScreenshotAction,
    TypeAction,
    WaitAction,
)

__all__ = [
    # Web
    "NavigateAction",
    "ClickAction",
    "TypeAction",
    "ExtractAction",
    "ScreenshotAction",
    "WaitAction",
    # Search
    "WebSearchAction",
    "LocalSearchAction",
    "DistanceAction",
    "RestaurantSearchAction",
    # Uber Eats
    "UberEatsSearchAction",
    "UberEatsOrderAction",
    "UberEatsCartAction",
    # Google
    "CreateDocumentAction",
    "WriteDocumentAction",
    "SendEmailAction",
    "CompileDocumentAction",
    "set_google_connector",
    "get_google_connector",
    # Research
    "ResearchAction",
    "QuickAnswerAction",
    "set_research_claude",
    "get_research_claude",
]
