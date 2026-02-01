"""Service-specific actions."""

from gru.actions.services.web import (
    NavigateAction,
    ClickAction,
    TypeAction,
    ExtractAction,
    ScreenshotAction,
    WaitAction,
)
from gru.actions.services.search import (
    WebSearchAction,
    LocalSearchAction,
    DistanceAction,
    RestaurantSearchAction,
)
from gru.actions.services.ubereats import (
    UberEatsSearchAction,
    UberEatsOrderAction,
    UberEatsCartAction,
)
from gru.actions.services.google import (
    CreateDocumentAction,
    WriteDocumentAction,
    SendEmailAction,
    CompileDocumentAction,
    set_google_connector,
    get_google_connector,
)
from gru.actions.services.research import (
    ResearchAction,
    QuickAnswerAction,
    set_research_claude,
    get_research_claude,
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
