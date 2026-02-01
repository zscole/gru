"""Action handlers for autonomous actions."""

from gru.actions.handlers.communication import (
    SendEmailHandler,
    SendSlackMessageHandler,
    SendSMSHandler,
    set_google_connector as set_comm_google,
    set_slack_connector as set_comm_slack,
)
from gru.actions.handlers.calendar import (
    CreateEventHandler,
    UpdateEventHandler,
    DeleteEventHandler,
    set_google_connector as set_cal_google,
)
from gru.actions.handlers.reservations import (
    OpenTableReservationHandler,
    ResyReservationHandler,
)
from gru.actions.handlers.payments import (
    VenmoPaymentHandler,
)
from gru.actions.handlers.purchases import (
    DoorDashOrderHandler,
    AmazonOrderHandler,
)


def set_google_connector(connector) -> None:
    """Set Google connector for all handlers that need it."""
    set_comm_google(connector)
    set_cal_google(connector)


def set_slack_connector(connector) -> None:
    """Set Slack connector for all handlers that need it."""
    set_comm_slack(connector)


__all__ = [
    "SendEmailHandler",
    "SendSlackMessageHandler",
    "SendSMSHandler",
    "CreateEventHandler",
    "UpdateEventHandler",
    "DeleteEventHandler",
    "OpenTableReservationHandler",
    "ResyReservationHandler",
    "VenmoPaymentHandler",
    "DoorDashOrderHandler",
    "AmazonOrderHandler",
    "set_google_connector",
    "set_slack_connector",
]
