import logging
import coloredlogs


def get_logger(name: str, level: str = "DEBUG") -> logging.Logger:
    """
    Creates and returns a uniform logger with colored output.

    Args:
        name (str): The name of the logger (usually __name__).
        level (str): The logging level (default: "DEBUG").

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)

    # Set log level for the logger
    logger.setLevel(getattr(logging, level.upper(), "DEBUG"))

    # Create a console handler
    handler = logging.StreamHandler()
    handler.setLevel(getattr(logging, level.upper(), "DEBUG"))

    # Define a logging format
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(handler)

    # Integrate with coloredlogs for better readability
    field_styles = {
        "filename": {"color": "green"},
        "levelname": {"bold": True, "color": "black"},
        "name": {"color": "blue"},
    }
    coloredlogs.install(
        logger=logger,
        level="INFO",
        fmt="[%(filename)s:%(lineno)d] %(name)s %(levelname)s - %(message)s",
        field_styles=field_styles,
    )

    return logger
