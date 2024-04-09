import argparse
import logging

from pathlib import Path
import sys


def get_kwargs() -> argparse.ArgumentParser:
    """Parse script args if run through command line.
    Here waits for logging level param.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-l",
        "--logger_level",
        metavar="<logger_level>",
        type=int,
        help=("NOTSET: 0, DEBUG: 10, INFO: 20, "
              "WARNING: 30, ERROR: 40, CRITICAL: 50"),
        default=20,
    )
    return parser


def get_logger(
        logger_name: str | None = None,
        level: int = logging.DEBUG) -> logging.Logger:
    """Custom logger."""
    formatter = logging.Formatter(
        "%(levelname)-8s [%(asctime)s] %(name)s:%(lineno)d: %(message)s"
    )

    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(level)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)
    return logger


def get_messages(question: str | None = None,
                 history: list[dict[str, str]] | None = None,
                 context: str | None = None,
                 system_prompt: str | None = None,
                 context_prompt: str | None = None
                 ) -> list[dict[str, str]]:

    if history is None:
        history = []

    if system_prompt is not None:
        system_prompt_message = {
            'role': 'system',
            'content': system_prompt
        }
        history = [system_prompt_message] + history

    if (len(history) > 1 and history[0]['role'] == "system"
            and history[1]['role'] == "system"):
        raise ValueError("Passed history contains system prompt."
                         "More then one systen prompt is not allowed.")

    if context is not None:
        context_prompt = context_prompt + "\n" if context_prompt else ""
        context_text = context_prompt + context

        context_message = {
            'role': 'user',
            'content': context_text
        }
        history.append(context_message)

    if question is not None:
        question_message = {
            "role": 'user',
            "content": question
        }
        history.append(question_message)
    return history
