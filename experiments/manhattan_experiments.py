from os.path import dirname, abspath

import sys

sys.path.insert(0, dirname(abspath(__file__)))
from manhattan_utils.generate_experiments import generate_manhattan_experiments
import logging, coloredlogs

logger = logging.getLogger(__name__)
field_styles = {
    "filename": {"color": "green"},
    "levelname": {"bold": True, "color": "black"},
    "name": {"color": "blue"},
}
coloredlogs.install(
    level="INFO",
    fmt="[%(filename)s:%(lineno)d] %(name)s %(levelname)s - %(message)s",
    field_styles=field_styles,
)

if __name__ == "__main__":
    generate_manhattan_experiments()

    raise NotImplementedError("TODO: implement this")
