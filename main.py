"""Start the 3D gravity sandbox.

    python main.py                     # supernova finale
    python main.py --preset solar      # solar system from the bundled JPL snapshot

Press H in the window for the controls.
"""
import argparse

from gravity_sandbox.app import run
from gravity_sandbox.presets import PRESETS


def window_size(text: str) -> tuple[int, int]:
    try:
        width, height = (int(part) for part in text.lower().split("x"))
    except ValueError:
        raise argparse.ArgumentTypeError("use WIDTHxHEIGHT, for example 1280x800") from None
    return width, height


def main() -> None:
    parser = argparse.ArgumentParser(description="3D N-body gravity sandbox")
    parser.add_argument("--preset", choices=list(PRESETS), default="supernova", help="scene to start with")
    parser.add_argument("--size", type=window_size, default=(1100, 700), help="window size, e.g. 1280x800")
    parser.add_argument("--frames", type=int, default=0, help="quit after this many frames (0 = run until closed)")
    parser.add_argument("--screenshot", help="save the last frame to this PNG file when quitting")
    args = parser.parse_args()
    run(preset=args.preset, size=args.size, frames=args.frames, screenshot=args.screenshot)


if __name__ == "__main__":
    main()
