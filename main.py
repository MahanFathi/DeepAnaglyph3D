import argparse

from config import get_cfg_defaults
from model import build_model
from data import build_dataset

def main():
    parser = argparse.ArgumentParser(description="Deep Neural Networks for 3D Anaglyph Image Generation")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="file",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--mode",
        default="test",
        metavar="mode",
        help="'train' or 'test'",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    # build the config
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    model, optimizer = build_model(cfg)
    dataset = build_dataset(cfg)


if __name__ == "__main__":
    main()
