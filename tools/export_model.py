import torch
import pickle
import argparse
import collections

BLACK_LISTS = []


def state2model(src, dst, prefix=""):
    state = torch.load(src, map_location="cpu")
    models = {}
    for key in state["model"]:
        if key not in BLACK_LISTS:
            models[prefix + key] = state["model"][key]
    state["model"] = collections.OrderedDict(models)
    torch.save(state, dst)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("src", type=str, help="source file path")
    parser.add_argument("dst", type=str, help="target file path")
    parser.add_argument("--prefix", type=str, default="backbone.bottom_up.model.")

    args = parser.parse_args()
    state2model(args.src, args.dst, args.prefix)
