import json


def build_chains(
    center: str, instruct_chains: dict[str, list[str]]
) -> dict[int, list[tuple[int, int]]]:
    center = int(center.split(".")[0])
    result: dict[int, list[tuple[int, int]]] = {}
    for img, chain in instruct_chains.items():
        img_idx = int(img.split(".")[0])
        result[img_idx] = []
        prev_junction = img_idx
        for junction in chain:
            junction = int(junction.split(".")[0])
            result[img_idx].append((prev_junction, junction))
            prev_junction = junction
        result[img_idx].append((prev_junction, center))
    return result


# TODO: Ability to build the perspective chain automatically according to the instructions.


def get_instructions(path_to_instruction: str) -> tuple[str, dict[str, list[str]]]:
    """Get instructions from instructions.json file.

    Parameters
    ----------
    path_to_instruction : str
        Where the instructions.json is at.

    Returns
    -------
    center: str
    chains: dict[str, list[str]]

    """
    with open(path_to_instruction, mode="r", encoding="UTF-8") as f:
        ic = json.load(f)
    return ic["center"], ic["chains"]


def get_instructions2(path_to_instruction: str) -> tuple[str, dict[str, str]]:
    with open(path_to_instruction, mode="r", encoding="UTF-8") as f:
        ic = json.load(f)
    return ic["center"], ic["chains"]


def build_chains2(
    center: str, instruct_pairs: dict[str, str]
) -> dict[str, list[tuple[str, str]]]:
    # center_idx = int(center.split(".")[0])
    result: dict[str, list[tuple[str, str]]] = {}
    for img, dest in instruct_pairs.items():
        prev_junction = img
        result[img] = []
        while dest != "":
            junction = dest
            dest = instruct_pairs[dest]
            result[img].append((prev_junction, junction))
            prev_junction = junction
        result[img].append((prev_junction, center))
    return result

if __name__ == "__main__":
    with open("./instructions.json", mode="r", encoding="UTF-8") as f:
        ic = json.load(f)
    print(build_chains2(ic["center"], ic["chains"]))
