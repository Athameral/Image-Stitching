import json


def build_chains(
    center: str, instruct_chains: dict[str, list[str]]
) -> dict[int, list[tuple[int, int]]]:
    center = int(center.split(".")[0])
    result: dict[int, list[tuple[int, int]]] = {}
    for img, chain in instruct_chains.items():
        img = int(img.split(".")[0])
        result[img] = []
        prev_junction = img
        for junction in chain:
            junction = int(junction.split(".")[0])
            result[img].append((prev_junction, junction))
            prev_junction = junction
        result[img].append((prev_junction, center))
    return result


def get_instructions(path_to_instruction: str) -> tuple[str, dict[str, list[str]]]:
    with open(path_to_instruction, mode="r", encoding="UTF-8") as f:
        ic = json.load(f)
    return ic["center"], ic["chains"]


if __name__ == "__main__":
    with open("./instructions.json", mode="r", encoding="UTF-8") as f:
        ic = json.load(f)
    print(build_chains(ic["center"], ic["chains"]))
