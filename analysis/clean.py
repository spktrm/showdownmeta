import os
import json

from tqdm.auto import tqdm

from meloetta.room import BattleRoom

from concurrent.futures import ThreadPoolExecutor


def clean_team(team):
    for poke in team:
        poke.pop("side")
        poke.pop("ident")
        poke.pop("searchid")
        poke.pop("slot")
        poke.pop("fainted")
        poke.pop("hp")
        poke.pop("hpcolor")
        poke.pop("maxhp")
        poke.pop("moves")
        poke.pop("boosts")
        poke.pop("status")
        poke.pop("statusStage")
        poke.pop("volatiles")
        poke.pop("turnstatuses")
        poke.pop("movestatuses")
        poke.pop("lastMove")
        poke.pop("statusData")
        poke.pop("timesAttacked")
        poke.pop("sprite")
        poke["moves"] = [move[0] for move in poke["moveTrack"]]
        poke.pop("moveTrack")
    return team


def main():
    rootdir = "logs/gen9ou/"
    cleaned = []
    room = BattleRoom()

    for match_fp in tqdm(os.listdir(rootdir)):
        match_path = os.path.join(rootdir, match_fp)

        with open(match_path, "r") as f:
            log = json.load(f)

        for log_line in log["log"].split("\n"):
            room.recieve(log_line)

        final_state = room.get_battle()
        player_lookup = {
            "p1": final_state["p1"]["name"],
            "p2": final_state["p2"]["name"],
        }
        winner = room.get_js_attr("winner")

        if player_lookup["p1"] == winner:
            label = 1
        elif player_lookup["p2"] == winner:
            label = -1
        else:
            label = 0

        p1_pokemon = clean_team(final_state["p1"]["pokemon"])
        p2_pokemon = clean_team(final_state["p2"]["pokemon"])

        datum = {
            "p1": p1_pokemon,
            "p2": p2_pokemon,
            "label": label,
        }
        cleaned.append(datum)
        room.reset()

    with open("analysis/cleaned.json", "w") as f:
        json.dump(cleaned, f)


if __name__ == "__main__":
    main()
