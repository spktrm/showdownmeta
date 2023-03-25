import json
import torch
import requests

from analysis.analyse import Sampler

import multiprocessing as mp

from meloetta.data import (
    BOOSTS,
    VOLATILES,
    PSEUDOWEATHERS,
    SIDE_CONDITIONS,
    get_item_effect_token,
    get_status_token,
    get_gender_token,
    get_species_token,
    get_ability_token,
    get_item_token,
    get_move_token,
    get_type_token,
    get_weather_token,
)

NATURES = {
    "lax": 0,
    "quiet": 1,
    "impish": 2,
    "bashful": 3,
    "gentle": 4,
    "sassy": 5,
    "bold": 6,
    "relaxed": 7,
    "careful": 8,
    "calm": 9,
    "rash": 10,
    "serious": 11,
    "brave": 12,
    "hasty": 13,
    "naughty": 14,
    "docile": 15,
    "mild": 16,
    "timid": 17,
    "hardy": 18,
    "modest": 19,
    "lonely": 20,
    "jolly": 21,
    "adamant": 22,
    "quirky": 23,
    "naive": 24,
}


class TrainDataset:
    def __init__(
        self,
        url: str,
        gen: int = 9,
        num_buffers: int = 2048,
    ):
        datum = requests.get(url).json()
        datum = json.loads(json.dumps(datum).lower().replace("-", "").replace(" ", ""))

        self.sampler = Sampler(datum)
        self.gen = gen

        self.anc_buffers = [
            torch.empty(6, 14, dtype=torch.long).share_memory_()
            for _ in range(num_buffers)
        ]
        self.pos_buffers = [
            torch.empty(6, 14, dtype=torch.long).share_memory_()
            for _ in range(num_buffers)
        ]
        self.neg_buffers = [
            torch.empty(6, 14, dtype=torch.long).share_memory_()
            for _ in range(num_buffers)
        ]

        self.free_queue = mp.Queue(maxsize=num_buffers)
        for m in range(num_buffers):
            self.free_queue.put(m)
        self.full_queue = mp.Queue(maxsize=num_buffers)

        procs = []
        for _ in range(16):
            proc = mp.Process(target=self.generate_batch)
            procs.append(proc)
            proc.start()

    def generate_batch(self):
        while True:
            try:
                teams = []
                team = self.sampler.generate()
                teams.append(team)

                positives = []
                init = [member["species"] for member in team[:5]]
                positives.append(self.sampler.generate(init=init))

                negatives = []
                negatives.append(self.sampler.generate())

                batch = []

                for team in teams + positives + negatives:
                    vector_team = []
                    for member in team:
                        member_vector = torch.tensor(
                            [
                                get_species_token(self.gen, "id", member["species"]),
                                get_ability_token(self.gen, "id", member["ability"]),
                                get_item_token(self.gen, "id", member["item"]),
                            ]
                            + [
                                get_move_token(self.gen, "id", move)
                                for move in member["moves"]
                            ]
                            + [-1 for _ in range(4 - len(member["moves"]))]
                            + [NATURES.get(member["nature"])]
                            + member["evs"]
                        )
                        vector_team.append(member_vector)
                    vector_team = torch.stack(vector_team)
                    batch.append(vector_team)

                batch = torch.stack(batch)
                anc, pos, neg = batch.chunk(3, 0)

                index = self.free_queue.get()

                self.anc_buffers[index][...] = anc
                self.pos_buffers[index][...] = pos
                self.neg_buffers[index][...] = neg

                self.full_queue.put(index)
            except:
                pass

    def get_batch(self, batch_size):
        indices = [self.full_queue.get() for _ in range(batch_size)]
        batch = (
            torch.stack([self.anc_buffers[index] for index in indices]),
            torch.stack([self.pos_buffers[index] for index in indices]),
            torch.stack([self.neg_buffers[index] for index in indices]),
        )
        for index in indices:
            self.free_queue.put(index)
        return batch
