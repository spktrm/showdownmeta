import torch
import torch.nn as nn

from meloetta.embeddings import (
    PokedexEmbedding,
    AbilityEmbedding,
    ItemEmbedding,
    MoveEmbedding,
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


class Model(nn.Module):
    def __init__(self, gen: int, embedding_size: int):
        super().__init__()

        species_embedding = PokedexEmbedding(gen=gen)
        self.encode_species_embedding = nn.Sequential(
            species_embedding,
            nn.Linear(species_embedding.embedding_dim, embedding_size),
        )
        ability_embedding = AbilityEmbedding(gen=gen)
        self.encode_ability_embedding = nn.Sequential(
            ability_embedding,
            nn.Linear(ability_embedding.embedding_dim, embedding_size),
        )
        item_embedding = ItemEmbedding(gen=gen)
        self.encode_item_embedding = nn.Sequential(
            item_embedding,
            nn.Linear(item_embedding.embedding_dim, embedding_size),
        )
        move_embedding = MoveEmbedding(gen=gen)
        self.encode_move_embedding = nn.Sequential(
            move_embedding,
            nn.Linear(move_embedding.embedding_dim, embedding_size),
        )
        self.encode_nature_embedding = nn.Embedding(len(NATURES), embedding_size)
        self.encode_ev_spread_embedding = nn.Linear(6, embedding_size)

        self.transformer_encoder = nn.TransformerEncoderLayer(embedding_size, 4)
        self.encoder = nn.TransformerEncoder(self.transformer_encoder, 3)

        self.decode_species_embedding = nn.Linear(
            embedding_size, species_embedding.num_embeddings
        )
        self.decode_ability_embedding = nn.Linear(
            embedding_size, ability_embedding.num_embeddings
        )
        self.decode_item_embedding = nn.Linear(
            embedding_size, item_embedding.num_embeddings
        )
        self.decode_move_embedding = nn.Linear(
            embedding_size, move_embedding.num_embeddings
        )
        self.decode_nature_embedding = nn.Linear(embedding_size, len(NATURES))
        self.decode_ev_spread_embedding = nn.Linear(embedding_size, 6)

    def get_targets(self, data: torch.Tensor):
        data_1 = data.long() + 1

        species_target = data_1[..., 0]
        ability_target = data_1[..., 1]
        item_target = data_1[..., 2]
        moves_target = data_1[..., 3:7]
        nature_target = data[..., 7]
        evs_target = data[..., 8:] / 252

        return (
            species_target,
            ability_target,
            item_target,
            moves_target,
            nature_target,
            evs_target,
        )

    def encode(self, data: torch.Tensor):
        (
            species_target,
            ability_target,
            item_target,
            moves_target,
            nature_target,
            evs_target,
        ) = self.get_targets(data)

        species = self.encode_species_embedding(species_target)
        ability = self.encode_ability_embedding(ability_target)
        item = self.encode_item_embedding(item_target)
        moves = self.encode_move_embedding(moves_target)
        nature = self.encode_nature_embedding(nature_target)
        evs = self.encode_ev_spread_embedding(evs_target)

        pkmn_embedding = species + ability + item + moves.sum(-2) + nature + evs
        pkmn_embedding = self.encoder(pkmn_embedding)
        return pkmn_embedding.mean(1)

    def forward(self, data: torch.Tensor):
        (
            species_target,
            ability_target,
            item_target,
            moves_target,
            nature_target,
            evs_target,
        ) = self.get_targets(data)

        species = self.encode_species_embedding(species_target)
        ability = self.encode_ability_embedding(ability_target)
        item = self.encode_item_embedding(item_target)
        moves = self.encode_move_embedding(moves_target)
        nature = self.encode_nature_embedding(nature_target)
        evs = self.encode_ev_spread_embedding(evs_target)

        pkmn_embedding = species + ability + item + moves.sum(-2) + nature + evs
        pkmn_embedding = self.encoder(pkmn_embedding)
        return pkmn_embedding.mean(1)

        # species = self.decode_species_embedding(pkmn_embedding)
        # ability = self.decode_ability_embedding(pkmn_embedding)
        # item = self.decode_item_embedding(pkmn_embedding)
        # moves = self.decode_move_embedding(pkmn_embedding)
        # nature = self.decode_nature_embedding(pkmn_embedding)
        # evs = self.decode_ev_spread_embedding(pkmn_embedding)

        # return (
        #     (
        #         species,
        #         ability,
        #         item,
        #         moves,
        #         nature,
        #         evs,
        #     ),
        #     (
        #         species_target,
        #         ability_target,
        #         item_target,
        #         moves_target,
        #         nature_target,
        #         evs_target,
        #     ),
        # )
