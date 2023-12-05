import copy
import math
import random
import itertools
from exceptions import CoalitionNotFoundException


class Coalition:
    def __init__(self, members: set, value: float):
        self.members = members
        self.value = value
        self.size = len(members)

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"{self.members}: {self.value}"

    def contains_player(self, player):
        return player in self.members

    def equals(self, coalition):
        return coalition.members == self.members


class Game:
    def __init__(self):
        self.coalitions = dict()
        self.players = None
        self.number_of_players = 0
        self.min_value = 0
        self.max_value = 0

    def __str__(self):
        return f"{self.players} - {[coal for coal in self.coalitions]}"

    def get_existing_coalition_value(self, new_coalition_members):
        if frozenset(new_coalition_members) in self.coalitions.keys():
            return self.coalitions[frozenset(new_coalition_members)]
        return None

    def add_coalition(self, members, coalition_value):
        self.coalitions[frozenset(members)] = coalition_value

    def update_players(self):
        self.players = set()
        for c in self.coalitions:
            self.players.update(c.members)
        self.number_of_players = len(self.players) - 1  # remove None as a player

    def get_coalition(self, members: set):
        # Do not compute value with None as a player
        if None in members:
            members.remove(None)
        # After removing a player there could be an empty list, which means the contribution of no player is added
        if len(members) == 0:
            members = {None}
        for coalition in self.coalitions:
            if coalition.members == members:
                return coalition
        raise CoalitionNotFoundException(f"Coalition with members {members} not found")

    def normalize(self):
        for coalition in self.coalitions:
            if coalition.value < self.min_value:
                self.min_value = coalition.value
            if coalition.value > self.max_value:
                self.max_value = coalition.value
        for coalition in self.coalitions:
            coalition.value = (coalition.value - self.min_value) / (self.max_value - self.min_value)


def create_test_game():
    coalitions = []
    a = Coalition({"a"}, 12)
    b = Coalition({"b"}, 6)
    c = Coalition({"c"}, 9)
    ab = Coalition({"a", "b"}, 24)
    ac = Coalition({"a", "c"}, 27)
    bc = Coalition({"b", "c"}, 15)
    abc = Coalition({"a", "b", "c"}, 36)
    none_val = Coalition({None}, 0)
    coalitions.extend([a, b, c, ab, ac, bc, abc, none_val])
    g = Game()
    for c in coalitions:
        g.add_coalition(c)
    g.update_players()
    return g

def create_test_game2():
    g = Game()
    g.players = {"a", "b", "c"}
    return g

def value_function(s: set):
    if s == {"a"}:
        return Coalition({"a"}, 12)
    elif s == {"b"}:
        return Coalition({"b"}, 6)
    elif s == {"c"}:
        return Coalition({"c"}, 9)
    elif s == {"a", "b"}:
        return Coalition({"a", "b"}, 24)
    elif s == {"a", "c"}:
        return Coalition({"a", "c"}, 27)
    elif s == {"b", "c"}:
        return Coalition({"b", "c"}, 15)
    elif s == {"a", "b", "c"}:
        return Coalition({"a", "b", "c"}, 36)
    elif s == set():
        return Coalition({None}, 0)


def calculate_shapley_value(game, debug=False):
    shapley_values = dict()
    for player in game.players:
        # No need to treat None as a player
        if player is not None:
            overall_contribution = 0
            if debug:
                print(f"Calculating Shapley value for player: {player}")
            # Sum over all coalitions
            for coalition in game.coalitions:
                if debug:
                    print(f"Coalition: {coalition}")
                # First compute weighting for marginal contribution
                possible_coalitions = (math.factorial(coalition.size - 1) * math.factorial(
                    game.number_of_players - coalition.size)) / math.factorial(game.number_of_players)
                # print(f"Poss. coals: {possible_coalitions}")
                coalition_members_without_player = copy.deepcopy(coalition.members)
                if player in coalition_members_without_player:
                    coalition_members_without_player.remove(player)
                # Compute marginal contribution
                contribution = coalition.value - game.get_coalition(coalition_members_without_player).value
                # print(f"Contribution: {contribution}")
                # Add up weighted marginal contribution
                overall_contribution += (possible_coalitions * contribution)
            if debug:
                print(f"Overall contribution of player {player} = {overall_contribution}")
                print("##################################################################")
            shapley_values[player] = overall_contribution
    return shapley_values


if __name__ == "__main__":
    game = create_test_game2()