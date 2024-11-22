import random

class Poker:
    def __init__(self):
        self.suits = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
        self.ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'Jack', 'Queen', 'King', 'Ace']
        self.deck = [f"{rank} of {suit}" for suit in self.suits for rank in self.ranks]
        self.players = {}

    def shuffle_deck(self):
        random.shuffle(self.deck)

    def deal_cards(self, num_players=5, cards_per_player=2):
        self.shuffle_deck()
        if num_players * cards_per_player > len(self.deck):
            raise ValueError("Not enough cards in the deck to deal!")
        
        for player_id in range(1, num_players + 1):
            self.players[f"Player {player_id}"] = [self.deck.pop() for _ in range(cards_per_player)]

    def display_hands(self):
        for player, hand in self.players.items():
            print(f"{player}: {', '.join(hand)}")

# Example usage
poker_game = Poker()
poker_game.deal_cards()
print(poker_game.display_hands())


