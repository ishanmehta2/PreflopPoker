import pandas as pd


class PreflopOdds:
    def __init__(self, filepath):
        self.odds_data = pd.read_csv(filepath)
        # print(self.odds_data)
        self.odds_dict = self.create_odds_dict()
    
    def create_odds_dict(self):
        odds_dict = {}
        for _, row in self.odds_data.iterrows():
            odds_dict[row['Name']] = row['Overall %']
        return odds_dict

    def get_hand_odds(self, hand):
        return self.odds_dict.get(hand, None)

def main():
    preflopOdds = PreflopOdds("/Users/ishan/Downloads/preflop_odds.csv")
    print(preflopOdds.get_hand_odds('AKs'))

if __name__ == "__main__":
    main()