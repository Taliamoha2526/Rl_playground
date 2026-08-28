import random
import pandas as pd

class SystemOpponent:
    def __init__(self, characters_df, secret_character_name=None):
        self.df = characters_df
        if secret_character_name and secret_character_name in self.df.index:
            self.secret_name = secret_character_name
        else:
            self.secret_name = random.choice(self.df.index)

        self.secret_character = self.df.loc[self.secret_name]

    def answer(self, feature: str, value) -> bool:
        return self.secret_character[feature] == value

    def reveal(self) -> pd.Series:
        return self.secret_character