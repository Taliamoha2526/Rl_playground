import os
import random
import pandas as pd
def generate_characters(n, save_csv = True):
    """ Function to generate a character space based on combinations of features values.
    parameters:
    n: number of characters to generate
    save_csv: save the character space to a csv file
    returns: df of characters"""
    # Define possible values for each feature
    features = {
        "Sex": ["male", "female"],
        "Region": ["Europe", "Asia", "North America", "South America", "Africa", "Australia"],
        "Age": ["child", "teen", "young", "middle aged", "senior"],
        "Hair color": ["black", "blonde", "brunette", "ginger", "bald", "wigs"],
        "Industry": ["sports", "music", "film", "science", "politics", "entrepreneur", "content", "fashion", "technology", "manufacturing"]
    }
    random.seed(0)
    # Generate n random characters
    data = []
    if n == "max":
        n = (len(features["Sex"]) * len(features["Region"]) * len(features["Age"]) * len(features["Hair color"]) * len(features["Industry"]))
    for i in range(n):
        character = {feature: random.choice(values) for feature, values in features.items()}
        data.append(character)

    # Convert to DataFrame
    df = pd.DataFrame(data)
    df.index = [f"Character {i + 1}" for i in range(n)]
    if save_csv:
        os.makedirs("data", exist_ok=True)
        csv_path = os.path.join("data", "characters.csv")
        df.to_csv(csv_path)
    return df
