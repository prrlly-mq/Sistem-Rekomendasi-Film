import os
import pandas as pd
from dotenv import load_dotenv

from data.film.llm_film_module import create_chatbot


def run_cli_chatbot():
    load_dotenv()

    print("🔄 Loading dataset...")

    film_df = pd.read_csv(
        "data/film/AllMovies_CLEANED.csv"
    )

    chatbot = create_chatbot(film_df)

    print("🎬 Chatbot Film siap!")
    print("Ketik 'exit' untuk keluar.\n")

    while True:
        msg = input("You: ")
        if msg.lower() == "exit":
            print("👋 Sampai jumpa!")
            break

        result = chatbot.chat(msg)
        print("Bot:", result["text"])


if __name__ == "__main__":
    run_cli_chatbot()