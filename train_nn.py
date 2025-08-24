from game_loader import GameLoader
from es_manager import ESManager
import config, nn_config as NN

def main():
    loader = GameLoader(config.DATA_FILE)
    all_cards  = getattr(loader, "all_cards", loader.cards)
    all_nobles = getattr(loader, "all_nobles", loader.nobles)
    ESManager(all_cards, all_nobles).run(generations=200)

if __name__ == "__main__":
    main()