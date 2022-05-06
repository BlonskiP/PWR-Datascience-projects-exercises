

class GameManager(ABC):
    "GameManager is responsible for game loop etc"

    def __init__(self):
        self.is_playing = False
        self.init_game()
        pass

    @abstractmethod
    def init_game(self):
        pass

    @abstractmethod
    def game_actions(self):
        pass

    def game_loop(self):
        while self.is_playing:
            self.game_actions()

    def start_game(self):
        self.init_game()
        self.is_playing = True
        self.game_loop()

    @abstractmethod
    def save_game_state(self):
        pass

    @abstractmethod
    def load_game_state(self,state):
        pass