import numpy as np
from scipy.signal import convolve2d
import time
from scipy.ndimage import convolve


class Game:
    def __init__(self):
        self.ligne=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O']
        self.taille_plateau = 15
        self.longueur_victoire = 5
        self.centre = (self.taille_plateau // 2, self.taille_plateau // 2)

        self.mode_jj = 1
        self.mode_jo = 2

        self.joueur_1 = 1
        self.joueur_2 = 2

        self.plateau = Plateau(self.taille_plateau)

        # Matrice pour détecter les coups adjacents
        self.matriceadj = np.array([[1, 1, 1],
                                  [1, 0, 1],
                                  [1, 1, 1]])
                                  
        # Patterns avec scores et masques
        self.patterns = {
                        'cinq': {'seq': [1,1,1,1,1], 'score': 1000000000000000000},
                        'quatre_ouvert': {'seq': [0,1,1,1,1,0], 'score': 10000000},
                        'quatre_semi':{'seq': [1,0,1,1], 'score': 1000000},  
                        'quatre_semi2': {'seq': [1,1,0,1], 'score': 1000000},  
                        'quatre_droite': {'seq': [1,1,1,1,0], 'score': 5000}, 
                        'quatre_gauche': {'seq': [0,1,1,1,1], 'score': 5000},
                        'trois_ouvert': {'seq': [0,1,1,1,0], 'score': 1000},
                        'trois_gauche': {'seq': [0,1,1,1], 'score': 100},  
                        'trois_droite': {'seq': [1,1,1,0], 'score': 100},
                        'deux_ouvert': {'seq': [0,1,1,0], 'score': 10} 

        }
       
        # Masques de détection pour la victoire
        self.masques = {
            5: [
                np.ones((1, self.longueur_victoire)),
                np.ones((self.longueur_victoire, 1)),
                np.eye(self.longueur_victoire),
                np.fliplr(np.eye(self.longueur_victoire))
            ]
        }
        
        # Directions d'analyse
        self.directions = [
            (1, 0),   # horizontal
            (0, 1),   # vertical
            (1, 1),   # diagonal principal
            (1, -1)   # diagonal inverse
        ]
        
        # Table de transposition pour l'optimisation
        self.transposition_table = {}
        
        # Initialisation du jeu
        self.Init_Game()

    # Initialisation du jeu
    def Init_Game(self):
        print("Bienvenue dans le jeu de Gomoku !\n")

        while True:
            print("Veuillez sélectionner le mode de jeu :")
            print("    -Joueur VS Joueur (jj)")
            print("    -Joueur VS Ordinateur (jo)")

            mode_input = input("Entrez jj ou jo (quit pour arrêter le jeu) : \n").lower()

            if mode_input == "quit":
                return

            if mode_input == "jj":
                self.mode = self.mode_jj
                break

            if mode_input == "jo":
                self.mode = self.mode_jo
                while True:
                    print("\nVeuillez sélectionner la priorité de jeu :")
                    print("    -Je commence (j1)")
                    print("    -Je seconde (j2)")

                    prio_temp = input("Entrez j1 ou j2 : \n").lower()

                    if prio_temp == "j1":
                        prio = self.joueur_1
                        break
                    elif prio_temp == "j2":
                        prio = self.joueur_2
                        break
                    elif prio_temp == "quit":
                        return

                    print("Choix non valide. Veuillez réessayer.\n")
                self.who_is_playing = prio
                break

            print("Mode de jeu non existant. Veuillez réessayer.\n")
        self.Play()

    # Lancement du jeu
    def Play(self):
        nb_turn = 1
        joueur = 2 if self.mode == self.mode_jo and self.who_is_playing == self.joueur_2 else 1
        taille_plateau_total = self.taille_plateau * self.taille_plateau

        while nb_turn <= taille_plateau_total:
            if self.check_winner():
                break

            self.Turn(joueur, nb_turn)

            joueur = 3 - joueur  # Alternance 1-2
            nb_turn += 1

        winner = self.check_winner()
        self.GameOver(winner)

    # Fin du jeu
    def GameOver(self, winner):
        print(self.plateau)
        if winner == 0:
            print("Match nul !")
        else:
            print(f"Le joueur {winner} a gagné !")

    # Vérification du gagnant
    def check_winner(self):
        for mask in self.masques[5]:
            conv_result = convolve2d(self.plateau.get_plateau() == 1, mask, mode="valid")
            if np.any(conv_result == self.longueur_victoire):
                return 1
            conv_result = convolve2d(self.plateau.get_plateau() == 1j, mask, mode="valid")
            if np.any(conv_result == self.longueur_victoire):
                return 2
        return 0

    # Tour de jeu
    def Turn(self, joueur, nb_Turn):
        action_messages = {
            (self.mode_jj, 1): "C'est au Joueur 1 de jouer",
            (self.mode_jj, 2): "C'est au Joueur 2 de jouer",
            (self.mode_jo, 1): "C'est à vous de jouer",
            (self.mode_jo, 2): "C'est à l'IA de jouer"
        }

        print(f"Tour : {nb_Turn-1}")
        print(self.plateau)
        print(action_messages[(self.mode, joueur)])

        if joueur == 2 and self.mode == self.mode_jo:
            if nb_Turn==1:
                position = self.centre
            elif nb_Turn==3 and self.plateau.get_plateau()[3,7]==0:
                position = (3,7)
            elif nb_Turn==3 and self.plateau.get_plateau()[11,7]==0:
                position = (11,7)
            else:
                position = self.minimax(self.plateau.get_plateau(), joueur, nb_Turn)
        else:
            position = self.get_player_move(nb_Turn)

        self.plateau.set_position(position, joueur)
        print("position jouée : ", (self.ligne[position[0]],position[1]))

    # Récupération du coup du joueur
    def get_player_move(self, nb_Turn):
        actions_possibles = self.actions(self.plateau.get_plateau(), nb_Turn)
        if nb_Turn == 1:
            print(f"Vous ne pouvez jouer qu'au centre : {((self.taille_plateau // 2) + 1, (self.taille_plateau // 2) + 1)}")
            return self.centre
        elif nb_Turn == 3:
            print("Vous pouvez jouer dans les cases situées en dehors du carré central 7x7, \nc'est-à-dire sur les lignes et colonnes avant la ligne E ou après la ligne J, \nainsi que sur les colonnes avant la colonne 5 ou après la colonne 10.")

        while True:
            try:
                ligne_input = input("Entrer la ligne (A à O) : ").upper()
                if len(ligne_input) != 1 or ligne_input < 'A' or ligne_input > 'O':
                    raise ValueError("Ligne invalide. Veuillez entrer une lettre entre A et O.")

                colonne_input = input("Entrer la colonne (0 à 14) : ")
                colonne = int(colonne_input) 

                ligne = ord(ligne_input) - ord('A')

                if (ligne, colonne) not in actions_possibles:
                    raise ValueError("Position invalide ou déjà occupée. Réessayez.")

                return (ligne, colonne)

            except (ValueError, IndexError):
                print("Position invalide ou déjà occupée. Réessayez.")
    
    # Résultat du coup
    def result(self, board, joueur, position):
        new_board = board.copy()
        val = 1j if joueur == 2 else joueur
        new_board[position[0], position[1]] = val
        return new_board

    # Actions possibles
    def actions(self, board, nb_Turn, bot=0):
        if nb_Turn == 1:
            actions_possibles = [self.centre]
        elif nb_Turn == 3:
            actions_possibles = [(i, j) for i in range(4,11) for j in range(4,11) if board[i, j] == 0]
            return [(i, j) for i in range(self.taille_plateau) for j in range(self.taille_plateau) if (i,j) not in actions_possibles and board[i, j] == 0]
        else:
            if bot == 1:
                boardcandidats = convolve((board != 0).astype(int), self.matriceadj, mode="constant", cval=0)
                actions_possibles = [(i, j) for i in range(boardcandidats.shape[0]) for j in range(boardcandidats.shape[1]) if boardcandidats[i, j] != 0 and board[i, j] == 0]
            else:
                actions_possibles = [(i, j) for i in range(self.taille_plateau) for j in range(self.taille_plateau) if board[i, j] == 0]
        return actions_possibles
    
    # Vérification des patterns
    def check_pattern(self, board, pattern, player_val):
        pattern = np.array(pattern)
        count = 0
        for dx, dy in self.directions:
            for i in range(self.taille_plateau):
                for j in range(self.taille_plateau):
                    matches = True
                    for k in range(len(pattern)):
                        ni = i + k*dx
                        nj = j + k*dy
                        if not (0 <= ni < self.taille_plateau and 
                               0 <= nj < self.taille_plateau):
                            matches = False
                            break
                        if ((pattern[k] == 1 and board[ni,nj] != player_val) or 
                            (pattern[k] == 0 and board[ni,nj] != 0)):
                            matches = False
                            break
                    if matches:
                        count += 1
        return count

    # Évaluation du plateau
    def evaluate_board(self, board, joueur):
        val_joueur = 1j if joueur == 2 else 1
        val_adversaire = 1 if joueur == 2 else 1j
        score = 0

        # Vérification des patterns pour les deux joueurs
        for pattern_info in self.patterns.values():
            # Score pour nos patterns
            count_own = self.check_pattern(board, pattern_info['seq'], val_joueur)
            score += count_own * pattern_info['score']
            
            # Score pour l'adversaire (défense plus importante)
            count_opp = self.check_pattern(board, pattern_info['seq'], val_adversaire)
            score -= count_opp * pattern_info['score'] * 1.5

        # Bonus pour le contrôle du centre
        centre_x, centre_y = self.taille_plateau // 2, self.taille_plateau // 2
        for i in range(self.taille_plateau):
            for j in range(self.taille_plateau):
                if board[i,j] == val_joueur:
                    distance_centre = abs(i - centre_x) + abs(j - centre_y)
                    score += max(0, (self.taille_plateau - distance_centre)) * 10

        return score

    # Sélection des meilleurs coups
    def get_best_moves(self, board, joueur, nb_Turn, top_n=7):
            moves = self.actions(board, nb_Turn, 1)
            move_scores = []
            
            for move in moves:
                # Simulation du coup
                new_board = self.result(board, joueur, move)
                
                # Vérification immédiate de la victoire
                if self.is_winning_move(new_board, move, joueur):
                    return [move]
                
                # Vérification du blocage d'une victoire adverse
                if self.blocks_win(board, move, 3-joueur):
                    return [move]
                
                # Évaluation normale
                score = self.evaluate_board(new_board, joueur)
                move_scores.append((move, score))

            # Tri et randomisation légère pour la variété
            best_moves = sorted(move_scores, key=lambda x: x[1], reverse=True)[:top_n]
            return [move for move, _ in best_moves]

    # Vérification d'un coup gagnant
    def is_winning_move(self, board, move, joueur):
        val = 1j if joueur == 2 else 1
        for dx, dy in self.directions:
            count = 1
            
            for step in [-1, 1]:
                i, j = move[0], move[1]
                while True:
                    i += dx * step
                    j += dy * step
                    if not (0 <= i < self.taille_plateau and 
                           0 <= j < self.taille_plateau):
                        break
                    if board[i,j] != val:
                        break
                    count += 1
                    if count >= 5:
                        return True
        return False

    # Vérification du blocage d'une victoire adverse
    def blocks_win(self, board, move, opponent):
        test_board = self.result(board, opponent, move)
        return self.is_winning_move(test_board, move, opponent)

    # Algorithme minimax avec élagage alpha-beta
    def minimax(self, board, joueur, nb_Turn, depth=5):
        start_time = time.time()
        alpha = float('-inf')
        beta = float('inf')

        moves = self.get_best_moves(board, joueur, nb_Turn)
        best_move = moves[0]
        board_hash = str(board)

        try:
            if board_hash in self.transposition_table:
                stored_depth, stored_move = self.transposition_table[board_hash]
                if stored_depth >= depth:
                    return stored_move

            for move in moves:
                if time.time() - start_time > 4.5:
                    break

                new_board = self.result(board, joueur, move)
                value = self.min_value(new_board, 3 - joueur, nb_Turn + 1, depth - 1, alpha, beta, start_time)

                if value > alpha:
                    alpha = value
                    best_move = move
                    self.transposition_table[board_hash] = (depth, move)

        except TimeoutError:
            return best_move
        print(time.time() - start_time)
        return best_move

    # Fonction de maximisation pour Minimax
    def max_value(self, board, joueur, nb_Turn, depth, alpha, beta, start_time):
        if time.time() - start_time > 4.5:
            return self.evaluate_board(board, joueur)

        if depth == 0 or self.check_winner() != 0:
            return self.evaluate_board(board, joueur)

        v = float('-inf')
        moves = self.get_best_moves(board, joueur, nb_Turn)

        for move in moves:
            if time.time() - start_time > 4.5:
                return v
            else:
                new_board = self.result(board, joueur, move)
                v = max(v, self.min_value(new_board, 3 - joueur, nb_Turn + 1, depth - 1, alpha, beta, start_time))
                if v >= beta:
                    return v
            alpha = max(alpha, v)
        return v

    # Fonction de minimisation pour Minimax
    def min_value(self, board, joueur, nb_Turn, depth, alpha, beta, start_time):
        if time.time() - start_time > 4.5:
            return self.evaluate_board(board, joueur)

        if depth == 0 or self.check_winner() != 0:
            return self.evaluate_board(board, joueur)

        v = float('inf')
        moves = self.get_best_moves(board, joueur, nb_Turn)

        for move in moves:
            if time.time() - start_time > 4.5:
                return v
            else:
                new_board = self.result(board, joueur, move)
                v = min(v, self.max_value(new_board, 3 - joueur, nb_Turn + 1, depth - 1, alpha, beta, start_time))
                if v <= alpha:
                    return v
            beta = min(beta, v)
        return v

class Plateau:
    def __init__(self, taille=15):
        self.plateau = np.zeros((taille, taille), dtype=complex)

    # Récupération du plateau
    def get_plateau(self):
        return self.plateau

    # Positionnement sur le plateau
    def set_position(self, pos, val):
        if val == 2:
            val = 1j
        self.plateau[pos[0], pos[1]] = val

    # Affichage du plateau
    def __str__(self):
        cellule = {0: " ", 1: "X", 1j: "O"}
        rep = "    "
        for i in range(len(self.plateau[0])):
            rep += f" {i :<3}"
        
        rep += "\n"
        i = 0
        string = "ABCDEFGHIJKLMNO"
        for ligne in self.plateau:
            rep += "    +" + "---+" * len(ligne) + "\n"
            rep += f"{string[i]}   "
            for cell in ligne:
                rep += f"| {cellule[cell]} "
            rep += "|\n"
            i += 1

        rep += "    +" + "---+" * len(self.plateau[0]) + "\n"
        return rep

def main():
    game = Game()

if __name__ == "__main__":
    main()