from pygame.locals import *
import pygame
import sys
import random
import subprocess
import time

mainClock = pygame.time.Clock()

pygame.init()
pygame.display.set_caption('game base')
screen = pygame.display.set_mode((1366, 768), 0, 32)
background = pygame.image.load('c.jpg').convert()
background = pygame.transform.smoothscale(background, screen.get_size())

backgroundR = pygame.image.load('xc.jpg').convert()
backgroundR = pygame.transform.smoothscale(backgroundR, screen.get_size())

bg = pygame.image.load('u.jpg').convert()
bg = pygame.transform.smoothscale(bg, screen.get_size())

font = pygame.font.SysFont('dejavusans', 41)

WIDTH = 800
HEIGHT = 600

# Set up colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)

# Set up fonts
font_small = pygame.font.Font(None, 35)
font_medium = pygame.font.Font(None, 35)
game_over = False

# Luwian words to guess
words_and_hieroglyphs = {
    "Varpalava King": "Varpalava.png",
    "Kurkuma City": "Kurkuma.png",
    "Tuvaranava City": "Tuvaranava.png",
    "Palaa Kingdom": "Palaa.png",
    "Tarkumuva King": "Tarkumuva.png",
    "Khamatu Kingdom": "Khamatu.png"
}
hieroglyph_hints = {
    "VARPALAVA KING": {
        "hint": "Slice of pizza",
        "description": "Capital of California"
    },
    "KURKUMA CITY": {
        "hint": "A with base",
        "description": "Rules over subjects"
    },
    "TUVARANAVA CITY": {
        "hint": "The vertical divide",
        "description": "Moses guides water"
    },
    "PALAA KINGDOM": {
        "hint": "Pot",
        "description": "Abbr for personal assistant"
    },
    "TARKUMUVA KING": {
        "hint": "The feet",
        "description": "In words like 'table' & 'tango' you'll find me"
    },
    "KHAMATU KINGDOM": {
        "hint": "Kite",
        "description": "The cuckoo goes..."
    }, }

word = random.choice(list(words_and_hieroglyphs.keys()))
hieroglyph_filename = words_and_hieroglyphs[word]
hieroglyph = pygame.image.load(hieroglyph_filename)
word = word.upper()
# Word length
word_length = len(word)


# Set up the list of correctly guessed letters
correct_letters = ["_"] * word_length

# Set up the list to track used letters
used_letters = []

# Number of tries allowed
tries_left = 6

# Main game loop
running = True
game_over = False


def draw_text(text, font, color, surface, x, y):
    textobj = font.render(text, 1, color)
    textrect = textobj.get_rect()
    textrect.topleft = (x, y)
    surface.blit(textobj, textrect)


def button(screen, position, text, size="small"):
    if size == "small":
        font = pygame.font.SysFont("Arial", 50)
        padding = 37
    else:
        font = pygame.font.SysFont("Arial", 70)
        padding = 140

    text_render = font.render(text, 1, (0, 0, 0))
    x, y = position
    w, h = text_render.get_width() + padding, text_render.get_height() + 5
    pygame.draw.line(screen, (150, 150, 150), (x, y), (x + w, y), 5)
    pygame.draw.line(screen, (150, 150, 150), (x, y - 2), (x, y + h), 5)
    pygame.draw.line(screen, (50, 50, 50), (x, y + h), (x + w, y + h), 5)
    pygame.draw.line(screen, (50, 50, 50), (x + w, y + h), [x + w, y], 5)
    pygame.draw.rect(screen, (150, 150, 150), (x, y, w, h))
    return screen.blit(text_render, (x, y))


def main_menu():
    while True:
        screen.blit(background, (0, 0))
        draw_text('Main Menu', font, (255, 255, 255), screen, 600, 20)

        mx, my = pygame.mouse.get_pos()

        button_1 = button(screen, (40, 700), "    Rules", "small")
        button_2 = button(screen, (500, 687), "           Play", "large")
        button_3 = button(screen, (1200, 703), "   Exit", "small")
        button_4 = button(screen, (1200, 643), "   Reset", "small")

        if button_1.collidepoint((mx, my)):
            if click:
                game()
        if button_2.collidepoint((mx, my)):
            if click:
                options()
        if button_3.collidepoint((mx, my)):
            if click:
                sys.exit()
        if button_4.collidepoint((mx, my)):
            if click:

                reset()

        click = False
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    pygame.quit()
                    sys.exit()
            if event.type == MOUSEBUTTONDOWN:
                if event.button == 1:
                    click = True

        pygame.display.update()
        pygame.display.flip()
        mainClock.tick(60)


def game():
    running = True
    while running:
        screen.blit(backgroundR, (0, 0))
        draw_text('                          Welcome to an interactive game on the hieroglyphs of ancient Luwia', font,
                  (3, 252, 98), screen, 20, 300)
        draw_text('                                                                  Make your mark here:', font,
                  (3, 252, 98), screen, 20, 330)
        draw_text('             Play mini games on each level to unlock the riddles you need to solve your ultimate task,',
                  font, (3, 252, 98), screen, 20, 368)
        draw_text('                                             To decipher what the Luwian hieroglyphs stand for.', font,
                  (3, 252, 98), screen, 20, 410)
        draw_text('                     Embark on a quest to explore the culture and improve your cognitive abilities!',
                  font, (3, 252, 98), screen, 20, 460)
        draw_text('                                                                    Press Esc To Exit                            ',
                  font, (3, 252, 98), screen, 20, 510)

        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    running = False

        pygame.display.update()
        mainClock.tick(60)


invis = (21, 33, 49)


def trigger_function():
    global font_color
    font_color = invis


def options():
    global game_over, running
    running = True
    hint_window_process = None
    tries_left = 6
    display_riddle = False
    hint_req = 0
    GAME_STATE_PLAYING = "playing"
    # GAME_STATE_HINT = "hint"
    game_state = GAME_STATE_PLAYING

    while running:
        screen.blit(bg, (0, 0))
        if "_" not in correct_letters:
            running = False
            continue
        else:
            draw_text('Press One for Hint', font,
                      (255, 255, 255), screen, 530, 500)
            screen.blit(hieroglyph, (550, 165))  # Pic
            text = font_medium.render(" ".join(correct_letters), True, WHITE)
            # _ " input part"
            screen.blit(text, (550, HEIGHT // 2 - text.get_height()+65))
            # hint=False

            text = font_small.render(
                "Used Letters: " + ", ".join(used_letters), True, WHITE)
            screen.blit(text, (550, 275))  # Text

            text = font_small.render(
                "Tries Left: " + str(tries_left), True, WHITE)
            screen.blit(text, (550, 310))  # Tries left
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_1:
                        # Change game state to hint mode
                        hint_req = 1

                    elif event.key == K_ESCAPE:
                        running = False
                    elif event.unicode.isalpha() or event.unicode.isspace():
                        letter = event.unicode.upper()
                        if letter not in used_letters:
                            used_letters.append(letter)
                            # if letter in word:
                            letter_found = False
                            for i in range(len(word)):
                                if word[i] == letter:
                                    correct_letters[i] = letter
                                    letter_found = True
                                    # continue

                            if not letter_found:
                                tries_left -= 1

            if tries_left == 0:
                screen.blit(bg, (0, 0))
                text = font.render(
                    "Game Over! " + " The word was " + word, True, RED)
                screen.blit(text, (WIDTH // 2 - text.get_width() +
                            500, HEIGHT // 2 - text.get_height()+200))

                game_over = True
                running = False

            elif "_" not in correct_letters:
                screen.blit(bg, (0, 0))
                text = font.render("You Win!", True, RED)
                screen.blit(text, (WIDTH // 2 - text.get_width() +
                            300, HEIGHT // 2 - text.get_height()+200))
                # text = font_medium.render(
                #     " ".join(correct_letters), True, WHITE)
                # # _ " input part"
                # screen.blit(text, (550, HEIGHT // 2 - text.get_height()+65))
                game_over = True
                # running = False

            pygame.display.flip()

            if hint_req == 1:
                # Choose a random hint game file
                hint_game_file = random.choice([
                    "Wormy.py",
                    "Tic_Tac_Toe.py",
                    "Tetris.py",
                    "Simon.py",
                    "Memory_Game.py",
                    "Dinosaur.py"
                ])

                # Open the hint game file
                hint_window_process = subprocess.Popen(
                    ['python', hint_game_file])
                hint_req = 0

            if hint_window_process is not None and hint_window_process.poll() is not None:
                # Subprocess completed
                # hint_completed = True
                return_code = hint_window_process.returncode

                # Display the riddle after the subprocess window is closed
                if return_code == 0:
                    display_riddle = True

            if display_riddle == True:
                if word in hieroglyph_hints:
                    hint = hieroglyph_hints[word]["description"]
                    hint_text = font_small.render(
                        "Hint: " + hint, True, WHITE)
                    screen.blit(hint_text, (530, 600))

            pygame.display.update()
            pygame.time.delay(1000)
            mainClock.tick(60)


def reset():
    global word, word_length, correct_letters, used_letters, tries_left, game_over, hieroglyph
    word = random.choice(list(words_and_hieroglyphs.keys()))
    hieroglyph_filename = words_and_hieroglyphs[word]
    hieroglyph = pygame.image.load(hieroglyph_filename)
    word = word.upper()
    word_length = len(word)
    correct_letters = ["_"] * word_length
    used_letters = []
    tries_left = 6
    game_over = False


def exit():
    sys.exit()


main_menu()
