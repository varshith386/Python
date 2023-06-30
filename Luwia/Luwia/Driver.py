import pygame
from pygame.locals import *
import random

# Initialize Pygame
pygame.init()

# Game window dimensions
WIDTH = 800
HEIGHT = 600

# Set the game window
window = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Luwian Puzzle")

# Set up colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)

# Set up fonts
font_small = pygame.font.Font(None, 24)
font_medium = pygame.font.Font(None, 36)

# Luwian words to guess
words = ["KUWAPIYA", "TARUWANNA", "HULIYAN", "KUWATLIYAN", "KUWANNA"]
word = random.choice(words)
word_length = len(word)

# Set up the list correctly guessed letters
correct_letters = ["_"] * word_length

# Set up the list to track used letters
used_letters = []

# Number of tries allowed
tries_left = 6

# Luwian hieroglyph
hieroglyph = pygame.image.load("hiero.jpg")
hieroglyph_rect = hieroglyph.get_rect()
hieroglyph_rect.x = 10
hieroglyph_rect.y = 10

# Puzzles/riddles
puzzles = ["What has a heart that doesn't beat?",
           "What has a head and a tail but no body?"]
answers = ["Artichoke", "Coin"]
current_puzzle = 0

# Main game loop
running = True
game_over = False
while running:
    window.fill(BLACK)

    # Draw the Luwian hieroglyph on the left side of the game window
    window.blit(hieroglyph, hieroglyph_rect)

    # Draw the correctly guessed letters on the left side of the game window
    text = font_medium.render(" ".join(correct_letters), True, WHITE)
    window.blit(text, (10, HEIGHT // 2 - text.get_height() // 2))

    # Draw the used letters on the left side of the game window
    text = font_small.render(
        "Used Letters: " + ", ".join(used_letters), True, WHITE)
    window.blit(text, (10, 200))

    # Draw the number of tries left on the left side of the game window
    text = font_small.render("Tries Left: " + str(tries_left), True, WHITE)
    window.blit(text, (10, 230))

    # Draw the current puzzle/riddle on the right side of the game window
    text = font_small.render(puzzles[current_puzzle], True, WHITE)
    window.blit(text, (WIDTH // 2 + 10, 10))

    # Draw the answer to the current puzzle/riddle on the right side of the game window
    if pygame.key.get_pressed()[pygame.K_SPACE]:
        text = font_small.render(
            "Answer: " + answers[current_puzzle], True, WHITE)
        window.blit(text, (WIDTH // 2 + 10, 40))

    # Check for game over conditions
    if tries_left == 0:
        text = font_medium.render("Game Over! The word was " + word, True, RED)
        window.blit(text, (WIDTH // 2 - text.get_width() //
                    2, HEIGHT // 2 - text.get_height() // 2))
        game_over = True
    elif correct_letters == list(word):
        text = font_medium.render("You Win!", True, RED)
        window.blit(text, (WIDTH // 2 - text.get_width() //
                    2, HEIGHT // 2 - text.get_height() // 2))
        game_over = True

    pygame.display.flip()

    # Delay after displaying the game result message
    if game_over:
        pygame.time.delay(2000)  # Delay for 2000 milliseconds (2 seconds)
        break

    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.unicode.isalpha():
                letter = event.unicode.upper()
                if letter not in used_letters:
                    used_letters.append(letter)
                    if letter in word:
                        for i in range(word_length):
                            if word[i] == letter:
                                correct_letters[i] = letter
                    else:
                        tries_left -= 1
            elif event.key == pygame.K_RETURN:
                # Move to the next puzzle/riddle
                current_puzzle = (current_puzzle + 1) % len(puzzles)

# Quit Pygame
pygame.quit()
