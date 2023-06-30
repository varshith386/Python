import pygame
import random

# Initialize Pygame
pygame.init()

# Set up the display
WIDTH = 800
HEIGHT = 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Pac-Man Game")

# Define colors
BLACK = (0, 0, 0)
YELLOW = (255, 255, 0)

# Define game variables
score = 0
hint_unlocked = False
game_over = False

# Set up Pac-Man and the food
pacman_image = pygame.image.load("pacman.png")
pacman_rect = pacman_image.get_rect()
pacman_rect.center = (WIDTH // 2, HEIGHT // 2)

food_list = []
for _ in range(10):
    food = pygame.Rect(random.randint(0, WIDTH - 20),
                       random.randint(0, HEIGHT - 20), 20, 20)
    food_list.append(food)

# Set up the game clock
clock = pygame.time.Clock()

# Game loop
while not game_over:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            game_over = True

    # Move Pac-Man
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        pacman_rect.x -= 5
    elif keys[pygame.K_RIGHT]:
        pacman_rect.x += 5
    elif keys[pygame.K_UP]:
        pacman_rect.y -= 5
    elif keys[pygame.K_DOWN]:
        pacman_rect.y += 5

    # Check for collisions with food
    for food in food_list:
        if pacman_rect.colliderect(food):
            food_list.remove(food)
            score += 1
            if score == 5:
                hint_unlocked = True

    # Clear the screen
    screen.fill(BLACK)

    # Draw Pac-Man and food
    screen.blit(pacman_image, pacman_rect)
    for food in food_list:
        pygame.draw.ellipse(screen, YELLOW, food)

    # Display the score
    font = pygame.font.Font(None, 36)
    score_text = font.render("Score: " + str(score), True, YELLOW)
    screen.blit(score_text, (10, 10))

    # Display hint or game result
    if hint_unlocked:
        hint_text = font.render("You Unlocked the Hint!", True, YELLOW)
        screen.blit(hint_text, (10, 50))
        game_over = True
    elif score >= 10:
        game_over = True
        result_text = font.render("You Won! Game Over", True, YELLOW)
        screen.blit(result_text, (WIDTH // 2 - 150, HEIGHT // 2))

    # Update the display
    pygame.display.flip()

    # Set the frames per second
    clock.tick(60)

# Quit the game
pygame.quit()
