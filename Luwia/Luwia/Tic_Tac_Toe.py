import pygame
import random
import time

# Initialize Pygame
pygame.init()

# Set the dimensions of the game window
WINDOW_WIDTH = 600
WINDOW_HEIGHT = 600

# Set the colors for the game
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

# Set the font for the game
FONT = pygame.font.SysFont('Arial', 60)

# Create the game window
window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption('Tic Tac Toe')

# Create the game board
board = [['', '', ''], ['', '', ''], ['', '', '']]

# Set the player turn
player = 'X'

# Set the game loop
running = True
winner = None  # Variable to store the winner
while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            # Get the position of the mouse click
            x, y = pygame.mouse.get_pos()
            # Determine the row and column of the click
            row = y // (WINDOW_HEIGHT // 3)
            col = x // (WINDOW_WIDTH // 3)
            # Check if the click is valid
            if board[row][col] == '':
                # Update the board
                board[row][col] = player
                # Switch the player turn
                player = 'O' if player == 'X' else 'X'

    # Draw the game board
    window.fill(WHITE)
    for row in range(3):
        for col in range(3):
            # Draw the X's and O's
            if board[row][col] == 'X':
                pygame.draw.line(window, RED, (col * WINDOW_WIDTH // 3 + 50, row * WINDOW_HEIGHT // 3 + 50),
                                 (col * WINDOW_WIDTH // 3 + WINDOW_WIDTH // 3 - 50, row * WINDOW_HEIGHT // 3 + WINDOW_HEIGHT // 3 - 50), 10)
                pygame.draw.line(window, RED, (col * WINDOW_WIDTH // 3 + 50, row * WINDOW_HEIGHT // 3 + WINDOW_HEIGHT // 3 - 50),
                                 (col * WINDOW_WIDTH // 3 + WINDOW_WIDTH // 3 - 50, row * WINDOW_HEIGHT // 3 + 50), 10)
            elif board[row][col] == 'O':
                pygame.draw.circle(window, BLUE, (col * WINDOW_WIDTH // 3 + WINDOW_WIDTH // 6,
                                   row * WINDOW_HEIGHT // 3 + WINDOW_HEIGHT // 6), WINDOW_WIDTH // 6 - 50, 10)
            # Draw the grid lines
            pygame.draw.rect(window, BLACK, (col * WINDOW_WIDTH // 3, row *
                             WINDOW_HEIGHT // 3, WINDOW_WIDTH // 3, WINDOW_HEIGHT // 3), 5)

    # Check for a winner
    for i in range(3):
        if board[i][0] == board[i][1] == board[i][2] != '':
            # Draw a line over the winning sequence
            pygame.draw.line(window, BLACK, (0, (i + 0.5) * WINDOW_HEIGHT // 3),
                             (WINDOW_WIDTH, (i + 0.5) * WINDOW_HEIGHT // 3), 5)
            # Store the winner
            winner = board[i][0]
            running = False
        elif board[0][i] == board[1][i] == board[2][i] != '':
            # Draw a line over the winning sequence
            pygame.draw.line(window, BLACK, ((i + 0.5) * WINDOW_WIDTH // 3, 0),
                             ((i + 0.5) * WINDOW_WIDTH // 3, WINDOW_HEIGHT), 5)
            # Store the winner
            winner = board[0][i]
            running = False
    if board[0][0] == board[1][1] == board[2][2] != '':
        # Draw a line over the winning sequence
        pygame.draw.line(window, BLACK, (50, 50),
                         (WINDOW_WIDTH - 50, WINDOW_HEIGHT - 50), 5)
        # Store the winner
        winner = board[0][0]
        running = False
    elif board[0][2] == board[1][1] == board[2][0] != '':
        # Draw a line over the winning sequence
        pygame.draw.line(window, BLACK, (WINDOW_WIDTH - 50, 50),
                         (50, WINDOW_HEIGHT - 50), 5)
        # Store the winner
        winner = board[0][2]
        running = False

    # Check if the game is a tie
    if all(all(row) for row in board) and running:
        winner = 'Tie'
        running = False

    # If it's the computer's turn, make a move
    if player == 'O' and running:
        # Choose the center cell if it's available
        if board[1][1] == '':
            row, col = 1, 1
        # Otherwise, choose a random corner cell
        else:
            corners = [(0, 0), (0, 2), (2, 0), (2, 2)]
            empty_corners = [
                corner for corner in corners if board[corner[0]][corner[1]] == '']
            if empty_corners:
                row, col = random.choice(empty_corners)
            # If all corners are taken, choose a random empty cell
            else:
                empty_cells = [(i, j) for i in range(3)
                               for j in range(3) if board[i][j] == '']
                row, col = random.choice(empty_cells)
        # Update the board
        board[row][col] = player
        # Switch the player turn
        player = 'X'

    # Update the display
    pygame.display.update()

# Delay before quitting Pygame
time.sleep(2)

# Clear the screen after X or O wins
window.fill(WHITE)
if winner == 'Tie':
    text = FONT.render('Tie game!', True, BLACK)
else:
    if winner == 'X':
        text = FONT.render('You Unlocked the Hint!', True, BLACK)
    else:
        text = FONT.render('Try again', True, BLACK)
window.blit(text, (WINDOW_WIDTH // 2 - text.get_width() //
                   2, WINDOW_HEIGHT // 2 - text.get_height() // 2))
pygame.display.update()

# Pause for 2 seconds before quitting Pygame
time.sleep(2)

# Quit Pygame
pygame.quit()
