import pygame
import random

# Initialize pygame
pygame.init()

# Set up the game window
window_width = 800
window_height = 600
window = pygame.display.set_mode((window_width, window_height))
RED = (255, 0, 0)
run = True
while run:
    pygame.draw.rect(window, RED, (400, 400, 20, 20), 0)
    window.fill(RED)
    pygame.display.update()
# pygame.display.set_caption("Find the Fox")

# # Define colors
# BLACK = (0, 0, 0)
# WHITE = (255, 255, 255)
# GREEN = (0, 255, 0)

# # Define hole properties
# hole_radius = 30
# hole_gap = 100
# hole_y = window_height // 2

# # Define fox properties
# fox_radius = 20
# fox_x = random.randint(hole_radius, window_width - hole_radius)
# fox_y = hole_y

# # Game loop
# running = True
# found_fox = False

# while running:
#     # Handle events
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             running = False
#         elif event.type == pygame.KEYDOWN:
#             if event.key == pygame.K_1:
#                 if fox_x == (window_width // 2) - (hole_gap + 2 * hole_radius):
#                     found_fox = True
#             elif event.key == pygame.K_2:
#                 if fox_x == (window_width // 2):
#                     found_fox = True
#             elif event.key == pygame.K_3:
#                 if fox_x == (window_width // 2) + (hole_gap + 2 * hole_radius):
#                     found_fox = True

#     # Clear the screen
#     window.fill(BLACK)

#     # Draw holes
#     for i in range(3):
#         hole_x = (window_width // 2) + (i - 1) * (hole_gap + 2 * hole_radius)
#         pygame.draw.circle(window, WHITE, (hole_x, hole_y), hole_radius)

#     # Draw fox
#     pygame.draw.circle(window, GREEN, (fox_x, fox_y), fox_radius)

#     # Update the display
#     pygame.display.flip()

#     # Move the fox during the night
#     if not found_fox:
#         if fox_x == (window_width // 2) - (hole_gap + 2 * hole_radius):
#             fox_x += hole_gap + 2 * hole_radius
#         elif fox_x == (window_width // 2) + (hole_gap + 2 * hole_radius):
#             fox_x -= hole_gap + 2 * hole_radius
#         else:
#             move_direction = random.choice([-1, 1])
#             fox_x += move_direction * (hole_gap + 2 * hole_radius)

# # Quit the game
# pygame.quit()
