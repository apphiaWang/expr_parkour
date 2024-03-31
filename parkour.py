import os
import sys
import math
import random
import pygame

WIDTH = 623
HEIGHT = 150

pygame.init()
pygame.mixer.init()
screen = pygame.display.set_mode( (WIDTH, HEIGHT) )
pygame.display.set_caption('Expression Parkour')

class BG:

	def __init__(self, x):
		self.width = WIDTH
		self.height = HEIGHT
		self.x = x
		self.y = 0
		self.set_texture()
		self.show()

	def update(self, dx):
		self.x += dx
		if self.x <= -WIDTH:
			self.x = WIDTH

	def show(self):
		screen.blit(self.texture, (self.x, self.y))

	def set_texture(self):
		path = os.path.join('assets/images/bg_new.png')
		self.texture = pygame.image.load(path)
		self.texture = pygame.transform.scale(self.texture, (self.width, self.height))

class MC:
	def __init__(self):
		self.width = 44
		self.height = 44
		self.x = 10
		self.y = 80
		self.texture = 'run1.png'
		self.texture_num = 0
		self.dy = 2
		self.gravity = 1.2
		self.onground = True
		self.jumping = False
		self.jump_stop = 10
		self.falling = False
		self.fall_stop = self.y
		self.shoveling = False
		self.shovel_stop = 0
		self.dashing = False
		self.dash_stop = 0
		self.set_texture()
		self.set_sound()
		self.show()

	def update(self, loops):
		# jumping
		if self.jumping:
			self.y -= self.dy
			if self.y <= self.jump_stop:
				self.fall()
		
		# falling after jumping
		elif self.falling:
			self.y += self.gravity * self.dy
			if self.y >= self.fall_stop:
				self.stop()

		# shoveling
		elif self.shoveling:
			if loops > self.shovel_stop:
				self.shoveling = False
				self.reset_size()
			else:
				self.texture_num = (self.texture_num + 1) % 3
				self.texture = f'shovel{self.texture_num}.png'
				self.set_texture()
		
		# stop dashing on time
		elif self.dashing:
			if loops > self.dash_stop:
				self.dashing = False
				self.reset_size()
			else:
				self.texture_num = (self.texture_num + 1) % 6
				self.texture = f'dash{self.texture_num}.png'
				self.set_texture()
		# running
		elif self.onground and loops % 4 == 0:
			self.texture_num = (self.texture_num + 1) % 3
			self.texture = f'run{self.texture_num}.png'
			self.set_texture()

	def show(self):
		screen.blit(self.texture, (self.x, self.y))

	def set_texture(self):
		path = os.path.join(f'assets/images/{self.texture}')
		self.texture = pygame.image.load(path)
		self.texture = pygame.transform.scale(self.texture, (self.width, self.height))

	def set_sound(self):
		path = os.path.join('assets/sounds/jump.wav')
		self.sound = pygame.mixer.Sound(path)

	def shovel(self, loops):
		# self.sound.play()
		self.shoveling = True
		self.shovel_stop = loops + 60
		self.shovel_resize()

	def dash_resize(self):
		self.width = 120
		self.height = 90
		self.y = 40

	def shovel_resize(self):
		self.width = 60
		self.height = 30
		self.y = 99

	def reset_size(self):
		self.width = 44
		self.height = 44
		self.y = 80

	def dash(self, loops):
		# self.sound.play()
		self.dashing = True
		self.dash_stop = loops + 120
		self.dash_resize()

	def jump(self):
		self.sound.play()
		self.jumping = True
		self.onground = False

	def fall(self):
		self.jumping = False
		self.falling = True

	def stop(self):
		self.falling = False
		self.onground = True

class Block:

	def __init__(self, x):
		self.width = 34
		self.height = 34
		self.x = x
		self.y = 100
		self.set_texture()
		self.show()

	def update(self, dx):
		self.x += dx	

	def show(self):
		screen.blit(self.texture, (self.x, self.y))

	def set_texture(self):
		path = os.path.join('assets/images/block.png')
		self.texture = pygame.image.load(path)
		self.texture = pygame.transform.scale(self.texture, (self.width, self.height))

class Wall:	
	def __init__(self, x):
		self.width = 30
		self.height = 55
		self.x = x
		self.y = 76
		self.set_texture()
		self.show()

	def update(self, dx):
		self.x += dx	

	def show(self):
		screen.blit(self.texture, (self.x, self.y))

	def set_texture(self):
		path = os.path.join('assets/images/wall1.png')
		self.texture = pygame.image.load(path)
		self.texture = pygame.transform.scale(self.texture, (self.width, self.height))

class Fense:
	def __init__(self, x):
		self.width = 30
		self.height = 80
		self.x = x
		self.y = 16
		self.set_texture()
		self.show()

	def update(self, dx):
		self.x += dx	

	def show(self):
		screen.blit(self.texture, (self.x, self.y))

	def set_texture(self):
		path = os.path.join('assets/images/wall1.png')
		self.texture = pygame.image.load(path)
		self.texture = pygame.transform.scale(self.texture, (self.width, self.height))

class Collision:
	def between(self, obj1, obj2):
		DELTA = -10
		if obj2.x - obj1.x - obj1.width >= DELTA:
			return False
		if obj1.x - obj2.x - obj2.width >= DELTA:
			return False
		if obj2.y - obj1.y - obj1.height >= DELTA:
			return False
		if obj1.y - obj2.y - obj2.height >= DELTA:
			return False
		return True

class Score:

	def __init__(self, hs):
		self.hs = hs
		self.act = 0
		self.font = pygame.font.SysFont('monospace', 18)
		self.color = (0, 0, 0)
		self.set_sound()
		self.show()

	def update(self, loops):
		self.act = loops // 10
		self.check_hs()
		self.check_sound()

	def show(self):
		self.lbl = self.font.render(f'HI {self.hs} {self.act}', 1, self.color)
		lbl_width = self.lbl.get_rect().width
		screen.blit(self.lbl, (WIDTH - lbl_width - 10, 10))

	def set_sound(self):
		path = os.path.join('assets/sounds/point.wav')
		self.sound = pygame.mixer.Sound(path)

	def check_hs(self):
		if self.act >= self.hs:
			self.hs = self.act

	def check_sound(self):
		if self.act % 100 == 0 and self.act != 0:
			self.sound.play()

class Game:
	def __init__(self, hs=0):
		self.bg = [BG(x=0), BG(x=WIDTH)]
		self.mc = MC()
		self.obstacles = []
		self.collision = Collision()
		self.score = Score(hs)
		self.speed = 3
		self.playing = False
		self.set_sound()
		self.set_labels()
		self.spawn_obstacle()

	def set_labels(self):
		big_font = pygame.font.SysFont('monospace', 24, bold=True)
		small_font = pygame.font.SysFont('monospace', 18)
		self.big_lbl = big_font.render(f'G A M E  O V E R', 1, (0, 0, 0))
		self.small_lbl = small_font.render(f'press up key to restart', 1, (0, 0, 0))

	def set_sound(self):
		path = os.path.join('assets/sounds/die.wav')
		self.sound = pygame.mixer.Sound(path)

	def start(self):
		self.playing = True

	def over(self):
		self.sound.play()
		screen.blit(self.big_lbl, (WIDTH // 2 - self.big_lbl.get_width() // 2, HEIGHT // 4))
		screen.blit(self.small_lbl, (WIDTH // 2 - self.small_lbl.get_width() // 2, HEIGHT // 2))
		self.playing = False

	def tospawn(self, loops):
		return loops % 100 == 0

	def spawn_obstacle(self):
		# list obstacles
		if len(self.obstacles) > 0:
			prev_obstacle = self.obstacles[-1]
			MIN_GAP = 100
			x = random.randint(prev_obstacle.x + self.mc.width + MIN_GAP, WIDTH + prev_obstacle.x + self.mc.width + MIN_GAP)

		# empty list
		else:
			x = random.randint(WIDTH + 100, 1000)
		
		obstacle_type = random.randint(0,4)
		obstacle = None
		if obstacle_type >= 0 and obstacle_type < 2 :
			# create the new block
			obstacle = Block(x)
		elif obstacle_type == 2:
			# create the new wall
			obstacle = Wall(x)
		else:
			# create the new fense
			obstacle = Fense(x)
		self.obstacles.append(obstacle)
	
	def restart(self):
		print('restart game')
		self.__init__(hs=self.score.hs)

def main():

	# objects
	game = Game()
	main_character = game.mc

	# variables
	clock = pygame.time.Clock()
	loops = 0
	over = False

	# mainloop
	while True:

		if game.playing:

			loops += 1

			# --- BG ---
			for bg in game.bg:
				bg.update(-game.speed)
				bg.show()

			# --- Main character ---
			main_character.update(loops)
			main_character.show()

			# --- obstacles ---
			if game.tospawn(loops):
				game.spawn_obstacle()

			for obstacle in game.obstacles:
				obstacle.update(-game.speed)
				obstacle.show()

				# collision
				if not main_character.dashing and game.collision.between(main_character, obstacle):
					if (obstacle.__class__.__name__ == 'Block'):
						print('clear score')
					elif obstacle.__class__.__name__ == 'Wall':
						over=True
					elif obstacle.__class__.__name__ == 'Fense':
						over=True

			if over:
				game.over()

			# -- score ---
			game.score.update(loops)
			game.score.show()

		# events
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				pygame.quit()
				sys.exit()
			
			if event.type == pygame.KEYDOWN:
				if event.key == pygame.K_SPACE:
					if not over:
						if main_character.onground:
							main_character.jump()
						if not game.playing:
							game.start()
				if event.key == pygame.K_LSHIFT:
					if not over:
						main_character.shovel(loops)
				if event.key == pygame.K_LCTRL:
					if not over:
						main_character.dash(loops)
				if event.key == pygame.K_UP:
					game.restart()
					main_character = game.mc
					loops = 0
					over = False

		clock.tick(80)
		pygame.display.update()

main()