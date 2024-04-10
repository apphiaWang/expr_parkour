import os
import sys
import cv2
import random
import pygame
import pygame.camera
from deepface import DeepFace
from au_model import create_model, get_expression_confidence_scores
import numpy as np

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 380

WIDTH = 800
HEIGHT = 200

CAM_WIDTH = 240
CAM_HEIGHT = 180

GAME_SPEED = 5

delta = 18

# init camera through opencv
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
cap.set(3, 640) 
cap.set(4, 480)

# init pygame
pygame.init()
pygame.mixer.init()
screen = pygame.display.set_mode( (SCREEN_WIDTH, SCREEN_HEIGHT) )

pygame.display.set_caption('Expression Parkour')

class AU_Model:

	def __init__(self):
		return

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

class Instruction:
	def __init__(self): 
		self.width = SCREEN_WIDTH - CAM_WIDTH
		self.height = CAM_HEIGHT
		self.x = CAM_WIDTH
		self.y = HEIGHT
		self.font = pygame.font.SysFont('monospace', 14)
		self.color = (0, 0, 0)
		self.show()

	def show(self):
		screen.fill((175, 200, 173), rect = (self.x, self.y, self.width, self.height))
		instr = """ How To Play:
		- Make a SMILEY face to jump over obstacle like Rock and Bush
		- Make a SURPRISE face to do a slide shovel below a ghost fire
		- When the enegery bar is full, make an ANGRY face to dash
		Rules:
		- You will die if you hit a rock or a ghost fire
		- You won't die if you hit a bush, but your energy will be cleared,
		and the number of enemy chasing you will increase by 1
		- When there are 3 enemies chasing you, you will be caught
		- You are unrivaled during dash
		"""
		for idx, line in enumerate(instr.split('\n')):
			lbl = self.font.render(line.strip(), 1, self.color)
			screen.blit(lbl, (self.x + 10, self.y + 5 + 17 * idx))

class ExpressionLabel:
	def __init__(self): 
		self.x = 10
		self.y = HEIGHT - 30
		self.font = pygame.font.SysFont('monospace', 18)
		self.expression = 'neutral'
		self.color = (0, 0, 0)
		self.show()

	def update(self, expr):
		self.expression = expr
		self.show()

	def show(self):
		# screen.fill((175, 200, 173), rect = (self.x, self.y, self.width, self.height))
		lbl = self.font.render(self.expression, 1, self.color)
		screen.blit(lbl, (self.x, self.y))

class Enemy:
	def __init__(self):
		self.width = 55
		self.height = 50
		self.x = 0
		self.y = HEIGHT - self.height - 30
		self.texture_num = 0
		self.number = 1
		self.set_texture()
		self.show()

	def update(self, loops):
		self.texture_num = int(loops/10) % 3
		self.set_texture()

	def show(self):
		screen.blit(self.texture, (self.x, self.y))

	def add_monster(self):
		if self.number < 3:
			self.number += 1
			self.x += 10
			self.set_texture()
			# TODO


	def reduce_monster(self):
		if self.number > 1:
			self.number -= 1
			self.x -= 10
			self.set_texture()
	
	def reset_monster(self):
		self.number = 1
		self.x = 10
		self.set_texture()
	
	def set_texture(self):
		path = os.path.join(f'assets/images/ghost{self.number}_{self.texture_num}.png')
		self.texture = pygame.image.load(path)
		self.texture = pygame.transform.scale(self.texture, (self.width, self.height))

class MC:
	def __init__(self):
		self.width = 48
		self.height = 60
		self.x = 80
		self.y = HEIGHT - 60 - delta
		self.texture_img = 'idle.png'
		self.texture_num = 0
		self.dy = 2
		self.gravity = 1.2
		self.onground = True
		self.jumping = False
		self.jump_stop = 40
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
				self.texture_num = 0
				self.texture_img = 'shovel.png'
				self.set_texture()
		
		# stop dashing on time
		elif self.dashing:
			if loops > self.dash_stop:
				self.dashing = False
				self.reset_size()
			else:
				self.texture_num = (self.texture_num + 1) % 6
				self.texture_img = f'dash{self.texture_num}.png'
				self.set_texture()
		# running
		elif self.onground and loops % 4 == 0:
			self.texture_num = (self.texture_num + 1) % 3
			self.texture_img = f'run{self.texture_num}.png'
			self.set_texture()

	def show(self):
		screen.blit(self.texture, (self.x, self.y))

	def set_texture(self):
		path = os.path.join(f'assets/images/{self.texture_img}')
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
		self.width = 90
		self.height = 60
		self.y = HEIGHT - 60 - delta

	def shovel_resize(self):
		self.width = 60
		self.height = 40
		self.y = HEIGHT - 40 - delta

	def reset_size(self):
		self.width = 48
		self.height = 60
		self.y = HEIGHT - 60 - delta

	def dash(self, loops):
		# self.sound.play()
		self.dashing = True
		self.dash_stop = loops + 120
		self.dash_resize()

	def jump(self):
		self.sound.play()
		self.jumping = True
		self.onground = False
		self.texture_img = 'jump.png'
		self.set_texture()

	def fall(self):
		self.jumping = False
		self.falling = True

	def in_action(self):
		return self.jumping or self.falling or self.shoveling or self.dashing
	
	def stop(self):
		self.falling = False
		self.onground = True

class Bush:
	def __init__(self, x):
		self.width = 25
		self.height = 25
		self.x = x
		self.y = HEIGHT - 25 - delta
		self.set_texture()
		self.show()

	def update(self, dx):
		self.x += dx	

	def show(self):
		screen.blit(self.texture, (self.x, self.y))

	def set_texture(self):
		path = os.path.join('assets/images/bush.png')
		self.texture = pygame.image.load(path)
		self.texture = pygame.transform.scale(self.texture, (self.width, self.height))

class Rock:	
	def __init__(self, x):
		self.width = 30
		self.height = 45
		self.x = x
		self.y = HEIGHT - 45 - delta
		self.broken = False
		self.set_texture()
		self.show()

	def update(self, dx):
		self.x += dx	

	def show(self):
		screen.blit(self.texture, (self.x, self.y))

	def broken_by_actor(self):
		self.broken = True

	def set_texture(self):
		path = os.path.join('assets/images/rock.png')
		self.texture = pygame.image.load(path)
		self.texture = pygame.transform.scale(self.texture, (self.width, self.height))

class Fire:
	def __init__(self, x):
		self.width = 55
		self.height = 75
		self.x = x
		self.y = HEIGHT - 75 - 55
		self.set_texture()
		self.show()

	def update(self, dx):
		self.x += dx	

	def show(self):
		screen.blit(self.texture, (self.x, self.y))

	def set_texture(self):
		path = os.path.join('assets/images/fire.png')
		self.texture = pygame.image.load(path)
		self.texture = pygame.transform.scale(self.texture, (self.width, self.height))

class Collision:
	def between(self, obj1, obj2):
		DELTA_H = -22
		DELTA_V = -12
		if obj2.x - obj1.x - obj1.width >= DELTA_H:
			return False
		if obj1.x - obj2.x - obj2.width >= DELTA_H:
			return False
		if obj2.y - obj1.y - obj1.height >= DELTA_V:
			return False
		if obj1.y - obj2.y - obj2.height >= DELTA_V:
			return False
		return True

class EnergyBar:
	def __init__(self):
		self.value = 0
		self.delta = 0
		self.width = 100
		self.height = 30
		self.x = WIDTH - 120
		self.y = 10
		self.set_texture()
		self.set_sound()
		self.show()

	def update(self):
		if (self.value < 5):
			self.delta += 1
			if self.delta == 100:
				self.value += 1
				if self.value == 5:
					self.full_sound.play()
				else:
					self.point_sound.play()
				self.delta = 0
				self.set_texture()

	def clear_energy(self, active=False):
		self.value = 0
		self.clear_sound.play()
		self.set_texture()
	
	def show(self):
		screen.blit(self.texture, (self.x, self.y))

	def set_texture(self):
		path = os.path.join(f'assets/images/energy_{self.value}.png')
		self.texture = pygame.image.load(path)
		self.texture = pygame.transform.scale(self.texture, (self.width, self.height))

	def set_sound(self):
		path = os.path.join('assets/sounds/point.wav')
		self.point_sound = pygame.mixer.Sound(path)
		path = os.path.join('assets/sounds/full.mp3')
		self.full_sound = pygame.mixer.Sound(path)
		path = os.path.join('assets/sounds/clear.mp3')
		self.clear_sound = pygame.mixer.Sound(path)

class Game:
	def __init__(self, hs=0):
		self.bg = [BG(x=0), BG(x=WIDTH)]
		self.instruction = Instruction()
		self.mc = MC()
		self.enemy = Enemy()
		self.energy_bar = EnergyBar()
		self.obstacles = []
		self.collision = Collision()
		self.speed = GAME_SPEED
		self.playing = False
		self.set_sound()
		self.set_labels()
		self.spawn_obstacle(0)

	def set_labels(self):
		big_font = pygame.font.SysFont('monospace', 24, bold=True)
		small_font = pygame.font.SysFont('monospace', 18)
		self.dead_lbl = big_font.render(f'G A M E  O V E R', 1, (0, 0, 0))
		self.restart_lbl = small_font.render(f'press up key to restart', 1, (0, 0, 0))
		self.start_lbl = big_font.render(f'press space to start game', 1, (0, 0, 0))

	def show_start_msg(self):
		screen.blit(self.start_lbl, (WIDTH // 2 - self.start_lbl.get_width() // 2, HEIGHT // 4))

	def set_sound(self):
		path = os.path.join('assets/sounds/die.wav')
		self.sound = pygame.mixer.Sound(path)

	def start(self):
		self.playing = True

	def over(self):
		self.sound.play()
		screen.blit(self.dead_lbl, (WIDTH // 2 - self.dead_lbl.get_width() // 2, HEIGHT // 4))
		screen.blit(self.restart_lbl, (WIDTH // 2 - self.restart_lbl.get_width() // 2, HEIGHT // 2))
		self.energy_bar.clear_energy()
		self.enemy.reset_monster()
		self.playing = False

	def tospawn(self, loops):
		return loops % 100 == 0

	def spawn_obstacle(self, loops):
		MIN_GAP = 200
		# list obstacles
		if len(self.obstacles) > 0:
			prev_obstacle = self.obstacles[-1]
			x = random.randint(prev_obstacle.x + self.mc.width + MIN_GAP, WIDTH + prev_obstacle.x + self.mc.width + MIN_GAP)
		else:
			x = 500

		obstacle = None
		if loops < 50:
			# only spawn bushes at the early stage
			obstacle = Bush(x)
		else:
			obstacle_type = random.randint(0,10)
			if obstacle_type >= 0 and obstacle_type < 3 :
				# create the new bush
				obstacle = Bush(x)
			elif obstacle_type >= 4 and obstacle_type < 6:
				# create the new rock
				obstacle = Rock(x)
			else:
				# create the new fire
				obstacle = Fire(x)
		self.obstacles.append(obstacle)
	
	def reset(self):
		self.__init__()

def get_dominant_expression(deepface_result, au_result):
	DEEPFACE_W = 0.95
	AU_W = 0.05
	target_exprs = ['happy', 'surprise', 'angry', 'neutral']
	final_scores = [] 
	for expr in target_exprs:
		final_scores.append(deepface_result[expr] * DEEPFACE_W + au_result[expr] * AU_W)
	return target_exprs[np.argmax(final_scores)]

def main():

	# initialize au detector
	au_detector = create_model()
	# objects
	game = Game()
	main_character = game.mc
	enemy = game.enemy 
	energy_bar = game.energy_bar

	expr_label = ExpressionLabel()
	expr = 'neutral'

	# variables
	loops = 0
	over = False
	hit_bush_start = -1

	game.show_start_msg()

	# mainloop
	while True:
		# read webcam images		
		success, frame = cap.read()
		if not success:
			print("Error: failed reading image from webcam")
			break
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		
		# predict expression
		if loops % 20 == 1 and game.playing and not main_character.in_action():
			# predict emotion with deepface
			deepface_result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
			deepface_result = deepface_result[0]['emotion']
			
			# face detection using OpenCV
			faces = face_cascade.detectMultiScale(frame, 1.1, 4)
			# AU recognition for face
			if len(faces) == 0:
				continue
			x, y, w, h = faces[0]
			face_img = frame[y:y+h, x:x+w]
			face_img = cv2.resize(face_img, (224, 224))
			face_img = np.expand_dims(face_img, axis=0)
			
			# detect au
			y_predict = au_detector.predict(face_img)
			ind = np.where(y_predict[1] > 0.8)[1]
			au_result = get_expression_confidence_scores(ind)

			expr = get_dominant_expression(deepface_result, au_result)
			if expr == "happy":
				if main_character.onground:
					main_character.jump()
			elif expr == "surprise":
				main_character.shovel(loops)
			# elif expr == "neutral":
			# 	continue
			elif expr == "angry" or expr == "disgust":
				if energy_bar.value == 5:
					main_character.dash(loops)
					energy_bar.clear_energy()
					enemy.reduce_monster()
		
		# display webcam image
		frame = cv2.flip(frame, 1) 
		frame = pygame.image.frombuffer(frame.tobytes(), frame.shape[1::-1], "RGB")
		frame = pygame.transform.scale(frame, (240, 180))
		screen.blit(frame, (0, HEIGHT))

		if game.playing:

			for bg in game.bg:
				bg.update(-game.speed)
				bg.show()
		
			game.instruction.show()

			main_character.update(loops)
			main_character.show()

			enemy.update(loops)
			enemy.show()

			energy_bar.update()
			energy_bar.show()

			if game.tospawn(loops):
				game.spawn_obstacle(loops)

			for obstacle in game.obstacles:
				obstacle.update(-game.speed)
				obstacle.show()

				hitting_bush  = hit_bush_start > 0 and (loops - hit_bush_start) <= main_character.width + 25
				# checking collision
				if (not hitting_bush
					and loops % 5 == 1 
					and not main_character.dashing 
					and game.collision.between(main_character, obstacle)):
					if (obstacle.__class__.__name__ == 'Bush'):
						energy_bar.clear_energy()
						enemy.add_monster()
						hit_bush_start = loops
						# if (enemy.number == 3):
						# 	over=True
					else:
						over=True

			# if over:
			# 	game.over()

			loops += 1

		expr_label.update(expr=expr)
		
		# normal control using keys/mouse input
		for event in pygame.event.get():
			if event.type == pygame.KEYDOWN and event.key == pygame.K_DOWN:
				over = True
				game.over()

			if event.type == pygame.QUIT:
				pygame.quit()
				sys.exit()
			
			if event.type == pygame.KEYDOWN:
				if event.key == pygame.K_SPACE:
					if not over:
						if not game.playing:
							game.start()
						elif main_character.onground:
							main_character.jump()
				if event.key == pygame.K_LSHIFT:
					if not over:
						main_character.shovel(loops)
				if event.key == pygame.K_LCTRL and energy_bar.value == 5:
					if not over:
						main_character.dash(loops)
						energy_bar.clear_energy()
						enemy.reduce_monster()
				if event.key == pygame.K_UP:
					game.reset()
					main_character = game.mc
					loops = 0
					over = False
					game.start()

		pygame.display.update()
		

main()

# release resources
cap.release()
pygame.quit()

